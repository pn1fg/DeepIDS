from __future__ import annotations

from typing import Any, Callable, Dict, Optional, cast
import importlib
import os

import matplotlib
from matplotlib.figure import Figure
import optuna
import torch
from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
)
from torch.utils.data import DataLoader, Dataset

from models.deep_learning import build_model
from training.callbacks import EarlyStopping
from training.train import train_epoch, validate_epoch
from utils.config import ATTACK_TYPES

matplotlib.use("Agg")


def _extract_labels(dataset: Dataset) -> torch.Tensor:
    if hasattr(dataset, "labels"):
        labels = getattr(dataset, "labels")
        return torch.as_tensor(labels)
    if hasattr(dataset, "targets"):
        labels = getattr(dataset, "targets")
        return torch.as_tensor(labels)
    batches = []
    loader = DataLoader(dataset, batch_size=1024, shuffle=False)
    for _, batch_labels in loader:
        batches.append(batch_labels.detach().cpu())
    if not batches:
        return torch.empty(0, dtype=torch.long)
    return torch.cat(batches, dim=0)


def _compute_class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    if num_classes <= 0:
        raise ValueError("num_classes必须为正整数")
    counts = torch.bincount(labels.to(torch.int64), minlength=num_classes).float()
    counts = torch.where(counts > 0, counts, torch.ones_like(counts))
    total = counts.sum()
    return total / (counts * num_classes)


def _resolve_architecture(architecture: Optional[str], trial: optuna.trial.Trial) -> str:
    if architecture:
        return str(architecture).lower()
    return str(
        trial.suggest_categorical("architecture", ["mlp", "lstm", "transformer", "cnn"])
    ).lower()


def _suggest_model_config(
    trial: optuna.trial.Trial,
    architecture: str,
    input_dim: int,
    output_dim: int,
) -> Dict[str, Any]:
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    if architecture in {"mlp", "fc", "dense"}:
        hidden_dim = trial.suggest_int("hidden_dim", 128, 512, step=32)
        num_layers = trial.suggest_int("num_layers", 2, 4)
        hidden_dims = [max(hidden_dim // (2**i), 32) for i in range(num_layers)]
        model_cfg: Dict[str, Any] = {
            "architecture": "mlp",
            "input_dim": input_dim,
            "output_dim": output_dim,
            "dropout": dropout,
            "hidden_dims": hidden_dims,
            "activation": "ReLU",
            "batch_norm": True,
        }
        return model_cfg
    if architecture == "lstm":
        hidden_dim = trial.suggest_int("hidden_dim", 64, 256, step=32)
        num_layers = trial.suggest_int("num_layers", 1, 3)
        bidirectional = bool(trial.suggest_categorical("bidirectional", [0, 1]))
        return {
            "architecture": "lstm",
            "input_dim": input_dim,
            "output_dim": output_dim,
            "dropout": dropout,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "bidirectional": bidirectional,
        }
    if architecture == "transformer":
        model_dim = int(trial.suggest_categorical("model_dim", [32, 64, 128]))
        candidate_heads = [h for h in [2, 4, 8] if model_dim % h == 0]
        num_heads = int(trial.suggest_categorical("num_heads", candidate_heads))
        num_layers = trial.suggest_int("num_layers", 1, 3)
        max_len = trial.suggest_int("max_len", 64, 256, step=64)
        return {
            "architecture": "transformer",
            "input_dim": input_dim,
            "output_dim": output_dim,
            "dropout": dropout,
            "model_dim": model_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "max_len": max_len,
        }
    if architecture == "cnn":
        channels_1 = trial.suggest_int("channels_1", 16, 64, step=16)
        channels_2 = trial.suggest_int("channels_2", 32, 128, step=32)
        kernel_size = trial.suggest_int("kernel_size", 3, 5)
        return {
            "architecture": "cnn",
            "input_dim": input_dim,
            "output_dim": output_dim,
            "dropout": dropout,
            "channels": [channels_1, channels_2],
            "kernel_size": kernel_size,
        }
    raise ValueError("不支持的architecture配置")


def define_objective(
    train_dataset: Dataset,
    val_dataset: Dataset,
    input_dim: int = 41,
    output_dim: int = len(ATTACK_TYPES),
    device: Optional[torch.device] = None,
    max_epochs: int = 30,
    patience: int = 5,
    num_workers: int = 0,
    seed: int = 42,
    architecture: Optional[str] = None,
    use_class_weights: bool = False,
) -> Callable[[optuna.trial.Trial], float]:
    if max_epochs <= 0:
        raise ValueError("max_epochs必须大于0")
    if patience <= 0:
        raise ValueError("patience必须大于0")
    if input_dim <= 0 or output_dim <= 0:
        raise ValueError("input_dim和output_dim必须为正整数")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = None
    if use_class_weights:
        labels = _extract_labels(train_dataset)
        class_weights = _compute_class_weights(labels, output_dim).to(device)

    def objective(trial: optuna.trial.Trial) -> float:
        torch.manual_seed(seed)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_int("batch_size", 16, 64, step=8)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)

        arch = _resolve_architecture(architecture, trial)
        model_cfg = _suggest_model_config(trial, arch, input_dim, output_dim)
        model = build_model({"model": model_cfg}, device=str(device))

        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        if class_weights is not None:
            loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        else:
            loss_fn = torch.nn.CrossEntropyLoss()

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        early_stopping = EarlyStopping(patience=patience, mode="max")
        best_val = 0.0

        for epoch in range(max_epochs):
            train_epoch(
                model=model,
                data_loader=train_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=device,
            )
            val_metrics = validate_epoch(
                model=model,
                data_loader=val_loader,
                loss_fn=loss_fn,
                device=device,
            )
            val_acc = float(val_metrics["accuracy"])
            if val_acc > best_val:
                best_val = val_acc

            trial.report(val_acc, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if early_stopping.step(val_acc):
                break

        return best_val

    return objective


def optimize_hyperparameters(
    train_dataset: Dataset,
    val_dataset: Dataset,
    input_dim: int = 41,
    output_dim: int = len(ATTACK_TYPES),
    n_trials: int = 50,
    max_epochs: int = 30,
    patience: int = 5,
    device: Optional[torch.device] = None,
    num_workers: int = 0,
    seed: int = 42,
    timeout: Optional[int] = None,
    study_name: Optional[str] = None,
    architecture: Optional[str] = None,
    use_class_weights: bool = False,
) -> Dict[str, Any]:
    if n_trials <= 0:
        raise ValueError("n_trials必须大于0")

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
    )

    objective = define_objective(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        input_dim=input_dim,
        output_dim=output_dim,
        device=device,
        max_epochs=max_epochs,
        patience=patience,
        num_workers=num_workers,
        seed=seed,
        architecture=architecture,
        use_class_weights=use_class_weights,
    )

    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    return {
        "best_params": dict(study.best_params),
        "best_value": float(study.best_value),
        "study": study,
    }


def visualize_optimization_history(study: optuna.study.Study, output_path: str) -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    ax = plot_optimization_history(study)
    try:
        figure = cast(Figure, ax.get_figure())
        figure.savefig(output_path)
    except Exception as exc:
        raise IOError("保存优化历史失败") from exc
    return output_path


def visualize_parameter_importance(study: optuna.study.Study, output_path: str) -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    try:
        ax = plot_param_importances(study)
        figure = cast(Figure, ax.get_figure())
    except Exception:
        figure = Figure()
    try:
        figure.savefig(output_path)
    except Exception as exc:
        raise IOError("保存参数重要性失败") from exc
    return output_path


def visualize_parameter_relationships(
    study: optuna.study.Study, output_path: str
) -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    ax = plot_parallel_coordinate(study)
    try:
        figure = cast(Figure, ax.get_figure())
        figure.savefig(output_path)
    except Exception as exc:
        raise IOError("保存参数关系图失败") from exc
    return output_path


def save_best_params(best_params: Dict[str, Any], output_path: str) -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    yaml_module = importlib.import_module("yaml")
    try:
        with open(output_path, "w", encoding="utf-8") as handle:
            yaml_module.safe_dump(best_params, handle, sort_keys=True)
    except OSError as exc:
        raise IOError("保存最优参数失败") from exc
    return output_path
