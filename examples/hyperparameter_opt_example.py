from __future__ import annotations

import os

import torch
from torch.utils.data import TensorDataset

from utils.config import ATTACK_TYPES
from training.hyperparameter_opt import (
    optimize_hyperparameters,
    save_best_params,
    visualize_optimization_history,
    visualize_parameter_importance,
    visualize_parameter_relationships,
)


def run_example() -> None:
    try:
        torch.manual_seed(0)
        features = torch.randn(240, 41)
        labels = torch.randint(0, len(ATTACK_TYPES), (240,))

        train_dataset = TensorDataset(features[:200], labels[:200])
        val_dataset = TensorDataset(features[200:], labels[200:])

        result = optimize_hyperparameters(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            n_trials=5,
            max_epochs=3,
            patience=1,
            device=torch.device("cpu"),
        )

        output_dir = "outputs/results/hpo_example"
        os.makedirs(output_dir, exist_ok=True)

        visualize_optimization_history(
            result["study"], os.path.join(output_dir, "optimization_history.png")
        )
        visualize_parameter_importance(
            result["study"], os.path.join(output_dir, "param_importance.png")
        )
        visualize_parameter_relationships(
            result["study"], os.path.join(output_dir, "param_relationships.png")
        )
        save_best_params(
            result["best_params"], os.path.join(output_dir, "best_params.yaml")
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError("超参数优化示例执行失败") from exc


def main() -> None:
    try:
        run_example()
    except RuntimeError as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
