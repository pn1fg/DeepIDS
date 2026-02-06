"""
训练流程模块。

提供:
- train_epoch: 单个epoch训练
- validate_epoch: 单个epoch验证
- train: 完整训练循环，支持早停与最佳模型保存
- train_loop: 兼容接口的训练循环包装

使用示例:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from models.deep_learning import DeepLearningClassifier
    from training.train import train

    x = torch.randn(100, 41)
    y = torch.randint(0, 22, (100,))
    train_loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(x, y), batch_size=32)

    model = DeepLearningClassifier(41, [256, 128, 64], 22, 0.3, "ReLU", True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        max_epochs=5,
        patience=2,
        device=torch.device("cpu"),
        save_path="outputs/models/best.pt",
        val_interval=1,
    )

错误处理说明:
    - 训练/验证过程中如遇到数据或计算错误，应捕获并抛出RuntimeError。
    - 保存模型失败时应抛出IOError，供上层日志系统处理。
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple, Optional, List

import torch
from torch import nn
from torch.nn import functional as F

from models.deep_learning import DeepLearningClassifier
from training.callbacks import EarlyStopping, ModelCheckpoint


def train_epoch(
    model: nn.Module,
    data_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    训练单个epoch。

    Args:
        model: 待训练模型。
        data_loader: 训练数据加载器，返回(features, labels)。
        optimizer: 优化器。
        loss_fn: 损失函数。
        device: 训练设备。

    Returns:
        包含平均损失与准确率的字典。
    """
    model.train(True)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    try:
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += int((preds == labels).sum().item())
            total_samples += int(labels.size(0))
    except Exception as exc:
        raise RuntimeError("训练过程发生异常") from exc

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return {"loss": avg_loss, "accuracy": accuracy}


def validate_epoch(
    model: nn.Module,
    data_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    loss_fn: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    验证单个epoch。

    Args:
        model: 待评估模型。
        data_loader: 验证数据加载器，返回(features, labels)。
        loss_fn: 损失函数。
        device: 验证设备。

    Returns:
        包含平均损失与准确率的字典。
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    try:
        with torch.no_grad():
            for features, labels in data_loader:
                features = features.to(device)
                labels = labels.to(device)
                logits = model(features)
                loss = loss_fn(logits, labels)
                total_loss += float(loss.item()) * labels.size(0)
                preds = torch.argmax(logits, dim=1)
                total_correct += int((preds == labels).sum().item())
                total_samples += int(labels.size(0))
    except Exception as exc:
        raise RuntimeError("验证过程发生异常") from exc

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return {"loss": avg_loss, "accuracy": accuracy}


def distill_epoch(
    student: nn.Module,
    teacher: nn.Module,
    data_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    hard_loss_fn: nn.Module,
    device: torch.device,
    temperature: float = 2.0,
    alpha: float = 0.5,
) -> Dict[str, float]:
    if temperature <= 0.0:
        raise ValueError("temperature必须大于0")
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("alpha必须在0到1之间")
    student.train(True)
    teacher.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    try:
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            student_logits = student(features)
            with torch.no_grad():
                teacher_logits = teacher(features)
            hard_loss = hard_loss_fn(student_logits, labels)
            soft_student = F.log_softmax(student_logits / temperature, dim=1)
            soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
            distill_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean")
            distill_loss = distill_loss * (temperature**2)
            loss = alpha * distill_loss + (1.0 - alpha) * hard_loss
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * labels.size(0)
            preds = torch.argmax(student_logits, dim=1)
            total_correct += int((preds == labels).sum().item())
            total_samples += int(labels.size(0))
    except Exception as exc:
        raise RuntimeError("蒸馏训练过程发生异常") from exc

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return {"loss": avg_loss, "accuracy": accuracy}


def distill_train(
    student: nn.Module,
    teacher: nn.Module,
    train_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    val_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    optimizer: Optional[torch.optim.Optimizer] = None,
    hard_loss_fn: Optional[nn.Module] = None,
    max_epochs: int = 100,
    patience: int = 10,
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None,
    val_interval: int = 1,
    temperature: float = 2.0,
    alpha: float = 0.5,
) -> Dict[str, List[float]]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student.to(device)
    teacher.to(device)
    if optimizer is None:
        optimizer = torch.optim.Adam(student.parameters(), lr=0.001, weight_decay=1e-5)
    if hard_loss_fn is None:
        hard_loss_fn = nn.CrossEntropyLoss()

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    early_stopping = EarlyStopping(patience=patience, mode="max")
    checkpoint = (
        ModelCheckpoint(save_path, monitor="val_accuracy", mode="max")
        if save_path
        else None
    )

    try:
        for epoch in range(1, max_epochs + 1):
            train_metrics = distill_epoch(
                student=student,
                teacher=teacher,
                data_loader=train_loader,
                optimizer=optimizer,
                hard_loss_fn=hard_loss_fn,
                device=device,
                temperature=temperature,
                alpha=alpha,
            )
            history["train_loss"].append(float(train_metrics["loss"]))
            history["train_acc"].append(float(train_metrics["accuracy"]))

            if epoch % val_interval == 0:
                val_metrics = validate_epoch(
                    model=student,
                    data_loader=val_loader,
                    loss_fn=hard_loss_fn,
                    device=device,
                )
                history["val_loss"].append(float(val_metrics["loss"]))
                history["val_acc"].append(float(val_metrics["accuracy"]))

                metrics = {
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                }
                if checkpoint is not None:
                    checkpoint.step(
                        model=student,
                        optimizer=optimizer,
                        epoch=epoch,
                        metrics=metrics,
                    )

                should_stop = early_stopping.step(float(val_metrics["accuracy"]))
                if should_stop:
                    break
    except Exception as exc:
        raise RuntimeError("蒸馏训练循环发生异常") from exc

    return history


def train(
    model: DeepLearningClassifier,
    train_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    val_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    optimizer: Optional[torch.optim.Optimizer] = None,
    loss_fn: Optional[nn.Module] = None,
    max_epochs: int = 100,
    patience: int = 10,
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None,
    val_interval: int = 1,
) -> Dict[str, List[float]]:
    """
    完整训练循环，支持早停与最佳模型保存。

    Args:
        model: 模型实例。
        train_loader: 训练数据加载器。
        val_loader: 验证数据加载器。
        optimizer: 优化器，默认Adam。
        loss_fn: 损失函数，默认CrossEntropyLoss。
        max_epochs: 最大训练轮数。
        patience: 早停耐心值。
        device: 设备，默认自动选择CPU或GPU。
        save_path: 最佳模型保存路径，为None时不保存。
        val_interval: 验证间隔（单位epoch）。

    Returns:
        训练历史字典，包含train_loss/val_loss/train_acc/val_acc。
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    early_stopping = EarlyStopping(patience=patience, mode="max")
    checkpoint = (
        ModelCheckpoint(save_path, monitor="val_accuracy", mode="max")
        if save_path
        else None
    )

    try:
        for epoch in range(1, max_epochs + 1):
            train_metrics = train_epoch(
                model=model,
                data_loader=train_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=device,
            )
            history["train_loss"].append(float(train_metrics["loss"]))
            history["train_acc"].append(float(train_metrics["accuracy"]))

            if epoch % val_interval == 0:
                val_metrics = validate_epoch(
                    model=model,
                    data_loader=val_loader,
                    loss_fn=loss_fn,
                    device=device,
                )
                history["val_loss"].append(float(val_metrics["loss"]))
                history["val_acc"].append(float(val_metrics["accuracy"]))

                metrics = {
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                }
                if checkpoint is not None:
                    checkpoint.step(
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        metrics=metrics,
                    )

                should_stop = early_stopping.step(float(val_metrics["accuracy"]))
                if should_stop:
                    break
    except Exception as exc:
        raise RuntimeError("训练循环发生异常") from exc

    return history


def train_loop(
    model: DeepLearningClassifier,
    train_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    val_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    max_epochs: int,
    patience: int,
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None,
    val_interval: int = 1,
) -> Dict[str, List[float]]:
    """
    兼容旧接口的训练循环包装。

    Args:
        model: 模型实例。
        train_loader: 训练数据加载器。
        val_loader: 验证数据加载器。
        optimizer: 优化器。
        loss_fn: 损失函数。
        max_epochs: 最大训练轮数。
        patience: 早停耐心值。
        device: 设备，默认自动选择CPU或GPU。
        save_path: 最佳模型保存路径，为None时不保存。
        val_interval: 验证间隔（单位epoch）。

    Returns:
        训练历史字典，包含train_loss/val_loss/train_acc/val_acc。
    """
    return train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        max_epochs=max_epochs,
        patience=patience,
        device=device,
        save_path=save_path,
        val_interval=val_interval,
    )
