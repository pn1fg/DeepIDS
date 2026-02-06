"""
训练回调模块。

提供EarlyStopping与ModelCheckpoint回调用于训练过程控制。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from models.deep_learning import DeepLearningClassifier, save_model


@dataclass
class EarlyStopping:
    """
    早停回调。

    Args:
        patience: 容忍无改进的轮数。
        mode: 指标方向，"max"表示越大越好，"min"表示越小越好。
        min_delta: 最小改进阈值。
    """

    patience: int
    mode: str = "max"
    min_delta: float = 0.0
    best_score: Optional[float] = None
    counter: int = 0

    def step(self, current: float) -> bool:
        """
        更新早停状态。

        Args:
            current: 当前指标值。

        Returns:
            是否触发早停。
        """
        if self.best_score is None:
            self.best_score = current
            self.counter = 0
            return False

        improved = False
        if self.mode == "max":
            improved = current > self.best_score + self.min_delta
        elif self.mode == "min":
            improved = current < self.best_score - self.min_delta
        else:
            raise ValueError("mode只能为'max'或'min'")

        if improved:
            self.best_score = current
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience


@dataclass
class ModelCheckpoint:
    """
    最佳模型保存回调。

    Args:
        save_path: 保存路径。
        monitor: 监控指标名称。
        mode: 指标方向，"max"表示越大越好，"min"表示越小越好。
    """

    save_path: str
    monitor: str = "val_accuracy"
    mode: str = "max"
    best_score: Optional[float] = None

    def step(
        self,
        model: DeepLearningClassifier,
        optimizer: Optional[torch.optim.Optimizer],
        epoch: int,
        metrics: Dict[str, Any],
    ) -> bool:
        """
        根据监控指标判断并保存模型。

        Args:
            model: 模型实例。
            optimizer: 优化器实例。
            epoch: 当前轮次。
            metrics: 指标字典。

        Returns:
            是否保存了新的最佳模型。
        """
        if self.monitor not in metrics:
            raise ValueError("监控指标不存在")

        current = float(metrics[self.monitor])
        if self.best_score is None:
            improved = True
        elif self.mode == "max":
            improved = current > self.best_score
        elif self.mode == "min":
            improved = current < self.best_score
        else:
            raise ValueError("mode只能为'max'或'min'")

        if improved:
            self.best_score = current
            try:
                save_model(
                    model=model,
                    save_path=self.save_path,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics={self.monitor: current},
                )
            except Exception as exc:
                raise IOError("保存模型检查点失败") from exc
            return True

        return False
