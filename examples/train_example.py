"""
训练模块使用示例。

展示数据加载、模型训练与结果保存的完整流程。
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from models.deep_learning import DeepLearningClassifier
from training.train import train
from utils.config import ATTACK_TYPES


def run_example() -> None:
    """
    运行训练示例。

    Raises:
        RuntimeError: 示例运行失败时抛出。
    """
    try:
        features = torch.randn(256, 41)
        labels = torch.randint(0, len(ATTACK_TYPES), (256,))
        dataset = TensorDataset(features, labels)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=32)

        model = DeepLearningClassifier(
            input_dim=41,
            hidden_dims=[256, 128, 64],
            output_dim=len(ATTACK_TYPES),
            dropout=0.3,
        )

        history = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=3,
            patience=2,
            save_path="outputs/models/best_model.pth",
            val_interval=1,
        )

        _ = history
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError("训练示例执行失败") from exc


def main() -> None:
    """
    主入口。

    Raises:
        SystemExit: 示例执行失败时退出程序。
    """
    try:
        run_example()
    except RuntimeError as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
