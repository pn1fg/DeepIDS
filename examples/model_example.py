"""
模型使用示例。

该示例演示模型创建、前向推理、保存与加载的完整流程。
"""

from __future__ import annotations

import torch

from models.deep_learning import DeepLearningClassifier, load_model, save_model
from utils.config import ATTACK_TYPES


def run_example() -> None:
    """
    运行模型使用示例。

    Raises:
        RuntimeError: 示例运行失败时抛出。
    """
    try:
        # 1. 创建模型
        model = DeepLearningClassifier(
            input_dim=41,
            hidden_dims=[256, 128, 64],
            output_dim=len(ATTACK_TYPES),
            dropout=0.3,
        )

        # 2. 创建随机输入
        x = torch.randn(32, 41)

        # 3. 前向传播
        output = model(x)
        print(f"Output shape: {output.shape}")

        # 4. 保存模型
        save_model(model, "model.pth")

        # 5. 加载模型
        loaded_model = load_model("model.pth")
        _ = loaded_model(x)
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError("模型示例执行失败") from exc


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
