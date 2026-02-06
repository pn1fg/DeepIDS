from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from models.deep_learning import DeepLearningClassifier
from training.evaluate import evaluate_model, generate_evaluation_report
from utils.config import ATTACK_TYPES


def run_example() -> None:
    try:
        features = torch.randn(200, 41)
        labels = torch.randint(0, len(ATTACK_TYPES), (200,))
        loader = DataLoader(TensorDataset(features, labels), batch_size=32)

        model = DeepLearningClassifier(
            input_dim=41,
            hidden_dims=[256, 128, 64],
            output_dim=len(ATTACK_TYPES),
            dropout=0.3,
        )

        evaluation = evaluate_model(
            model=model, data_loader=loader, num_classes=len(ATTACK_TYPES)
        )
        report = generate_evaluation_report(
            evaluation, output_dir="outputs/results/eval_example"
        )

        _ = report
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError("评估示例执行失败") from exc


def main() -> None:
    try:
        run_example()
    except RuntimeError as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
