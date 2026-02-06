from __future__ import annotations

import torch

from inference.predictor import Predictor
from models.deep_learning import DeepLearningClassifier, save_model
from utils.config import ATTACK_TYPES, get_config
from utils.logger import get_logger


def run_example() -> None:
    try:
        config = get_config("config.yaml")
        logger = get_logger("ids.inference_example", config=config.as_dict())
        logger.info("开始推理示例")
        model = DeepLearningClassifier(
            input_dim=41,
            hidden_dims=[256, 128, 64],
            output_dim=len(ATTACK_TYPES),
            dropout=0.3,
        )
        save_path = "outputs/models/inference_example.pth"
        save_model(model, save_path)

        predictor = Predictor(
            model_path=save_path,
            confidence_threshold=0.7,
        )
        feature = torch.randn(41)
        pred_idx, confidence = predictor.predict_one(feature)

        batch = torch.randn(8, 41)
        batch_result = predictor.predict_batch(batch)
        probs = predictor.get_prediction_confidence(batch)

        _ = (pred_idx, confidence, batch_result, probs)
        logger.info("推理示例完成")
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError("推理示例执行失败") from exc


def main() -> None:
    try:
        run_example()
    except RuntimeError as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
