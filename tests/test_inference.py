import os
import time

import torch

from models.deep_learning import DeepLearningClassifier, save_model
from inference.predictor import Predictor


def test_predict_single_and_batch(tmp_path):
    model = DeepLearningClassifier()
    predictor = Predictor(model=model, device=torch.device("cpu"), normalize=False)
    feature = torch.randn(41)
    single = predictor.predict(feature)
    assert 0 <= single["pred"] <= model.output_dim - 1
    assert 0.0 <= single["confidence"] <= 1.0
    assert single["inference_time_ms"] >= 0.0

    batch = torch.randn(6, 41)
    result = predictor.predict_batch(batch)
    assert result["preds"].shape[0] == 6
    assert result["confidences"].shape[0] == 6
    assert "throughput" in result


def test_confidence_and_cache():
    model = DeepLearningClassifier()
    predictor = Predictor(
        model=model, device=torch.device("cpu"), normalize=False, cache_size=2
    )
    feature = torch.randn(41)
    probs = predictor.get_prediction_confidence(feature)
    assert probs.shape[-1] == model.output_dim
    _ = predictor.predict_one(feature)
    assert len(predictor._cache) == 1


def test_hot_reload(tmp_path):
    model = DeepLearningClassifier()
    checkpoint = tmp_path / "model.pth"
    save_model(model, str(checkpoint))

    predictor = Predictor(
        model_path=str(checkpoint),
        device=torch.device("cpu"),
        normalize=False,
        enable_hot_reload=True,
    )
    feature = torch.randn(41)
    _ = predictor.predict_one(feature)
    initial_mtime = predictor._model_mtime

    new_model = DeepLearningClassifier()
    for param in new_model.parameters():
        torch.nn.init.constant_(param, 0.0)
    save_model(new_model, str(checkpoint))
    future_time = time.time() + 5
    os.utime(str(checkpoint), (future_time, future_time))
    expected_mtime = os.path.getmtime(str(checkpoint))
    _ = predictor.predict_one(feature)
    assert predictor._model_mtime == expected_mtime
    assert predictor._model_mtime != initial_mtime


def test_benchmark():
    model = DeepLearningClassifier()
    predictor = Predictor(model=model, device=torch.device("cpu"), normalize=False)
    metrics = predictor.benchmark_inference(batch_size=4, num_runs=3, warmup=1)
    assert metrics["avg_latency_ms"] >= 0.0
    assert metrics["throughput"] > 0.0
