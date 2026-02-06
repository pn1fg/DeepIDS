"""
推理引擎模块。

提供Predictor类用于批量与单样本实时推理，支持置信度阈值与设备管理。

使用示例:
    import torch
    from models.deep_learning import DeepLearningClassifier
    from inference.predictor import Predictor

    model = DeepLearningClassifier(41, [256, 128, 64], 5, 0.3)
    predictor = Predictor(model=model, confidence_threshold=0.7)
    x = torch.randn(4, 41)
    preds = predictor.predict_batch(x)
    one_pred = predictor.predict_one(x[0])

错误处理说明:
    - 输入维度不匹配或设备错误时抛出ValueError或RuntimeError。
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple
import os
import time

import torch
from torch import nn

from models.deep_learning import load_model
from utils.config import get_feature_config, load_config


class Predictor:
    """
    推理引擎。

    Args:
        model: 训练好的模型实例。
        device: 推理设备，默认CPU。
        confidence_threshold: 置信度阈值，范围0.0-1.0。

    使用示例:
        predictor = Predictor(model, torch.device("cpu"), 0.7)
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        model_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        confidence_threshold: float = 0.7,
        normalize: bool = True,
        cache_size: int = 0,
        enable_hot_reload: bool = False,
        config_path: str = "config.yaml",
    ) -> None:
        if confidence_threshold < 0.0 or confidence_threshold > 1.0:
            raise ValueError("confidence_threshold应在[0,1]范围内")
        if model is None and model_path is None:
            raise ValueError("model与model_path不能同时为空")
        if model is not None and model_path is not None:
            raise ValueError("model与model_path不能同时提供")
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model_path = model_path
        self.model = (
            load_model(model_path, device=str(self.device))
            if model_path is not None
            else model
        )
        if self.model is None:
            raise ValueError("模型初始化失败")
        self.model = self.model.eval()
        self.model.to(self.device)
        self.confidence_threshold = confidence_threshold
        self.normalize = normalize
        self._bounds = None
        if normalize:
            schema, bounds = get_feature_config(load_config(config_path))
            if not bounds or len(schema) == 0:
                raise ValueError("特征归一化范围未配置")
            self._bounds = [bounds[name] for name in schema]
        self._cache_size = max(int(cache_size), 0)
        self._cache: OrderedDict[Tuple[float, ...], Dict[str, Any]] = OrderedDict()
        self._enable_hot_reload = enable_hot_reload
        self._model_mtime = self._get_model_mtime()
        self.total_samples = 0
        self.total_time = 0.0

    def _get_model_mtime(self) -> Optional[float]:
        if self.model_path is None:
            return None
        try:
            return os.path.getmtime(self.model_path)
        except OSError:
            return None

    def _maybe_reload(self) -> None:
        if not self._enable_hot_reload or self.model_path is None:
            return
        current_mtime = self._get_model_mtime()
        if current_mtime is None or self._model_mtime == current_mtime:
            return
        try:
            self.model = load_model(self.model_path, device=str(self.device))
            self.model = self.model.eval()
            self.model.to(self.device)
            self._model_mtime = current_mtime
            self._cache.clear()
        except Exception as exc:
            raise RuntimeError("模型热加载失败") from exc

    def _cache_key(self, feature: torch.Tensor) -> Tuple[float, ...]:
        return tuple(round(float(x), 6) for x in feature.tolist())

    def _get_cached(self, key: Tuple[float, ...]) -> Optional[Dict[str, Any]]:
        if self._cache_size <= 0:
            return None
        value = self._cache.get(key)
        if value is not None:
            self._cache.move_to_end(key)
        return value

    def _set_cache(self, key: Tuple[float, ...], value: Dict[str, Any]) -> None:
        if self._cache_size <= 0:
            return
        self._cache[key] = value
        self._cache.move_to_end(key)
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def _normalize(self, features: torch.Tensor) -> torch.Tensor:
        if not self.normalize or self._bounds is None:
            return features
        if features.shape[-1] != len(self._bounds):
            raise ValueError("特征维度与配置不一致")
        mins = torch.tensor([b[0] for b in self._bounds], device=features.device)
        maxs = torch.tensor([b[1] for b in self._bounds], device=features.device)
        denom = torch.where(maxs > mins, maxs - mins, torch.ones_like(maxs))
        normalized = (features - mins) / denom
        return torch.clamp(normalized, 0.0, 1.0)

    def _prepare_features(self, features: Any) -> torch.Tensor:
        tensor = (
            features
            if isinstance(features, torch.Tensor)
            else torch.as_tensor(features)
        )
        return tensor.to(torch.float32)

    def _postprocess(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        probs = torch.softmax(logits, dim=1)
        conf, idx = torch.max(probs, dim=1)
        return idx, conf

    def get_prediction_confidence(self, features: Any) -> torch.Tensor:
        self._maybe_reload()
        tensor = self._prepare_features(features)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2:
            raise ValueError("features应为二维(batch_size, input_dim)")
        try:
            tensor = tensor.to(self.device)
            tensor = self._normalize(tensor)
            with torch.no_grad():
                logits = self.model(tensor)
                return torch.softmax(logits, dim=1)
        except Exception as exc:
            raise RuntimeError("置信度计算失败") from exc

    def predict(self, feature: Any) -> Dict[str, Any]:
        start = time.perf_counter()
        pred_idx, confidence = self.predict_one(feature)
        elapsed = time.perf_counter() - start
        self.total_samples += 1
        self.total_time += elapsed
        return {
            "pred": pred_idx,
            "confidence": confidence,
            "inference_time_ms": float(elapsed * 1000.0),
        }

    def predict_batch(self, features: torch.Tensor) -> Dict[str, Any]:
        """
        批量推理。

        Args:
            features: 形状为(batch_size, input_dim)的特征张量。

        Returns:
            字典，包含preds(类别索引)与confidences(最大概率)。
        """
        self._maybe_reload()
        features = self._prepare_features(features)
        if features.ndim != 2:
            raise ValueError("features应为二维(batch_size, input_dim)")
        try:
            start = time.perf_counter()
            features = features.to(self.device)
            features = self._normalize(features)

            cached_preds = []
            cached_conf = []
            cached_probs = []
            to_infer = []
            to_infer_idx = []

            for idx in range(features.size(0)):
                key = self._cache_key(features[idx].detach().cpu())
                cached = self._get_cached(key)
                if cached is None:
                    to_infer.append(features[idx])
                    to_infer_idx.append(idx)
                else:
                    cached_preds.append((idx, cached["pred"]))
                    cached_conf.append((idx, cached["confidence"]))
                    cached_probs.append((idx, cached["probs"]))

            output_dim = int(getattr(self.model, "output_dim", 0))
            if output_dim <= 0:
                output_dim = int(self.model(features[:1]).shape[1])
            preds = torch.zeros(features.size(0), dtype=torch.long)
            conf = torch.zeros(features.size(0), dtype=torch.float32)
            probs = torch.zeros((features.size(0), output_dim), dtype=torch.float32)

            if to_infer:
                batch = torch.stack(to_infer, dim=0)
                with torch.no_grad():
                    logits = self.model(batch)
                    batch_probs = torch.softmax(logits, dim=1)
                    batch_conf, batch_preds = torch.max(batch_probs, dim=1)
                for local_idx, global_idx in enumerate(to_infer_idx):
                    pred_val = int(batch_preds[local_idx].item())
                    conf_val = float(batch_conf[local_idx].item())
                    probs_val = batch_probs[local_idx].cpu()
                    preds[global_idx] = pred_val
                    conf[global_idx] = conf_val
                    probs[global_idx] = probs_val
                    key = self._cache_key(features[global_idx].detach().cpu())
                    self._set_cache(
                        key,
                        {
                            "pred": pred_val,
                            "confidence": conf_val,
                            "probs": probs_val,
                        },
                    )

            for idx, pred_val in cached_preds:
                preds[idx] = pred_val
            for idx, conf_val in cached_conf:
                conf[idx] = conf_val
            for idx, probs_val in cached_probs:
                probs[idx] = probs_val

            mask = conf >= self.confidence_threshold
            elapsed = time.perf_counter() - start
            self.total_samples += int(features.size(0))
            self.total_time += elapsed
            throughput = (
                float(features.size(0)) / elapsed if elapsed > 0 else float("inf")
            )
            return {
                "preds": preds,
                "confidences": conf,
                "probs": probs,
                "mask": mask,
                "inference_time_ms": float(elapsed * 1000.0),
                "throughput": throughput,
            }
        except Exception as exc:
            raise RuntimeError("批量推理失败") from exc

    def predict_one(self, feature: Any) -> Tuple[int, float]:
        """
        单样本推理。

        Args:
            feature: 形状为(input_dim,)的特征张量。

        Returns:
            (pred_idx, confidence)。
        """
        self._maybe_reload()
        feature_t = self._prepare_features(feature)
        if feature_t.ndim != 1:
            raise ValueError("feature应为一维(input_dim,)")
        try:
            feature_b = feature_t.unsqueeze(0).to(self.device)
            feature_b = self._normalize(feature_b)
            if self._cache_size > 0:
                key = self._cache_key(feature_b[0].detach().cpu())
                cached = self._get_cached(key)
                if cached is not None:
                    return int(cached["pred"]), float(cached["confidence"])
            with torch.no_grad():
                logits = self.model(feature_b)
                preds, conf = self._postprocess(logits)
                pred_val = int(preds.item())
                conf_val = float(conf.item())
                if self._cache_size > 0:
                    probs = torch.softmax(logits, dim=1)[0].detach().cpu()
                    self._set_cache(
                        self._cache_key(feature_b[0].detach().cpu()),
                        {"pred": pred_val, "confidence": conf_val, "probs": probs},
                    )
                return pred_val, conf_val
        except Exception as exc:
            raise RuntimeError("单样本推理失败") from exc

    def benchmark_inference(
        self,
        batch_size: int = 32,
        num_runs: int = 50,
        warmup: int = 5,
    ) -> Dict[str, float]:
        if batch_size <= 0 or num_runs <= 0:
            raise ValueError("batch_size与num_runs必须大于0")
        device = self.device
        input_dim = int(self.model.input_dim)
        times = []
        for _ in range(warmup):
            sample = torch.randn(batch_size, input_dim, device=device)
            sample = self._normalize(sample)
            with torch.no_grad():
                _ = self.model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize()

        for _ in range(num_runs):
            sample = torch.randn(batch_size, input_dim, device=device)
            sample = self._normalize(sample)
            start = time.perf_counter()
            with torch.no_grad():
                _ = self.model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        avg_latency = float(sum(times) / len(times))
        throughput = (
            float(batch_size / avg_latency) if avg_latency > 0 else float("inf")
        )
        p95_index = max(int(0.95 * len(times)) - 1, 0)
        return {
            "avg_latency_ms": avg_latency * 1000.0,
            "throughput": throughput,
            "p95_latency_ms": float(sorted(times)[p95_index] * 1000.0),
        }


class EnsemblePredictor:
    def __init__(
        self,
        predictors: Optional[Sequence[Predictor]] = None,
        models: Optional[Sequence[nn.Module]] = None,
        device: Optional[torch.device] = None,
        confidence_threshold: float = 0.7,
        method: str = "soft",
        weights: Optional[Sequence[float]] = None,
        meta_model: Optional[nn.Module] = None,
        normalize: bool = True,
        config_path: str = "config.yaml",
    ) -> None:
        if confidence_threshold < 0.0 or confidence_threshold > 1.0:
            raise ValueError("confidence_threshold应在[0,1]范围内")
        if predictors is None:
            if models is None or len(models) == 0:
                raise ValueError("predictors或models必须提供")
            self.predictors = [
                Predictor(
                    model=model,
                    device=device,
                    confidence_threshold=confidence_threshold,
                    normalize=normalize,
                    config_path=config_path,
                )
                for model in models
            ]
        else:
            if len(predictors) == 0:
                raise ValueError("predictors不能为空")
            self.predictors = list(predictors)
        self.confidence_threshold = confidence_threshold
        method = method.lower()
        if method == "boosting":
            method = "weighted"
        if method not in {"soft", "vote", "weighted", "stacking"}:
            raise ValueError("method不支持")
        if method == "stacking" and meta_model is None:
            raise ValueError("stacking需要提供meta_model")
        self.method = method
        self.meta_model = meta_model
        if weights is not None:
            if len(weights) != len(self.predictors):
                raise ValueError("weights长度与predictors不一致")
            weights_arr = torch.tensor(weights, dtype=torch.float32)
            if float(weights_arr.sum().item()) <= 0.0:
                raise ValueError("weights权重和必须大于0")
            self.weights = (weights_arr / weights_arr.sum()).tolist()
        else:
            self.weights = None

    def _aggregate_probs(self, probs_list: List[torch.Tensor]) -> torch.Tensor:
        if self.method == "soft":
            return torch.stack(probs_list, dim=0).mean(dim=0)
        if self.method == "weighted":
            weights = self.weights or [1.0 / len(probs_list)] * len(probs_list)
            weighted = [p * float(w) for p, w in zip(probs_list, weights)]
            return torch.stack(weighted, dim=0).sum(dim=0)
        if self.method == "vote":
            preds = [torch.argmax(p, dim=1) for p in probs_list]
            num_classes = int(probs_list[0].shape[1])
            votes = torch.zeros(
                preds[0].shape[0], num_classes, dtype=torch.float32
            )
            for pred in preds:
                votes.scatter_add_(
                    1,
                    pred.unsqueeze(1),
                    torch.ones_like(pred).float().unsqueeze(1),
                )
            return votes / float(len(probs_list))
        if self.method == "stacking":
            if self.meta_model is None:
                raise ValueError("meta_model未设置")
            try:
                meta_device = next(self.meta_model.parameters()).device
            except StopIteration:
                meta_device = torch.device("cpu")
            stacked = torch.cat([p.to(meta_device) for p in probs_list], dim=1)
            with torch.no_grad():
                logits = self.meta_model(stacked)
                return torch.softmax(logits, dim=1).cpu()
        raise ValueError("method不支持")

    def predict_batch(self, features: Any) -> Dict[str, Any]:
        start = time.perf_counter()
        outputs = [pred.predict_batch(features) for pred in self.predictors]
        probs_list = [out["probs"] for out in outputs]
        probs = self._aggregate_probs(probs_list)
        conf, preds = torch.max(probs, dim=1)
        mask = conf >= self.confidence_threshold
        elapsed = time.perf_counter() - start
        throughput = float(probs.shape[0]) / elapsed if elapsed > 0 else float("inf")
        return {
            "preds": preds,
            "confidences": conf,
            "probs": probs,
            "mask": mask,
            "inference_time_ms": float(elapsed * 1000.0),
            "throughput": throughput,
        }

    def predict_one(self, feature: Any) -> Tuple[int, float]:
        result = self.predict_batch(torch.as_tensor(feature).unsqueeze(0))
        return int(result["preds"][0].item()), float(result["confidences"][0].item())
