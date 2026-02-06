from __future__ import annotations

from typing import Any, Dict, Iterable, Optional
import logging
import threading
import time
import tracemalloc

from scapy.all import sniff

from data.data_loader import DataLoader, PacketRecord
from data.datasets import ATTACK_TYPES
from inference.alert_system import AlertSystem
from inference.predictor import Predictor
from models.feature_extractor import FeatureExtractor
from utils.config import load_config

logger = logging.getLogger("ids.real_time_monitor")


class RealTimeMonitor:
    def __init__(
        self,
        predictor: Optional[Predictor] = None,
        feature_extractor: Optional[FeatureExtractor] = None,
        alert_system: Optional[AlertSystem] = None,
        model_path: Optional[str] = None,
        interface: Optional[str] = None,
        bpf_filter: Optional[str] = None,
        confidence_threshold: float = 0.7,
        normalize: bool = True,
        cache_size: int = 0,
        enable_hot_reload: bool = False,
        config_path: str = "config.yaml",
        monitor_interval_seconds: Optional[float] = None,
    ) -> None:
        if predictor is None:
            predictor = Predictor(
                model_path=model_path,
                confidence_threshold=confidence_threshold,
                normalize=normalize,
                cache_size=cache_size,
                enable_hot_reload=enable_hot_reload,
                config_path=config_path,
            )
        self.predictor = predictor
        self.feature_extractor = feature_extractor or FeatureExtractor(
            config_path=config_path
        )
        self.alert_system = alert_system
        self.interface = interface
        self.bpf_filter = bpf_filter
        self._data_loader = DataLoader()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None
        self._last_perf_ts = time.perf_counter()
        self._last_cpu_time = time.process_time()
        self._last_performance: Dict[str, float] = {
            "cpu_percent": 0.0,
            "memory_bytes": 0.0,
            "avg_latency_ms": 0.0,
            "throughput": 0.0,
        }

        cfg = load_config(config_path)
        inference_cfg = cfg.get("inference") or {}
        if monitor_interval_seconds is None:
            monitor_interval_seconds = float(
                inference_cfg.get("monitor_interval_seconds", 1)
            )
        self.monitor_interval_seconds = max(float(monitor_interval_seconds), 0.1)

        self._total_packets = 0
        self._total_bytes = 0
        self._normal_count = 0
        self._threat_count = 0
        self._attack_counts = {name: 0 for name in ATTACK_TYPES.values()}
        self._total_inference_time = 0.0
        self._total_inference_samples = 0
        self._labeled_total = 0
        self._labeled_correct = 0
        self._errors = 0
        if not tracemalloc.is_tracing():
            tracemalloc.start()

    def start_monitoring(
        self,
        interface: Optional[str] = None,
        bpf_filter: Optional[str] = None,
        packet_iter: Optional[Iterable[Any]] = None,
        timeout: Optional[int] = None,
        count: Optional[int] = None,
        background: bool = False,
    ) -> Optional[threading.Thread]:
        if self._start_time is None:
            self._start_time = time.perf_counter()
        self._stop_event.clear()
        if interface is not None:
            self.interface = interface
        if bpf_filter is not None:
            self.bpf_filter = bpf_filter

        def runner() -> None:
            try:
                if packet_iter is not None:
                    for packet in packet_iter:
                        if self._stop_event.is_set():
                            break
                        try:
                            _ = self.process_packet(packet)
                        except RuntimeError:
                            continue
                    return
                if count is not None:
                    sniff(
                        iface=self.interface,
                        filter=self.bpf_filter,
                        prn=self.process_packet,
                        store=False,
                        count=count,
                    )
                    return
                while not self._stop_event.is_set():
                    sniff(
                        iface=self.interface,
                        filter=self.bpf_filter,
                        prn=self.process_packet,
                        store=False,
                        timeout=timeout or int(self.monitor_interval_seconds),
                        stop_filter=lambda _: self._stop_event.is_set(),
                    )
                    _ = self.monitor_performance()
            except Exception as exc:
                logger.error("监控循环异常: %s", exc)
                raise RuntimeError("实时监控启动失败") from exc

        if background:
            self._thread = threading.Thread(target=runner, daemon=True)
            self._thread.start()
            return self._thread

        runner()
        return None

    def process_packet(self, packet: Any) -> Dict[str, Any]:
        try:
            record, label = self._to_record(packet)
            features = self.feature_extractor.extract(record)
            start = time.perf_counter()
            pred, confidence = self.predictor.predict_one(features)
            elapsed = time.perf_counter() - start
            self.update_statistics(record, pred, confidence, elapsed, label)
            if self.alert_system is not None and int(pred) != 0:
                detection = {
                    "src_ip": record.src_ip,
                    "dst_ip": record.dst_ip,
                    "pred": int(pred),
                    "confidence": float(confidence),
                }
                try:
                    alert = self.alert_system.generate_alert(detection)
                    if alert is not None:
                        _ = self.alert_system.log_alert(alert)
                        self.alert_system.send_alert(alert)
                except Exception as exc:
                    logger.warning("告警处理失败: %s", exc)
            self._maybe_update_performance()
            return {
                "pred": pred,
                "confidence": confidence,
                "inference_time_ms": float(elapsed * 1000.0),
            }
        except Exception as exc:
            with self._lock:
                self._errors += 1
            raise RuntimeError("处理数据包失败") from exc

    def update_statistics(
        self,
        record: PacketRecord,
        pred: int,
        confidence: float,
        inference_time: float,
        label: Optional[int] = None,
    ) -> None:
        _ = confidence
        with self._lock:
            self._total_packets += 1
            self._total_bytes += int(record.length)
            if pred == 0:
                self._normal_count += 1
            else:
                self._threat_count += 1
            attack_name = ATTACK_TYPES.get(int(pred), str(pred))
            if attack_name not in self._attack_counts:
                self._attack_counts[attack_name] = 0
            self._attack_counts[attack_name] += 1
            self._total_inference_time += float(inference_time)
            self._total_inference_samples += 1
            if label is not None:
                self._labeled_total += 1
                if int(label) == int(pred):
                    self._labeled_correct += 1

    def get_statistics(self) -> Dict[str, Any]:
        with self._lock:
            elapsed = (
                time.perf_counter() - self._start_time
                if self._start_time is not None
                else 0.0
            )
            avg_latency_ms = (
                self._total_inference_time / self._total_inference_samples * 1000.0
                if self._total_inference_samples > 0
                else 0.0
            )
            throughput = self._total_inference_samples / elapsed if elapsed > 0 else 0.0
            accuracy = (
                self._labeled_correct / self._labeled_total
                if self._labeled_total > 0
                else None
            )
            avg_packet_size = (
                self._total_bytes / self._total_packets
                if self._total_packets > 0
                else 0.0
            )
            alert_stats = (
                self.alert_system.get_alert_statistics()
                if self.alert_system is not None
                else None
            )
            return {
                "traffic": {
                    "total_packets": self._total_packets,
                    "total_bytes": self._total_bytes,
                    "avg_packet_size": avg_packet_size,
                },
                "detection": {
                    "normal": self._normal_count,
                    "threat": self._threat_count,
                    "attack_counts": dict(self._attack_counts),
                    "accuracy": accuracy,
                },
                "performance": {
                    "avg_latency_ms": avg_latency_ms,
                    "throughput": throughput,
                    "cpu_percent": self._last_performance["cpu_percent"],
                    "memory_bytes": self._last_performance["memory_bytes"],
                },
                "alerts": alert_stats,
                "errors": self._errors,
            }

    def monitor_performance(self) -> Dict[str, float]:
        now = time.perf_counter()
        cpu_now = time.process_time()
        wall_delta = now - self._last_perf_ts
        cpu_delta = cpu_now - self._last_cpu_time
        cpu_percent = float(cpu_delta / wall_delta * 100.0) if wall_delta > 0 else 0.0
        memory_bytes = float(tracemalloc.get_traced_memory()[0])
        with self._lock:
            avg_latency_ms = (
                self._total_inference_time / self._total_inference_samples * 1000.0
                if self._total_inference_samples > 0
                else 0.0
            )
            elapsed = now - self._start_time if self._start_time is not None else 0.0
            throughput = self._total_inference_samples / elapsed if elapsed > 0 else 0.0
            self._last_performance = {
                "cpu_percent": cpu_percent,
                "memory_bytes": memory_bytes,
                "avg_latency_ms": avg_latency_ms,
                "throughput": throughput,
            }
        self._last_perf_ts = now
        self._last_cpu_time = cpu_now
        return dict(self._last_performance)

    def stop_monitoring(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _to_record(self, packet: Any) -> tuple[PacketRecord, Optional[int]]:
        if isinstance(packet, PacketRecord):
            label = getattr(packet, "label", None)
            return packet, label
        if isinstance(packet, dict):
            label = packet.get("label")
            record = self.feature_extractor._to_record(packet)
            return record, label
        record = self._data_loader._parse_packet(packet)
        return record, None

    def _maybe_update_performance(self) -> None:
        now = time.perf_counter()
        if now - self._last_perf_ts >= self.monitor_interval_seconds:
            _ = self.monitor_performance()
