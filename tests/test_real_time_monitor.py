import time

import torch
from scapy.all import IP, TCP

from data.data_loader import PacketRecord
from inference.alert_system import AlertSystem
from inference.predictor import Predictor
from inference.real_time_monitor import RealTimeMonitor
from models.deep_learning import DeepLearningClassifier
from models.feature_extractor import FeatureExtractor


class DummyPredictor:
    def __init__(self, preds):
        self._preds = list(preds)
        self._idx = 0

    def predict_one(self, features):
        pred = self._preds[self._idx % len(self._preds)]
        self._idx += 1
        return pred, 0.9


class DummyAlertSystem:
    def __init__(self):
        self.generated = []
        self.logged = []
        self.sent = []

    def generate_alert(self, detection):
        self.generated.append(detection)
        return detection

    def log_alert(self, alert):
        self.logged.append(alert)
        return "log"

    def send_alert(self, alert):
        self.sent.append(alert)

    def get_alert_statistics(self):
        return {"total_alerts": len(self.logged)}


def _make_record(ts: float) -> PacketRecord:
    return PacketRecord(
        timestamp=ts,
        src_ip="10.0.0.1",
        dst_ip="10.0.0.2",
        protocol="TCP",
        src_port=1234,
        dst_port=80,
        length=128,
        tcp_flags=2,
        tcp_window=1024,
        ttl=64,
        tos=0,
        ip_length=128,
        icmp_type=None,
        icmp_code=None,
    )


def _build_deterministic_predictor() -> Predictor:
    model = DeepLearningClassifier()
    for param in model.parameters():
        torch.nn.init.constant_(param, 0.0)
    last_layer = model.network[-1]
    if hasattr(last_layer, "bias") and last_layer.bias is not None:
        with torch.no_grad():
            last_layer.bias[1] = 5.0
    return Predictor(model=model, device=torch.device("cpu"), normalize=False)


def test_process_packet_updates_stats():
    extractor = FeatureExtractor(config_path="config.yaml")
    monitor = RealTimeMonitor(
        predictor=DummyPredictor([0]),
        feature_extractor=extractor,
        monitor_interval_seconds=0.1,
    )
    record = _make_record(1.0)
    result = monitor.process_packet(record)

    stats = monitor.get_statistics()
    assert result["pred"] == 0
    assert stats["traffic"]["total_packets"] == 1
    assert stats["detection"]["normal"] == 1


def test_start_monitoring_iterable():
    extractor = FeatureExtractor(config_path="config.yaml")
    alert_system = DummyAlertSystem()
    monitor = RealTimeMonitor(
        predictor=DummyPredictor([1, 2]),
        feature_extractor=extractor,
        alert_system=alert_system,
        monitor_interval_seconds=0.1,
    )
    records = [_make_record(1.0), _make_record(2.0)]
    monitor.start_monitoring(packet_iter=records)

    stats = monitor.get_statistics()
    assert stats["traffic"]["total_packets"] == 2
    assert stats["detection"]["threat"] == 2
    assert stats["alerts"]["total_alerts"] == 2
    assert len(alert_system.sent) == 2


def test_monitor_performance_keys():
    extractor = FeatureExtractor(config_path="config.yaml")
    monitor = RealTimeMonitor(
        predictor=DummyPredictor([0]),
        feature_extractor=extractor,
        monitor_interval_seconds=0.1,
    )
    perf = monitor.monitor_performance()
    assert "cpu_percent" in perf
    assert "memory_bytes" in perf
    assert "avg_latency_ms" in perf
    assert "throughput" in perf


def test_end_to_end_pipeline_alert(tmp_path):
    extractor = FeatureExtractor(config_path="config.yaml")
    alert_system = AlertSystem(log_dir=str(tmp_path))
    predictor = _build_deterministic_predictor()
    monitor = RealTimeMonitor(
        predictor=predictor,
        feature_extractor=extractor,
        alert_system=alert_system,
        monitor_interval_seconds=0.1,
    )
    packet = IP(src="10.0.0.1", dst="8.8.8.8") / TCP(sport=1234, dport=80, flags="S")
    packet = IP(bytes(packet))
    result = monitor.process_packet(packet)
    stats = monitor.get_statistics()

    assert result["pred"] == 1
    assert stats["traffic"]["total_packets"] == 1
    assert stats["detection"]["threat"] == 1
    assert stats["alerts"]["total_alerts"] == 1


def test_pressure_monitoring_stability():
    extractor = FeatureExtractor(config_path="config.yaml")
    monitor = RealTimeMonitor(
        predictor=DummyPredictor([0, 1, 2, 3, 4]),
        feature_extractor=extractor,
        monitor_interval_seconds=0.1,
    )
    records = [_make_record(float(idx)) for idx in range(500)]
    monitor.start_monitoring(packet_iter=records)
    stats = monitor.get_statistics()

    assert stats["traffic"]["total_packets"] == 500
    assert stats["errors"] == 0
    assert stats["performance"]["avg_latency_ms"] >= 0.0


def test_performance_metrics_after_processing():
    extractor = FeatureExtractor(config_path="config.yaml")
    monitor = RealTimeMonitor(
        predictor=DummyPredictor([0]),
        feature_extractor=extractor,
        monitor_interval_seconds=0.05,
    )
    records = [_make_record(float(idx)) for idx in range(50)]
    monitor.start_monitoring(packet_iter=records)
    time.sleep(0.01)
    perf = monitor.monitor_performance()

    assert perf["memory_bytes"] >= 0.0
    assert perf["avg_latency_ms"] >= 0.0
    assert perf["throughput"] >= 0.0
