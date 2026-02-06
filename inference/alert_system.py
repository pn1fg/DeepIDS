from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Deque, Dict, Iterable, Optional, Tuple
import json
import logging
import os
import time

from data.datasets import ATTACK_TYPES

logger = logging.getLogger("ids.alert_system")


@dataclass
class Alert:
    timestamp: float
    src_ip: str
    dst_ip: str
    attack_type: str
    confidence: float
    level: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": datetime.utcfromtimestamp(self.timestamp).isoformat() + "Z",
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "attack_type": self.attack_type,
            "confidence": self.confidence,
            "level": self.level,
        }


class AlertSystem:
    def __init__(
        self,
        log_dir: str = "outputs/logs/alerts",
        dedup_window_seconds: float = 60.0,
        max_queue: int = 1000,
        notifiers: Optional[Dict[str, Callable[[Alert], None]]] = None,
    ) -> None:
        self.log_dir = log_dir
        self.dedup_window_seconds = max(float(dedup_window_seconds), 0.0)
        self._queue: Deque[Alert] = deque(maxlen=max(int(max_queue), 1))
        self._dedup_cache: Dict[Tuple[str, str], float] = {}
        self._attack_counter: Counter[str] = Counter()
        self._source_counter: Counter[str] = Counter()
        self._time_series: Counter[str] = Counter()
        self._total_alerts = 0
        self._dedup_skipped = 0
        self._notifiers = notifiers or {}

    def classify_alert_level(self, confidence: float) -> str:
        if confidence > 0.9:
            return "严重"
        if confidence > 0.7:
            return "高"
        if confidence > 0.5:
            return "中"
        return "低"

    def generate_alert(self, detection: Dict[str, Any]) -> Optional[Alert]:
        src_ip = str(detection.get("src_ip", ""))
        dst_ip = str(detection.get("dst_ip", ""))
        confidence = float(detection.get("confidence", 0.0))
        attack_type = detection.get("attack_type")
        if attack_type is None:
            pred = detection.get("pred")
            if pred is None:
                raise ValueError("缺少attack_type或pred")
            attack_type = ATTACK_TYPES.get(int(pred), str(pred))
        attack_type = str(attack_type)
        if not src_ip or not dst_ip:
            raise ValueError("src_ip与dst_ip不能为空")
        level = self.classify_alert_level(confidence)
        alert = Alert(
            timestamp=time.time(),
            src_ip=src_ip,
            dst_ip=dst_ip,
            attack_type=attack_type,
            confidence=confidence,
            level=level,
        )
        if self.deduplicate_alerts(alert):
            self._dedup_skipped += 1
            return None
        self._queue.append(alert)
        return alert

    def log_alert(self, alert: Alert) -> str:
        os.makedirs(self.log_dir, exist_ok=True)
        date_str = datetime.utcfromtimestamp(alert.timestamp).strftime("%Y%m%d")
        log_path = os.path.join(self.log_dir, f"alerts_{date_str}.jsonl")
        payload = json.dumps(alert.to_dict(), ensure_ascii=False)
        try:
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(payload + "\n")
        except OSError as exc:
            logger.error("写入告警日志失败: %s", exc)
            raise RuntimeError("写入告警日志失败") from exc
        self._update_statistics(alert)
        return log_path

    def deduplicate_alerts(self, alert: Alert) -> bool:
        key = (alert.src_ip, alert.attack_type)
        last_ts = self._dedup_cache.get(key)
        now = alert.timestamp
        if last_ts is not None and now - last_ts <= self.dedup_window_seconds:
            return True
        self._dedup_cache[key] = now
        return False

    def get_alert_statistics(self, top_n: int = 5) -> Dict[str, Any]:
        total = self._total_alerts
        top_sources = self._source_counter.most_common(max(int(top_n), 1))
        time_series = [
            {"time": key, "count": int(self._time_series[key])}
            for key in sorted(self._time_series.keys())
        ]
        return {
            "total_alerts": total,
            "dedup_skipped": self._dedup_skipped,
            "attack_distribution": dict(self._attack_counter),
            "top_sources": top_sources,
            "time_series": time_series,
        }

    def send_alert(
        self, alert: Alert, channels: Optional[Iterable[str]] = None
    ) -> None:
        if channels is None:
            channels = self._notifiers.keys()
        for name in channels:
            notifier = self._notifiers.get(name)
            if notifier is None:
                continue
            try:
                notifier(alert)
            except Exception as exc:
                logger.warning("告警发送失败: %s", exc)

    def _update_statistics(self, alert: Alert) -> None:
        self._total_alerts += 1
        self._attack_counter[alert.attack_type] += 1
        self._source_counter[alert.src_ip] += 1
        minute_key = datetime.utcfromtimestamp(alert.timestamp).strftime(
            "%Y-%m-%d %H:%M"
        )
        self._time_series[minute_key] += 1
