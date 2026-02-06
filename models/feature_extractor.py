"""
特征提取器模块。

该模块提供从数据包记录中提取41维特征并归一化到[0, 1]的接口。

使用示例:
    import numpy as np
    from models.feature_extractor import FeatureExtractor
    from data.data_loader import PacketRecord

    extractor = FeatureExtractor(config_path="config.yaml")
    record = PacketRecord(
        timestamp=0.1,
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
    features = extractor.extract(record)
    assert features.shape == (41,)

错误处理说明:
    提取过程中如遇到数据异常，将抛出RuntimeError。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import ipaddress
import math
import logging

import numpy as np

from data.data_loader import PacketRecord
from utils.config import load_config, get_feature_config, get_feature_thresholds


logger = logging.getLogger("ids.feature_extractor")


class FeatureExtractor:
    """
    41维特征提取器。

    Args:
        feature_dim: 输出特征维度，默认41。

    使用示例:
        extractor = FeatureExtractor(41)
        features = extractor.extract(raw_sample)
    """

    def __init__(
        self,
        feature_dim: int = 41,
        config_path: str = "config.yaml",
        schema: Optional[List[str]] = None,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        thresholds: Optional[Dict[str, int]] = None,
    ) -> None:
        if feature_dim <= 0:
            raise ValueError("feature_dim必须为正整数")
        self.feature_dim = feature_dim
        self.config_path = config_path

        if schema is None or bounds is None or thresholds is None:
            config = load_config(config_path)
            schema_cfg, bounds_cfg = get_feature_config(config)
            thresholds_cfg = get_feature_thresholds(config)
            schema = schema or schema_cfg
            bounds = bounds or bounds_cfg
            thresholds = thresholds or thresholds_cfg
        if thresholds is None:
            thresholds = {}

        if len(schema) != feature_dim:
            raise ValueError("特征数量与feature_dim不一致")
        if not bounds:
            raise ValueError("特征归一化范围未配置")

        if not thresholds:
            raise ValueError("特征阈值未配置")

        self.schema = schema
        self.bounds = bounds
        self.thresholds = thresholds
        self._first_timestamp: Optional[float] = None
        self._last_timestamp: Optional[float] = None
        self._total_packets: int = 0
        self._total_bytes: int = 0
        self._max_packet_size: Optional[int] = None
        self._min_packet_size: Optional[int] = None
        self._mean_packet_size: float = 0.0
        self._m2_packet_size: float = 0.0
        self._ttl_sum: float = 0.0
        self._ttl_count: int = 0
        self._initial_window_size: Optional[int] = None
        self._tcp_window_sum: float = 0.0
        self._tcp_window_count: int = 0

    def extract(self, raw_sample: Any) -> np.ndarray:
        """
        从原始样本中提取41维特征。

        Args:
            raw_sample: 原始网络数据样本，可以是字典、对象或已解析结构。

        Returns:
            形状为(feature_dim,)的特征向量。
        """
        try:
            record = self._to_record(raw_sample)
            features = self._build_features(record)
            return self._normalize(features)
        except Exception as exc:
            logger.exception("特征提取失败")
            raise RuntimeError("特征提取过程发生异常") from exc

    def _to_record(self, raw_sample: Any) -> PacketRecord:
        if isinstance(raw_sample, PacketRecord):
            return raw_sample
        if isinstance(raw_sample, dict):
            return PacketRecord(
                timestamp=float(raw_sample.get("timestamp", 0.0)),
                src_ip=str(raw_sample.get("src_ip", "")),
                dst_ip=str(raw_sample.get("dst_ip", "")),
                protocol=str(raw_sample.get("protocol", "OTHER")),
                src_port=raw_sample.get("src_port"),
                dst_port=raw_sample.get("dst_port"),
                length=int(raw_sample.get("length", 0)),
                tcp_flags=raw_sample.get("tcp_flags"),
                tcp_window=raw_sample.get("tcp_window"),
                ttl=raw_sample.get("ttl"),
                tos=raw_sample.get("tos"),
                ip_length=raw_sample.get("ip_length"),
                icmp_type=raw_sample.get("icmp_type"),
                icmp_code=raw_sample.get("icmp_code"),
            )
        raise ValueError("raw_sample类型不受支持")

    def _safe_ip(self, value: str) -> Optional[ipaddress._BaseAddress]:
        try:
            return ipaddress.ip_address(value)
        except Exception:
            return None

    def _build_features(self, record: PacketRecord) -> Dict[str, float]:
        protocol = record.protocol.upper() if record.protocol else "OTHER"
        is_tcp = protocol == "TCP"
        is_udp = protocol == "UDP"
        is_icmp = protocol == "ICMP"

        protocol_type = 0 if is_tcp else 1 if is_udp else 2 if is_icmp else 2

        src_ip_obj = self._safe_ip(record.src_ip)
        dst_ip_obj = self._safe_ip(record.dst_ip)
        src_ip_numeric = int(src_ip_obj) if src_ip_obj is not None else 0
        dst_ip_numeric = int(dst_ip_obj) if dst_ip_obj is not None else 0

        now = float(record.timestamp)
        if self._first_timestamp is None:
            self._first_timestamp = now
        inter_packet_time = (
            0.0
            if self._last_timestamp is None
            else max(now - self._last_timestamp, 0.0)
        )
        connection_duration = max(now - self._first_timestamp, 0.0)
        self._last_timestamp = now

        self._total_packets += 1
        self._total_bytes += int(record.length)

        packet_length = int(record.length)
        if self._max_packet_size is None or packet_length > self._max_packet_size:
            self._max_packet_size = packet_length
        if self._min_packet_size is None or packet_length < self._min_packet_size:
            self._min_packet_size = packet_length

        delta = packet_length - self._mean_packet_size
        self._mean_packet_size += delta / self._total_packets
        delta2 = packet_length - self._mean_packet_size
        self._m2_packet_size += delta * delta2

        avg_packet_size = self._mean_packet_size
        std_packet_size = math.sqrt(
            self._m2_packet_size / max(self._total_packets - 1, 1)
        )

        src_port = int(record.src_port) if record.src_port is not None else 0
        dst_port = int(record.dst_port) if record.dst_port is not None else 0
        well_known_max = self.thresholds["well_known_max"]
        ephemeral_min = self.thresholds["ephemeral_min"]
        ephemeral_max = self.thresholds["ephemeral_max"]
        src_port_well_known = 1 if 0 <= src_port <= well_known_max else 0
        dst_port_well_known = 1 if 0 <= dst_port <= well_known_max else 0
        src_port_is_ephemeral = 1 if ephemeral_min <= src_port <= ephemeral_max else 0
        dst_port_is_ephemeral = 1 if ephemeral_min <= dst_port <= ephemeral_max else 0

        src_ip_private = 1 if src_ip_obj is not None and src_ip_obj.is_private else 0
        dst_ip_private = 1 if dst_ip_obj is not None and dst_ip_obj.is_private else 0
        if src_ip_private == 1 and dst_ip_private == 0:
            flow_direction = 1
        elif src_ip_private == 0 and dst_ip_private == 1:
            flow_direction = 0
        else:
            flow_direction = 0

        tcp_flags_raw = int(record.tcp_flags) if record.tcp_flags is not None else 0
        tcp_flags_syn = 1 if tcp_flags_raw & 0x02 else 0
        tcp_flags_ack = 1 if tcp_flags_raw & 0x10 else 0
        tcp_flags_fin = 1 if tcp_flags_raw & 0x01 else 0
        tcp_flags_rst = 1 if tcp_flags_raw & 0x04 else 0
        tcp_flags_psh = 1 if tcp_flags_raw & 0x08 else 0
        tcp_flags_urg = 1 if tcp_flags_raw & 0x20 else 0

        tcp_window = int(record.tcp_window) if record.tcp_window is not None else 0
        if is_tcp and self._initial_window_size is None:
            self._initial_window_size = tcp_window
        if is_tcp:
            self._tcp_window_sum += tcp_window
            self._tcp_window_count += 1
        tcp_window_avg = (
            self._tcp_window_sum / self._tcp_window_count
            if self._tcp_window_count > 0
            else 0.0
        )

        ttl = int(record.ttl) if record.ttl is not None else 0
        if ttl > 0:
            self._ttl_sum += ttl
            self._ttl_count += 1
        avg_ttl = self._ttl_sum / self._ttl_count if self._ttl_count > 0 else 0.0

        tos = int(record.tos) if record.tos is not None else 0
        ip_length = (
            int(record.ip_length) if record.ip_length is not None else packet_length
        )
        ip_version = int(src_ip_obj.version) if src_ip_obj is not None else 0

        packet_rate = (
            self._total_packets / connection_duration
            if connection_duration > 0.0
            else 0.0
        )
        byte_rate = (
            self._total_bytes / connection_duration
            if connection_duration > 0.0
            else 0.0
        )

        features = {
            "protocol_type": protocol_type,
            "src_ip_numeric": src_ip_numeric,
            "dst_ip_numeric": dst_ip_numeric,
            "src_port": src_port,
            "dst_port": dst_port,
            "packet_length": packet_length,
            "inter_packet_time": inter_packet_time,
            "tcp_flags_syn": tcp_flags_syn,
            "tcp_flags_ack": tcp_flags_ack,
            "tcp_flags_fin": tcp_flags_fin,
            "tcp_flags_rst": tcp_flags_rst,
            "tcp_flags_psh": tcp_flags_psh,
            "tcp_flags_urg": tcp_flags_urg,
            "flow_direction": flow_direction,
            "packet_count": self._total_packets,
            "total_bytes": self._total_bytes,
            "avg_packet_size": avg_packet_size,
            "max_packet_size": self._max_packet_size or 0,
            "min_packet_size": self._min_packet_size or 0,
            "std_packet_size": std_packet_size,
            "initial_window_size": self._initial_window_size or 0,
            "avg_ttl": avg_ttl,
            "connection_duration": connection_duration,
            "other_feature_1": tcp_window,
            "other_feature_2": ttl,
            "other_feature_3": tos,
            "other_feature_4": ip_length,
            "other_feature_5": src_ip_private,
            "other_feature_6": dst_ip_private,
            "other_feature_7": src_port_well_known,
            "other_feature_8": dst_port_well_known,
            "other_feature_9": src_port_is_ephemeral,
            "other_feature_10": dst_port_is_ephemeral,
            "other_feature_11": packet_rate,
            "other_feature_12": byte_rate,
            "other_feature_13": packet_length,
            "other_feature_14": ip_version,
            "other_feature_15": tcp_flags_raw,
            "other_feature_16": tcp_window_avg,
            "other_feature_17": 1 if is_tcp else 0,
            "other_feature_18": 1 if is_udp else 0,
        }
        return features

    def _normalize(self, features: Dict[str, float]) -> np.ndarray:
        output: List[float] = []
        for name in self.schema:
            value = features.get(name, 0.0)
            if (
                value is None
                or isinstance(value, float)
                and (math.isnan(value) or math.isinf(value))
            ):
                value = 0.0
            min_val, max_val = self.bounds.get(name, (0.0, 1.0))
            if max_val <= min_val:
                norm = 0.0
            else:
                clipped = min(max(float(value), min_val), max_val)
                norm = (clipped - min_val) / (max_val - min_val)
            output.append(norm)
        return np.array(output, dtype=np.float32)
