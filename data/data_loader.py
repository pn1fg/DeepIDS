"""
数据加载模块。

提供:
- DataLoader: 支持从PCAP文件与实时接口捕获网络流量，并解析IP/TCP/UDP/ICMP层

使用示例:
    from data.data_loader import DataLoader

    loader = DataLoader()
    records = loader.load_pcap("data/raw/sample.pcap", limit=100)

    # 实时捕获（可能需要管理员权限与Npcap等驱动）
    # live_records = loader.capture_live(interface="Ethernet", timeout=10, count=50)

错误处理说明:
    - 文件读取、实时捕获与数据包解析异常将抛出RuntimeError
    - 详细错误信息通过日志记录
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import logging

from scapy.all import IP, IPv6, TCP, UDP, ICMP, rdpcap, sniff


logger = logging.getLogger("ids.data_loader")


@dataclass
class PacketRecord:
    timestamp: float
    src_ip: str
    dst_ip: str
    protocol: str
    src_port: Optional[int]
    dst_port: Optional[int]
    length: int
    tcp_flags: Optional[int]
    tcp_window: Optional[int]
    ttl: Optional[int]
    tos: Optional[int]
    ip_length: Optional[int]
    icmp_type: Optional[int]
    icmp_code: Optional[int]


class DataLoader:
    """
    数据加载器，解析网络数据包为结构化记录。

    使用示例:
        loader = DataLoader()
        records = loader.load_pcap("data/raw/sample.pcap")
    """

    def load_pcap(
        self, file_path: str, limit: Optional[int] = None
    ) -> List[PacketRecord]:
        """
        从PCAP文件加载并解析数据包。

        Args:
            file_path: PCAP文件路径。
            limit: 解析的最大数据包数量。

        Returns:
            PacketRecord列表。
        """
        try:
            packets = rdpcap(file_path)
        except Exception as exc:
            logger.error("读取PCAP失败: %s", file_path)
            raise RuntimeError("读取PCAP失败") from exc

        records: List[PacketRecord] = []
        count = 0
        for pkt in packets:
            if limit is not None and count >= limit:
                break
            try:
                rec = self._parse_packet(pkt)
                records.append(rec)
                count += 1
            except Exception as exc:
                logger.warning("解析数据包失败，已跳过: %s", exc)
                continue
        return records

    def capture_live(
        self,
        interface: Optional[str] = None,
        timeout: Optional[int] = None,
        count: Optional[int] = None,
    ) -> List[PacketRecord]:
        """
        从实时网络接口捕获并解析数据包。

        Args:
            interface: 接口名称。
            timeout: 捕获超时时间（秒）。
            count: 最大捕获数量。

        Returns:
            PacketRecord列表。
        """
        try:
            packets = sniff(iface=interface, timeout=timeout, count=count)
        except Exception as exc:
            logger.error("实时捕获失败: iface=%s", interface)
            raise RuntimeError("实时捕获失败") from exc

        records: List[PacketRecord] = []
        for pkt in packets:
            try:
                rec = self._parse_packet(pkt)
                records.append(rec)
            except Exception as exc:
                logger.warning("解析数据包失败，已跳过: %s", exc)
                continue
        return records

    def _parse_packet(self, pkt) -> PacketRecord:
        """
        解析单个数据包。

        Args:
            pkt: Scapy数据包实例。

        Returns:
            PacketRecord。
        """
        timestamp = float(getattr(pkt, "time", 0.0))
        length = int(len(pkt))

        src_ip = ""
        dst_ip = ""
        protocol = "OTHER"
        src_port: Optional[int] = None
        dst_port: Optional[int] = None
        tcp_flags: Optional[int] = None
        tcp_window: Optional[int] = None
        ttl: Optional[int] = None
        tos: Optional[int] = None
        ip_length: Optional[int] = None
        icmp_type: Optional[int] = None
        icmp_code: Optional[int] = None

        if IP in pkt:
            src_ip = pkt[IP].src
            dst_ip = pkt[IP].dst
            ttl = int(getattr(pkt[IP], "ttl", 0))
            tos = int(getattr(pkt[IP], "tos", 0))
            ip_length = int(getattr(pkt[IP], "len", length))
        elif IPv6 in pkt:
            src_ip = pkt[IPv6].src
            dst_ip = pkt[IPv6].dst

        if TCP in pkt:
            protocol = "TCP"
            src_port = int(pkt[TCP].sport)
            dst_port = int(pkt[TCP].dport)
            tcp_flags = int(pkt[TCP].flags)
            tcp_window = int(getattr(pkt[TCP], "window", 0))
        elif UDP in pkt:
            protocol = "UDP"
            src_port = int(pkt[UDP].sport)
            dst_port = int(pkt[UDP].dport)
        elif ICMP in pkt:
            protocol = "ICMP"
            icmp_type = int(getattr(pkt[ICMP], "type", 0))
            icmp_code = int(getattr(pkt[ICMP], "code", 0))

        return PacketRecord(
            timestamp=timestamp,
            src_ip=src_ip,
            dst_ip=dst_ip,
            protocol=protocol,
            src_port=src_port,
            dst_port=dst_port,
            length=length,
            tcp_flags=tcp_flags,
            tcp_window=tcp_window,
            ttl=ttl,
            tos=tos,
            ip_length=ip_length,
            icmp_type=icmp_type,
            icmp_code=icmp_code,
        )
