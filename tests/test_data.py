import pytest
from scapy.all import Ether, ICMP, IP, TCP, UDP, wrpcap

from data.data_loader import DataLoader
from utils.config import Config, save_yaml
from utils.logger import get_logger


def _write_pcap(tmp_path, packets):
    pcap_path = tmp_path / "sample.pcap"
    wrpcap(str(pcap_path), packets)
    return pcap_path


def test_load_pcap_parses_tcp(tmp_path):
    pkt = (
        Ether()
        / IP(src="10.0.0.1", dst="10.0.0.2")
        / TCP(sport=1234, dport=80, flags="S")
    )
    pcap_path = _write_pcap(tmp_path, [pkt])

    loader = DataLoader()
    records = loader.load_pcap(str(pcap_path))

    assert len(records) == 1
    record = records[0]
    assert record.src_ip == "10.0.0.1"
    assert record.dst_ip == "10.0.0.2"
    assert record.protocol == "TCP"
    assert record.src_port == 1234
    assert record.dst_port == 80
    assert record.tcp_flags is not None


def test_load_pcap_parses_udp(tmp_path):
    pkt = Ether() / IP(src="10.0.0.3", dst="10.0.0.4") / UDP(sport=53, dport=5353)
    pcap_path = _write_pcap(tmp_path, [pkt])

    loader = DataLoader()
    records = loader.load_pcap(str(pcap_path))

    record = records[0]
    assert record.protocol == "UDP"
    assert record.src_port == 53
    assert record.dst_port == 5353


def test_load_pcap_parses_icmp(tmp_path):
    pkt = Ether() / IP(src="192.168.1.1", dst="192.168.1.2") / ICMP(type=8, code=0)
    pcap_path = _write_pcap(tmp_path, [pkt])

    loader = DataLoader()
    records = loader.load_pcap(str(pcap_path))

    record = records[0]
    assert record.protocol == "ICMP"
    assert record.icmp_type == 8
    assert record.icmp_code == 0


def test_load_pcap_with_limit(tmp_path):
    packets = [
        Ether() / IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=1, dport=2),
        Ether() / IP(src="10.0.0.3", dst="10.0.0.4") / UDP(sport=3, dport=4),
    ]
    pcap_path = _write_pcap(tmp_path, packets)

    loader = DataLoader()
    records = loader.load_pcap(str(pcap_path), limit=1)

    assert len(records) == 1


def test_load_pcap_missing_file():
    loader = DataLoader()
    with pytest.raises(RuntimeError):
        loader.load_pcap("missing.pcap")


def test_config_env_overrides(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_data = {
        "data": {"feature_dim": 41},
        "logging": {"level": "INFO", "log_file": str(tmp_path / "logs/app.log")},
        "environments": {"dev": {"data": {"feature_dim": 8}}},
    }
    save_yaml(config_data, str(config_path))

    monkeypatch.setenv("IDS_ENV", "dev")
    cfg = Config(path=str(config_path))
    assert cfg.get_section("data", "feature_dim") == 8

    monkeypatch.setenv("IDS__DATA__FEATURE_DIM", "12")
    cfg = Config(path=str(config_path))
    assert cfg.get_section("data", "feature_dim") == 12


def test_logger_writes_file(tmp_path):
    log_path = tmp_path / "logs/app.log"
    logger = get_logger(
        name="ids.test_logger",
        level="INFO",
        log_file=str(log_path),
        console=False,
    )
    logger.info("test-message")
    for handler in logger.handlers:
        handler.flush()
    content = log_path.read_text(encoding="utf-8")
    assert "test-message" in content
