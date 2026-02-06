from data.data_loader import PacketRecord
from models.feature_extractor import FeatureExtractor


def test_extract_shape_and_range():
    extractor = FeatureExtractor(config_path="config.yaml")
    record = PacketRecord(
        timestamp=1.0,
        src_ip="10.0.0.1",
        dst_ip="8.8.8.8",
        protocol="TCP",
        src_port=70000,
        dst_port=80,
        length=200,
        tcp_flags=2,
        tcp_window=1024,
        ttl=64,
        tos=0,
        ip_length=200,
        icmp_type=None,
        icmp_code=None,
    )

    features = extractor.extract(record)

    assert features.shape == (41,)
    assert float(features.min()) >= 0.0
    assert float(features.max()) <= 1.0


def test_extract_handles_missing_fields():
    extractor = FeatureExtractor(config_path="config.yaml")
    raw_sample = {"protocol": "UDP", "length": 0}

    features = extractor.extract(raw_sample)

    assert features.shape == (41,)
    assert float(features.min()) >= 0.0
    assert float(features.max()) <= 1.0


def test_inter_arrival_increases():
    extractor = FeatureExtractor(config_path="config.yaml")
    record1 = PacketRecord(
        timestamp=1.0,
        src_ip="10.0.0.1",
        dst_ip="10.0.0.2",
        protocol="UDP",
        src_port=53,
        dst_port=5353,
        length=100,
        tcp_flags=None,
        tcp_window=None,
        ttl=64,
        tos=0,
        ip_length=100,
        icmp_type=None,
        icmp_code=None,
    )
    record2 = PacketRecord(
        timestamp=2.0,
        src_ip="10.0.0.1",
        dst_ip="10.0.0.2",
        protocol="UDP",
        src_port=53,
        dst_port=5353,
        length=100,
        tcp_flags=None,
        tcp_window=None,
        ttl=64,
        tos=0,
        ip_length=100,
        icmp_type=None,
        icmp_code=None,
    )

    f1 = extractor.extract(record1)
    f2 = extractor.extract(record2)

    idx = extractor.schema.index("inter_packet_time")
    assert f1[idx] <= f2[idx]
