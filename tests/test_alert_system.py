from inference.alert_system import AlertSystem


def test_classify_alert_level():
    system = AlertSystem()
    assert system.classify_alert_level(0.91) == "严重"
    assert system.classify_alert_level(0.71) == "高"
    assert system.classify_alert_level(0.51) == "中"
    assert system.classify_alert_level(0.2) == "低"


def test_generate_and_deduplicate(tmp_path):
    system = AlertSystem(log_dir=str(tmp_path), dedup_window_seconds=60)
    detection = {
        "src_ip": "10.0.0.1",
        "dst_ip": "10.0.0.2",
        "attack_type": "DoS",
        "confidence": 0.95,
    }
    alert = system.generate_alert(detection)
    assert alert is not None
    log_path = system.log_alert(alert)
    assert log_path.endswith(".jsonl")

    alert_dup = system.generate_alert(detection)
    assert alert_dup is None


def test_statistics_update(tmp_path):
    system = AlertSystem(log_dir=str(tmp_path))
    detection = {
        "src_ip": "192.168.1.1",
        "dst_ip": "8.8.8.8",
        "attack_type": "Probe",
        "confidence": 0.8,
    }
    alert = system.generate_alert(detection)
    assert alert is not None
    _ = system.log_alert(alert)
    stats = system.get_alert_statistics()
    assert stats["total_alerts"] == 1
    assert stats["attack_distribution"]["Probe"] == 1
