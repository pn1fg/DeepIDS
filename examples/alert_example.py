from __future__ import annotations

from inference.alert_system import AlertSystem


def run_example() -> None:
    try:
        system = AlertSystem(log_dir="outputs/logs/alerts")
        detection = {
            "src_ip": "10.0.0.1",
            "dst_ip": "10.0.0.2",
            "attack_type": "DoS",
            "confidence": 0.92,
        }
        alert = system.generate_alert(detection)
        if alert is not None:
            _ = system.log_alert(alert)
            system.send_alert(alert)
        stats = system.get_alert_statistics()
        _ = stats
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError("告警示例执行失败") from exc


def main() -> None:
    try:
        run_example()
    except RuntimeError as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
