from __future__ import annotations

from inference.real_time_monitor import RealTimeMonitor
from models.deep_learning import DeepLearningClassifier, save_model
from utils.config import ATTACK_TYPES


def run_example() -> None:
    try:
        model = DeepLearningClassifier(
            input_dim=41,
            hidden_dims=[256, 128, 64],
            output_dim=len(ATTACK_TYPES),
            dropout=0.3,
        )
        save_path = "outputs/models/monitor_example.pth"
        save_model(model, save_path)

        monitor = RealTimeMonitor(
            model_path=save_path,
            bpf_filter="tcp",
        )
        monitor.start_monitoring(timeout=5)
        stats = monitor.get_statistics()
        _ = stats
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError("监控示例执行失败") from exc


def main() -> None:
    try:
        run_example()
    except RuntimeError as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
