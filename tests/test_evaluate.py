import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.deep_learning import DeepLearningClassifier
from training.evaluate import (
    compute_metrics,
    evaluate_model,
    generate_classification_report,
    generate_confusion_matrix,
    generate_evaluation_report,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
)


def test_compute_metrics_values():
    y_true = [0, 1, 1, 2]
    y_pred = [0, 1, 0, 2]
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 0.75
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0


def test_generate_confusion_matrix_shape():
    matrix = generate_confusion_matrix([0, 1, 1], [0, 1, 0], num_classes=3)
    assert matrix.shape == (3, 3)


def test_generate_classification_report_keys():
    report = generate_classification_report(
        [0, 1, 1], [0, 1, 0], target_names=["a", "b"]
    )
    assert "accuracy" in report


def test_plot_functions_create_files(tmp_path):
    class_names = ["n", "d", "p"]
    matrix = np.array([[2, 0, 0], [0, 1, 1], [0, 0, 2]])
    y_true = [0, 1, 2, 0, 2]
    y_scores = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.1, 0.2, 0.7],
            [0.8, 0.1, 0.1],
            [0.05, 0.2, 0.75],
        ]
    )

    cm_path = tmp_path / "cm.png"
    roc_path = tmp_path / "roc.png"
    pr_path = tmp_path / "pr.png"

    plot_confusion_matrix(matrix, class_names, str(cm_path))
    plot_roc_curve(
        y_true,
        y_scores,
        num_classes=3,
        class_names=class_names,
        output_path=str(roc_path),
    )
    plot_precision_recall_curve(
        y_true,
        y_scores,
        num_classes=3,
        class_names=class_names,
        output_path=str(pr_path),
    )

    assert cm_path.exists()
    assert roc_path.exists()
    assert pr_path.exists()


def test_evaluate_model_and_report(tmp_path):
    torch.manual_seed(0)
    features = torch.randn(40, 41)
    num_classes = DeepLearningClassifier().output_dim
    labels = torch.randint(0, num_classes, (40,))
    loader = DataLoader(TensorDataset(features, labels), batch_size=16)
    model = DeepLearningClassifier()

    evaluation = evaluate_model(
        model=model, data_loader=loader, num_classes=num_classes
    )
    assert evaluation["confusion_matrix"].shape == (num_classes, num_classes)
    assert evaluation["y_scores"].shape[1] == num_classes

    report = generate_evaluation_report(evaluation, output_dir=str(tmp_path))
    assert report["report_path"]
    assert report["confusion_matrix_path"]
    assert report["roc_curve_path"]
    assert report["precision_recall_curve_path"]
