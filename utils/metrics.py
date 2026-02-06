from __future__ import annotations

from typing import Dict, Sequence

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def calculate_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    average: str = "weighted",
) -> Dict[str, float]:
    if len(y_true) == 0:
        raise ValueError("y_true不能为空")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true与y_pred长度不一致")
    accuracy = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    return {
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
    }
