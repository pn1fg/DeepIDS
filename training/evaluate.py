"""
模型评估模块。

提供多分类评估指标、混淆矩阵、ROC与PR曲线绘制，以及评估报告生成。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import os

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
    auc,
    precision_recall_curve,
)
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt
import seaborn as sns

from models.deep_learning import DeepLearningClassifier
from utils.config import ATTACK_TYPES


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    average: str = "weighted",
) -> Dict[str, float]:
    """
    计算多分类指标。

    Args:
        y_true: 真实标签序列。
        y_pred: 预测标签序列。
        average: 平均方式。

    Returns:
        指标字典，包含accuracy/precision/recall/f1。
    """
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


def generate_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int,
) -> np.ndarray:
    """
    生成混淆矩阵。

    Args:
        y_true: 真实标签序列。
        y_pred: 预测标签序列。
        num_classes: 类别数量。

    Returns:
        混淆矩阵。
    """
    if num_classes <= 1:
        raise ValueError("num_classes必须大于1")
    labels = list(range(num_classes))
    return confusion_matrix(y_true, y_pred, labels=labels)


def generate_classification_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    target_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    生成分类报告。

    Args:
        y_true: 真实标签序列。
        y_pred: 预测标签序列。
        target_names: 类别名称。

    Returns:
        分类报告字典。
    """
    if target_names is None:
        return classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
    labels = list(range(len(target_names)))
    return classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: Sequence[str],
    output_path: str,
) -> str:
    """
    绘制混淆矩阵热力图。

    Args:
        matrix: 混淆矩阵。
        class_names: 类别名称。
        output_path: 输出路径。

    Returns:
        保存路径。
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    try:
        plt.savefig(output_path)
    except Exception as exc:
        raise IOError("保存混淆矩阵失败") from exc
    finally:
        plt.close()
    return output_path


def plot_roc_curve(
    y_true: Sequence[int],
    y_scores: np.ndarray,
    num_classes: int,
    class_names: Sequence[str],
    output_path: str,
) -> str:
    """
    绘制多分类ROC-AUC曲线。

    Args:
        y_true: 真实标签序列。
        y_scores: 预测概率矩阵，形状为(n_samples, num_classes)。
        num_classes: 类别数量。
        class_names: 类别名称。
        output_path: 输出路径。

    Returns:
        保存路径。
    """
    if y_scores.shape[1] != num_classes:
        raise ValueError("y_scores与num_classes不匹配")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))
    plt.figure(figsize=(7, 6))
    for idx in range(num_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, idx], y_scores[:, idx])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[idx]} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    try:
        plt.savefig(output_path)
    except Exception as exc:
        raise IOError("保存ROC曲线失败") from exc
    finally:
        plt.close()
    return output_path


def plot_precision_recall_curve(
    y_true: Sequence[int],
    y_scores: np.ndarray,
    num_classes: int,
    class_names: Sequence[str],
    output_path: str,
) -> str:
    """
    绘制多分类Precision-Recall曲线。

    Args:
        y_true: 真实标签序列。
        y_scores: 预测概率矩阵，形状为(n_samples, num_classes)。
        num_classes: 类别数量。
        class_names: 类别名称。
        output_path: 输出路径。

    Returns:
        保存路径。
    """
    if y_scores.shape[1] != num_classes:
        raise ValueError("y_scores与num_classes不匹配")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))
    plt.figure(figsize=(7, 6))
    for idx in range(num_classes):
        precision, recall, _ = precision_recall_curve(y_bin[:, idx], y_scores[:, idx])
        plt.plot(recall, precision, label=class_names[idx])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.tight_layout()
    try:
        plt.savefig(output_path)
    except Exception as exc:
        raise IOError("保存PR曲线失败") from exc
    finally:
        plt.close()
    return output_path


def evaluate_model(
    model: DeepLearningClassifier,
    data_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    device: Optional[torch.device] = None,
    num_classes: int = len(ATTACK_TYPES),
    class_names: Optional[Sequence[str]] = None,
    loss_fn: Optional[torch.nn.Module] = None,
) -> Dict[str, Any]:
    """
    在测试集上评估模型并返回所有指标。

    Args:
        model: 模型实例。
        data_loader: 测试数据加载器。
        device: 评估设备。
        num_classes: 类别数量。
        class_names: 类别名称。
        loss_fn: 损失函数。

    Returns:
        评估结果字典。
    """
    if class_names is None:
        class_names = list(ATTACK_TYPES.values())
    if len(class_names) != num_classes:
        raise ValueError("class_names长度必须与num_classes一致")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    if loss_fn is None:
        loss_fn = torch.nn.CrossEntropyLoss()

    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    total_loss = 0.0
    total_samples = 0
    try:
        with torch.no_grad():
            for features, labels in data_loader:
                features = features.to(device)
                labels = labels.to(device)
                logits = model(features)
                loss = loss_fn(logits, labels)
                total_loss += float(loss.item()) * labels.size(0)
                total_samples += int(labels.size(0))
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())
    except Exception as exc:
        raise RuntimeError("评估过程发生异常") from exc

    if total_samples == 0:
        raise ValueError("评估数据为空")

    logits_np = torch.cat(all_logits, dim=0).numpy()
    labels_np = torch.cat(all_labels, dim=0).numpy()
    probs = torch.softmax(torch.from_numpy(logits_np), dim=1).numpy()
    preds = np.argmax(probs, axis=1)

    avg_loss = total_loss / max(total_samples, 1)
    metrics = compute_metrics(labels_np.tolist(), preds.tolist())
    per_class = precision_recall_fscore_support(
        labels_np, preds, labels=list(range(num_classes)), zero_division=0
    )
    per_class_metrics = {
        class_names[i]: {
            "precision": float(per_class[0][i]),
            "recall": float(per_class[1][i]),
            "f1_score": float(per_class[2][i]),
            "support": int(per_class[3][i]),
        }
        for i in range(num_classes)
    }
    report = generate_classification_report(labels_np, preds, target_names=class_names)
    matrix = generate_confusion_matrix(labels_np, preds, num_classes=num_classes)

    return {
        "loss": float(avg_loss),
        "metrics": metrics,
        "per_class_metrics": per_class_metrics,
        "classification_report": report,
        "confusion_matrix": matrix,
        "y_true": labels_np,
        "y_pred": preds,
        "y_scores": probs,
        "class_names": list(class_names),
    }


def generate_evaluation_report(
    evaluation: Dict[str, Any],
    output_dir: str,
) -> Dict[str, Any]:
    """
    生成完整评估报告（文本与图表）。

    Args:
        evaluation: evaluate_model返回的结果。
        output_dir: 输出目录。

    Returns:
        报告信息字典。
    """
    os.makedirs(output_dir, exist_ok=True)
    class_names = evaluation["class_names"]
    num_classes = len(class_names)

    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    roc_path = os.path.join(output_dir, "roc_curve.png")
    pr_path = os.path.join(output_dir, "precision_recall_curve.png")
    report_path = os.path.join(output_dir, "evaluation_report.txt")

    plot_confusion_matrix(evaluation["confusion_matrix"], class_names, cm_path)
    plot_roc_curve(
        evaluation["y_true"],
        evaluation["y_scores"],
        num_classes,
        class_names,
        roc_path,
    )
    plot_precision_recall_curve(
        evaluation["y_true"],
        evaluation["y_scores"],
        num_classes,
        class_names,
        pr_path,
    )

    metrics = evaluation["metrics"]
    report_lines = [
        "Evaluation Summary",
        f"Loss: {evaluation['loss']:.6f}",
        f"Accuracy: {metrics['accuracy']:.6f}",
        f"Precision: {metrics['precision']:.6f}",
        f"Recall: {metrics['recall']:.6f}",
        f"F1-Score: {metrics['f1_score']:.6f}",
        "",
        "Classification Report",
    ]
    report_text = classification_report(
        evaluation["y_true"],
        evaluation["y_pred"],
        labels=list(range(num_classes)),
        target_names=class_names,
        zero_division=0,
    )
    report_lines.append(report_text)

    try:
        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(report_lines))
    except OSError as exc:
        raise IOError("写入评估报告失败") from exc

    return {
        "report_path": report_path,
        "confusion_matrix_path": cm_path,
        "roc_curve_path": roc_path,
        "precision_recall_curve_path": pr_path,
    }
