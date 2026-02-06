"""
数据验证与可视化工具模块。

提供:
- validate_data_quality: 验证数据质量并返回报告字典
- compute_basic_stats: 计算特征基础统计
- plot_feature_distributions: 绘制特征分布图并保存
- generate_quality_report_md: 生成Markdown质量报告文本

使用示例:
    import pandas as pd
    from utils.data_utils import (
        validate_data_quality,
        compute_basic_stats,
        plot_feature_distributions,
        generate_quality_report_md,
    )

    df = pd.read_csv("data/processed/features.csv")
    report = validate_data_quality(df, required_cols=["label"], label_col="label")
    stats = compute_basic_stats(df, feature_cols=[c for c in df.columns if c != "label"])
    plot_feature_distributions(df, feature_cols=list(stats.keys()), out_dir="outputs/figures")
    md = generate_quality_report_md(report, stats)

错误处理说明:
    - 输入为空、列缺失或类型错误将抛出ValueError
    - 可视化保存失败将抛出IOError
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Any
import logging
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


logger = logging.getLogger("ids.data_utils")


def validate_data_quality(
    df: pd.DataFrame,
    required_cols: Optional[Sequence[str]] = None,
    label_col: Optional[str] = None,
) -> Dict[str, Any]:
    """
    验证数据质量，输出基础质量指标。

    Args:
        df: 数据表。
        required_cols: 必需列列表。
        label_col: 标签列名，用于类别分布统计。

    Returns:
        报告字典，包含行数、缺失值统计、重复数、类别分布、异常值比例。
    """
    if df is None or df.empty:
        raise ValueError("输入数据为空")

    report: Dict[str, Any] = {}
    report["row_count"] = int(df.shape[0])
    report["col_count"] = int(df.shape[1])

    # 缺失值统计
    missing = df.isna().sum().to_dict()
    report["missing_counts"] = {str(k): int(v) for k, v in missing.items()}

    # 重复行统计
    report["duplicate_count"] = int(df.duplicated().sum())

    # 必需列检查
    if required_cols:
        missing_required = [c for c in required_cols if c not in df.columns]
        if missing_required:
            raise ValueError(f"缺少必需列: {missing_required}")

    # 类别分布
    if label_col and label_col in df.columns:
        counts = df[label_col].value_counts(dropna=False).to_dict()
        report["class_distribution"] = {
            int(k) if pd.notna(k) else -1: int(v) for k, v in counts.items()
        }

    # 异常值检测（IQR法，仅数值列）
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_ratio: Dict[str, float] = {}
    for col in numeric_cols:
        series = df[col].dropna().astype(float)
        if series.empty:
            outlier_ratio[col] = 0.0
            continue
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        ratio = float(((series < lower) | (series > upper)).mean())
        outlier_ratio[col] = ratio
    report["outlier_ratio"] = outlier_ratio

    return report


def compute_basic_stats(
    df: pd.DataFrame, feature_cols: Sequence[str]
) -> Dict[str, Dict[str, float]]:
    """
    计算特征的基础统计。

    Args:
        df: 数据表。
        feature_cols: 特征列名。

    Returns:
        每个特征的统计字典（mean/std/min/max）。
    """
    if df is None or df.empty:
        raise ValueError("输入数据为空")
    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"特征列不存在: {col}")

    stats: Dict[str, Dict[str, float]] = {}
    for col in feature_cols:
        series = df[col].astype(float)
        stats[col] = {
            "mean": float(series.mean()),
            "std": float(series.std(ddof=1)) if series.size > 1 else 0.0,
            "min": float(series.min()),
            "max": float(series.max()),
        }
    return stats


def plot_feature_distributions(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    out_dir: str,
    bins: int = 50,
    kde: bool = False,
) -> List[str]:
    """
    绘制特征分布直方图并保存。

    Args:
        df: 数据表。
        feature_cols: 需要绘制的特征列表。
        out_dir: 输出目录。
        bins: 直方图桶数。
        kde: 是否绘制核密度曲线。

    Returns:
        保存的文件路径列表。
    """
    if not feature_cols:
        raise ValueError("feature_cols不能为空")
    os.makedirs(out_dir, exist_ok=True)

    saved_paths: List[str] = []
    for col in feature_cols:
        if col not in df.columns:
            logger.warning("列不存在，跳过: %s", col)
            continue
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col].astype(float), bins=bins, kde=kde)
        plt.title(f"Distribution: {col}")
        plt.xlabel(col)
        plt.ylabel("count")
        path = os.path.join(out_dir, f"{col}_dist.png")
        try:
            plt.tight_layout()
            plt.savefig(path)
        except Exception as exc:
            plt.close()
            logger.exception("保存可视化失败: %s", path)
            raise IOError("保存可视化失败") from exc
        finally:
            plt.close()
        saved_paths.append(path)
    return saved_paths


def generate_quality_report_md(
    report: Dict[str, Any],
    stats: Optional[Dict[str, Dict[str, float]]] = None,
) -> str:
    """
    生成Markdown格式的数据质量报告文本。

    Args:
        report: 质量报告字典。
        stats: 基础统计字典。

    Returns:
        Markdown文本。
    """
    lines: List[str] = []
    lines.append("# 数据质量报告")
    lines.append("")
    lines.append(f"- 行数: {report.get('row_count')}")
    lines.append(f"- 列数: {report.get('col_count')}")
    lines.append(f"- 重复行数: {report.get('duplicate_count')}")
    lines.append("")

    lines.append("## 缺失值统计")
    missing = report.get("missing_counts", {}) or {}
    for k, v in missing.items():
        lines.append(f"- {k}: {v}")
    lines.append("")

    class_dist = report.get("class_distribution")
    if class_dist is not None:
        lines.append("## 类别分布")
        for k, v in class_dist.items():
            lines.append(f"- 类别 {k}: {v}")
        lines.append("")

    outliers = report.get("outlier_ratio", {}) or {}
    lines.append("## 异常值比例(IQR)")
    for k, v in outliers.items():
        lines.append(f"- {k}: {v:.4f}")
    lines.append("")

    if stats:
        lines.append("## 特征基础统计")
        for col, s in stats.items():
            line = (
                f"- {col}: mean={s['mean']:.4f}, std={s['std']:.4f}, "
                f"min={s['min']:.4f}, max={s['max']:.4f}"
            )
            lines.append(line)
        lines.append("")

    return "\n".join(lines)
