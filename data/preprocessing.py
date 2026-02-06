"""
数据预处理管道模块。

提供:
- clean_data: 去重与格式清理
- engineer_features: 特征选择与类型转换
- build_time_windows: 时间序列窗口构建

使用示例:
    import pandas as pd
    from data.preprocessing import clean_data, engineer_features, build_time_windows

    raw = pd.DataFrame(
        [
            {
                "timestamp": 1.0,
                "src_ip": "10.0.0.1",
                "dst_ip": "10.0.0.2",
                "packet_length": 120,
                "label": 0,
            },
            {
                "timestamp": 1.5,
                "src_ip": "10.0.0.1",
                "dst_ip": "10.0.0.2",
                "packet_length": 120,
                "label": 0,
            },
        ]
    )
    cleaned = clean_data(raw)
    features = engineer_features(cleaned, feature_cols=["packet_length"], label_col="label")
    windows, labels = build_time_windows(features, label_col="label", window_size=2)

错误处理说明:
    - 输入为空或缺少关键列将抛出ValueError
    - 处理过程中异常将抛出RuntimeError
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple
import logging

import numpy as np
import pandas as pd


logger = logging.getLogger("ids.preprocessing")


def clean_data(
    data: pd.DataFrame,
    required_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    数据清理，移除重复与格式错误的样本。

    Args:
        data: 原始数据表。
        required_cols: 必需列名。

    Returns:
        清理后的DataFrame。
    """
    try:
        if data is None or data.empty:
            raise ValueError("输入数据为空")

        cleaned = data.copy()
        cleaned = cleaned.drop_duplicates()

        if required_cols:
            missing = [col for col in required_cols if col not in cleaned.columns]
            if missing:
                raise ValueError("缺少必需列")

        cleaned = cleaned.replace([np.inf, -np.inf], np.nan).dropna()
        return cleaned
    except Exception as exc:
        logger.exception("数据清理失败")
        raise RuntimeError("数据清理失败") from exc


def engineer_features(
    data: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str = "label",
    cast_float: bool = True,
) -> pd.DataFrame:
    """
    特征工程：特征选择与类型转换。

    Args:
        data: 清理后的数据表。
        feature_cols: 特征列名列表。
        label_col: 标签列名。
        cast_float: 是否将特征转为float。

    Returns:
        仅包含特征与标签的DataFrame。
    """
    try:
        if data is None or data.empty:
            raise ValueError("输入数据为空")
        if not feature_cols:
            raise ValueError("feature_cols不能为空")
        if label_col not in data.columns:
            raise ValueError("label_col不存在")

        missing = [col for col in feature_cols if col not in data.columns]
        if missing:
            raise ValueError("特征列缺失")

        subset = data[list(feature_cols) + [label_col]].copy()
        if cast_float:
            subset[feature_cols] = subset[feature_cols].astype(float)
        subset[label_col] = subset[label_col].astype(int)
        return subset
    except Exception as exc:
        logger.exception("特征工程失败")
        raise RuntimeError("特征工程失败") from exc


def build_time_windows(
    data: pd.DataFrame,
    label_col: str,
    window_size: int,
    stride: int = 1,
    timestamp_col: str = "timestamp",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    构建时间序列滑动窗口。

    Args:
        data: 特征与标签数据表。
        label_col: 标签列名。
        window_size: 窗口长度。
        stride: 滑动步长。
        timestamp_col: 时间戳列名。

    Returns:
        (windows, labels)
    """
    try:
        if data is None or data.empty:
            raise ValueError("输入数据为空")
        if window_size <= 0 or stride <= 0:
            raise ValueError("window_size与stride必须为正整数")
        if label_col not in data.columns:
            raise ValueError("label_col不存在")

        sorted_data = (
            data.sort_values(by=timestamp_col)
            if timestamp_col in data.columns
            else data
        )
        feature_cols = [
            c for c in sorted_data.columns if c != label_col and c != timestamp_col
        ]
        if not feature_cols:
            raise ValueError("特征列为空")

        values = sorted_data[feature_cols].to_numpy(dtype=np.float32)
        labels = sorted_data[label_col].to_numpy(dtype=np.int64)
        windows: List[np.ndarray] = []
        win_labels: List[int] = []

        for start in range(0, len(values) - window_size + 1, stride):
            end = start + window_size
            windows.append(values[start:end])
            win_labels.append(int(labels[end - 1]))

        if not windows:
            raise ValueError("窗口大小超过数据长度")

        return np.stack(windows, axis=0), np.array(win_labels, dtype=np.int64)
    except Exception as exc:
        logger.exception("时间序列处理失败")
        raise RuntimeError("时间序列处理失败") from exc
