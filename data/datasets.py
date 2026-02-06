"""
数据集处理模块。

提供:
- load_labeled_dataset: 加载标记数据集
- split_dataset: 按70/15/15切分训练/验证/测试集
- IDSFeatureDataset: PyTorch Dataset封装
- create_dataloaders: 构建DataLoader并支持类别不平衡处理

使用示例:
    from data.datasets import load_labeled_dataset, split_dataset, create_dataloaders

    x, y, names = load_labeled_dataset("data/processed/features.csv", label_col="label")
    splits = split_dataset(x, y, split_ratio=(0.7, 0.15, 0.15), random_state=42)
    loaders = create_dataloaders(*splits, batch_size=128, balance=True)

错误处理说明:
    - 文件读取与数据格式错误将抛出RuntimeError
    - 数据维度不一致将抛出ValueError
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import logging

import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from utils.config import load_config


logger = logging.getLogger("ids.datasets")

ATTACK_TYPES = {
    0: "Normal",
    1: "DoS",
    2: "DDoS",
    3: "Probe",
    4: "PortScan",
    5: "BruteForce",
    6: "WebAttack",
    7: "SQLInjection",
    8: "XSS",
    9: "CommandInjection",
    10: "Malware",
    11: "Botnet",
    12: "Ransomware",
    13: "Trojan",
    14: "Backdoor",
    15: "PrivilegeEscalation",
    16: "CredentialTheft",
    17: "DataExfiltration",
    18: "DNSAttack",
    19: "MITM",
    20: "ARPSpoofing",
    21: "Phishing",
}


@dataclass
class DatasetSplits:
    train: Tuple[np.ndarray, np.ndarray]
    val: Tuple[np.ndarray, np.ndarray]
    test: Tuple[np.ndarray, np.ndarray]


def load_labeled_dataset(
    file_path: str,
    label_col: str = "label",
    drop_cols: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    加载标记数据集。

    Args:
        file_path: 数据文件路径（CSV）。
        label_col: 标签列名。
        drop_cols: 需要删除的列名。

    Returns:
        (features, labels, feature_names)
    """
    try:
        data = pd.read_csv(file_path)
    except Exception as exc:
        logger.exception("读取数据集失败: %s", file_path)
        raise RuntimeError("读取数据集失败") from exc

    if label_col not in data.columns:
        raise ValueError("label_col不存在")

    drop_cols = list(drop_cols) if drop_cols else []
    cols_to_drop = set(drop_cols + [label_col])
    feature_names = [c for c in data.columns if c not in cols_to_drop]
    if not feature_names:
        raise ValueError("特征列为空")

    features = data[feature_names].to_numpy(dtype=np.float32)
    labels = data[label_col].to_numpy(dtype=np.int64)
    return features, labels, feature_names


def split_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    random_state: int = 42,
    stratify: bool = True,
) -> DatasetSplits:
    """
    切分数据集为训练/验证/测试集。

    Args:
        features: 特征矩阵。
        labels: 标签向量。
        split_ratio: 训练/验证/测试比例。
        random_state: 随机种子。
        stratify: 是否按类别分层切分。

    Returns:
        DatasetSplits
    """
    if features.shape[0] != labels.shape[0]:
        raise ValueError("features与labels样本数不一致")
    if len(split_ratio) != 3 or not np.isclose(sum(split_ratio), 1.0):
        raise ValueError("split_ratio需为3元素且总和为1.0")

    train_ratio, val_ratio, test_ratio = split_ratio
    total = features.shape[0]
    train_size = int(round(total * train_ratio))
    val_size = int(round(total * val_ratio))
    test_size = total - train_size - val_size

    if train_size <= 0 or val_size <= 0 or test_size <= 0:
        raise ValueError("数据量不足以完成切分")

    x_train, x_temp, y_train, y_temp = _split_with_strategy(
        features,
        labels,
        train_size=train_size,
        random_state=random_state,
        stratify=stratify,
    )

    val_size_temp = int(round(len(x_temp) * (val_size / (val_size + test_size))))
    x_val, x_test, y_val, y_test = _split_with_strategy(
        x_temp,
        y_temp,
        train_size=val_size_temp,
        random_state=random_state,
        stratify=stratify,
    )

    return DatasetSplits(
        train=(x_train, y_train),
        val=(x_val, y_val),
        test=(x_test, y_test),
    )


def _split_with_strategy(
    features: np.ndarray,
    labels: np.ndarray,
    train_size: int,
    random_state: int,
    stratify: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total = features.shape[0]
    test_size = total - train_size
    if train_size <= 0 or test_size <= 0:
        raise ValueError("切分大小无效")

    can_stratify = stratify and _can_stratify(labels)
    if can_stratify:
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            train_size=train_size,
            test_size=test_size,
            random_state=random_state,
        )
        train_idx, test_idx = next(splitter.split(features, labels))
    else:
        rng = np.random.RandomState(random_state)
        indices = rng.permutation(total)
        train_idx = indices[:train_size]
        test_idx = indices[train_size:]

    return features[train_idx], features[test_idx], labels[train_idx], labels[test_idx]


def _can_stratify(labels: np.ndarray) -> bool:
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2:
        return False
    return bool(np.min(counts) >= 2)


class IDSFeatureDataset(Dataset):
    """
    PyTorch Dataset封装。

    Args:
        features: 特征矩阵。
        labels: 标签向量。
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        if features.shape[0] != labels.shape[0]:
            raise ValueError("features与labels样本数不一致")
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def _compute_sample_weights(labels: np.ndarray) -> np.ndarray:
    classes, counts = np.unique(labels, return_counts=True)
    class_weights = {cls: 1.0 / count for cls, count in zip(classes, counts)}
    return np.array([class_weights[label] for label in labels], dtype=np.float32)


def compute_class_weights(labels: np.ndarray, num_classes: Optional[int] = None) -> np.ndarray:
    if labels.size == 0:
        raise ValueError("labels不能为空")
    labels = labels.astype(np.int64)
    if num_classes is None:
        num_classes = int(labels.max()) + 1
    if num_classes <= 0:
        raise ValueError("num_classes必须为正整数")
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    total = float(counts.sum())
    weights = total / (counts * float(num_classes))
    return weights.astype(np.float32)


def select_features_rfe(
    features: np.ndarray,
    labels: np.ndarray,
    n_features: int,
    step: int = 1,
    estimator: Optional[LogisticRegression] = None,
    random_state: int = 42,
) -> Tuple[np.ndarray, List[int], RFE]:
    if features.shape[0] != labels.shape[0]:
        raise ValueError("features与labels样本数不一致")
    if n_features <= 0 or n_features > features.shape[1]:
        raise ValueError("n_features取值不合法")
    if step <= 0:
        raise ValueError("step必须为正整数")
    if estimator is None:
        estimator = LogisticRegression(
            max_iter=1000,
            multi_class="auto",
            solver="lbfgs",
            random_state=random_state,
        )
    selector = RFE(estimator=estimator, n_features_to_select=n_features, step=step)
    selector.fit(features, labels)
    mask = selector.get_support()
    selected_indices = np.flatnonzero(mask).tolist()
    return features[:, mask], selected_indices, selector


def smote_oversample(
    features: np.ndarray,
    labels: np.ndarray,
    target_ratio: float = 1.0,
    k_neighbors: int = 5,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    if features.shape[0] != labels.shape[0]:
        raise ValueError("features与labels样本数不一致")
    if target_ratio <= 0.0:
        raise ValueError("target_ratio必须大于0")
    if k_neighbors <= 0:
        raise ValueError("k_neighbors必须大于0")
    rng = np.random.RandomState(random_state)
    classes, counts = np.unique(labels, return_counts=True)
    max_count = int(np.max(counts)) if counts.size > 0 else 0
    target_count = int(round(max_count * target_ratio))
    if target_count <= 0:
        return features, labels
    synthetic_features: List[np.ndarray] = []
    synthetic_labels: List[np.ndarray] = []
    for cls, count in zip(classes, counts):
        if count >= target_count:
            continue
        class_mask = labels == cls
        class_features = features[class_mask]
        if class_features.shape[0] == 0:
            continue
        if class_features.shape[0] == 1:
            num_needed = target_count - count
            replicated = np.repeat(class_features, num_needed, axis=0)
            synthetic_features.append(replicated)
            synthetic_labels.append(np.full(num_needed, cls, dtype=np.int64))
            continue
        neighbor_count = min(k_neighbors + 1, class_features.shape[0])
        nn = NearestNeighbors(n_neighbors=neighbor_count)
        nn.fit(class_features)
        neighbors = nn.kneighbors(class_features, return_distance=False)
        num_needed = target_count - count
        for _ in range(num_needed):
            idx = rng.randint(0, class_features.shape[0])
            neighbor_pool = neighbors[idx][1:] if neighbors.shape[1] > 1 else neighbors[idx]
            if neighbor_pool.size == 0:
                neighbor_idx = idx
            else:
                neighbor_idx = rng.choice(neighbor_pool)
            sample = class_features[idx]
            neighbor = class_features[neighbor_idx]
            gap = rng.rand()
            synthetic = sample + gap * (neighbor - sample)
            synthetic_features.append(synthetic.reshape(1, -1))
            synthetic_labels.append(np.array([cls], dtype=np.int64))
    if not synthetic_features:
        return features, labels
    new_features = np.vstack([features, *synthetic_features]).astype(np.float32)
    new_labels = np.concatenate([labels, *synthetic_labels]).astype(np.int64)
    return new_features, new_labels


def create_dataloaders(
    train_split: Tuple[np.ndarray, np.ndarray],
    val_split: Tuple[np.ndarray, np.ndarray],
    test_split: Tuple[np.ndarray, np.ndarray],
    batch_size: int,
    balance: bool = False,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """
    创建DataLoader并支持类别不平衡处理。

    Args:
        train_split: 训练集。
        val_split: 验证集。
        test_split: 测试集。
        batch_size: 批大小。
        balance: 是否启用类别平衡采样。
        num_workers: DataLoader工作线程数。

    Returns:
        dict包含train/val/test的DataLoader。
    """
    train_ds = IDSFeatureDataset(*train_split)
    val_ds = IDSFeatureDataset(*val_split)
    test_ds = IDSFeatureDataset(*test_split)

    if balance:
        weights = _compute_sample_weights(train_split[1]).tolist()
        sampler = WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}


def load_from_config(
    file_path: str,
    label_col: str = "label",
    drop_cols: Optional[Sequence[str]] = None,
    config_path: str = "config.yaml",
    balance: bool = False,
    feature_select: bool = False,
    n_features: Optional[int] = None,
    smote: bool = False,
    smote_ratio: float = 1.0,
    smote_k_neighbors: int = 5,
    random_state: int = 42,
) -> Dict[str, DataLoader]:
    """
    通过配置文件加载并构建DataLoader。

    Args:
        file_path: 数据文件路径。
        label_col: 标签列名。
        drop_cols: 需要删除的列名。
        config_path: 配置文件路径。
        balance: 是否启用类别平衡采样。

    Returns:
        dict包含train/val/test的DataLoader。
    """
    config = load_config(config_path)
    data_cfg = config.get("data") or {}
    split_ratio = tuple(data_cfg.get("split_ratio", [0.7, 0.15, 0.15]))
    batch_size = int(data_cfg.get("batch_size", 128))

    features, labels, _ = load_labeled_dataset(
        file_path, label_col=label_col, drop_cols=drop_cols
    )
    if smote:
        features, labels = smote_oversample(
            features,
            labels,
            target_ratio=smote_ratio,
            k_neighbors=smote_k_neighbors,
            random_state=random_state,
        )
    if feature_select:
        if n_features is None:
            n_features = int(features.shape[1])
        features, _, _ = select_features_rfe(
            features,
            labels,
            n_features=n_features,
            random_state=random_state,
        )
    splits = split_dataset(features, labels, split_ratio=split_ratio)
    return create_dataloaders(
        splits.train,
        splits.val,
        splits.test,
        batch_size=batch_size,
        balance=balance,
    )
