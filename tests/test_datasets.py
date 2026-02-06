import numpy as np
import pandas as pd

from data.datasets import (
    IDSFeatureDataset,
    load_labeled_dataset,
    split_dataset,
    create_dataloaders,
)


def _write_csv(tmp_path, rows):
    file_path = tmp_path / "features.csv"
    pd.DataFrame(rows).to_csv(file_path, index=False)
    return file_path


def test_load_labeled_dataset(tmp_path):
    rows = [
        {"f1": 0.1, "f2": 0.2, "label": 0},
        {"f1": 0.3, "f2": 0.4, "label": 1},
    ]
    file_path = _write_csv(tmp_path, rows)

    features, labels, names = load_labeled_dataset(str(file_path), label_col="label")

    assert features.shape == (2, 2)
    assert labels.tolist() == [0, 1]
    assert names == ["f1", "f2"]


def test_split_dataset_ratio():
    features = np.random.rand(100, 5).astype(np.float32)
    labels = np.array([0] * 50 + [1] * 50)
    splits = split_dataset(
        features, labels, split_ratio=(0.7, 0.15, 0.15), random_state=1
    )

    assert splits.train[0].shape[0] == 70
    assert splits.val[0].shape[0] == 15
    assert splits.test[0].shape[0] == 15


def test_dataset_len_and_getitem():
    features = np.random.rand(10, 3).astype(np.float32)
    labels = np.arange(10)
    ds = IDSFeatureDataset(features, labels)

    assert len(ds) == 10
    x, y = ds[0]
    assert x.shape[0] == 3
    assert int(y.item()) == 0


def test_create_dataloaders_with_balance():
    features = np.random.rand(20, 4).astype(np.float32)
    labels = np.array([0] * 18 + [1] * 2)
    splits = split_dataset(
        features, labels, split_ratio=(0.7, 0.15, 0.15), random_state=2
    )

    loaders = create_dataloaders(
        splits.train, splits.val, splits.test, batch_size=4, balance=True
    )

    batch = next(iter(loaders["train"]))
    assert batch[0].shape[0] == 4
