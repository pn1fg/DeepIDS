import numpy as np
import pandas as pd

from data.preprocessing import clean_data, engineer_features, build_time_windows


def test_clean_data_removes_duplicates_and_nans():
    df = pd.DataFrame(
        [
            {"timestamp": 1.0, "packet_length": 100, "label": 0},
            {"timestamp": 1.0, "packet_length": 100, "label": 0},
            {"timestamp": 2.0, "packet_length": np.nan, "label": 1},
        ]
    )

    cleaned = clean_data(df, required_cols=["timestamp", "packet_length", "label"])

    assert cleaned.shape[0] == 1


def test_engineer_features_selects_and_casts():
    df = pd.DataFrame(
        [
            {"timestamp": 1.0, "packet_length": "120", "label": "1"},
        ]
    )

    engineered = engineer_features(
        df, feature_cols=["packet_length"], label_col="label"
    )

    assert engineered["packet_length"].dtype.kind == "f"
    assert engineered["label"].dtype.kind in {"i", "u"}


def test_build_time_windows_shapes():
    df = pd.DataFrame(
        [
            {"timestamp": 1.0, "f1": 0.1, "label": 0},
            {"timestamp": 2.0, "f1": 0.2, "label": 0},
            {"timestamp": 3.0, "f1": 0.3, "label": 1},
            {"timestamp": 4.0, "f1": 0.4, "label": 1},
        ]
    )

    windows, labels = build_time_windows(df, label_col="label", window_size=2, stride=1)

    assert windows.shape == (3, 2, 1)
    assert labels.tolist() == [0, 1, 1]
