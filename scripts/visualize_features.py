"""
特征分布可视化脚本。

使用示例:
    .\.venv\Scripts\python scripts\visualize_features.py --input data/processed/features.csv --label-col label
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from utils.data_utils import compute_basic_stats, plot_feature_distributions, validate_data_quality, generate_quality_report_md


def main() -> None:
    parser = argparse.ArgumentParser(description="特征分布可视化与质量报告生成")
    parser.add_argument("--input", required=True, help="输入CSV文件")
    parser.add_argument("--label-col", default="label", help="标签列名")
    parser.add_argument("--out-dir", default="outputs/figures", help="图像输出目录")
    parser.add_argument("--report-path", default="outputs/reports/data_quality_report.md", help="报告输出路径")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    feature_cols = [c for c in df.columns if c != args.label_col]

    report = validate_data_quality(df, required_cols=[args.label_col], label_col=args.label_col)
    stats = compute_basic_stats(df, feature_cols)
    plot_feature_distributions(df, feature_cols, out_dir=args.out_dir)

    report_text = generate_quality_report_md(report, stats)
    os.makedirs(os.path.dirname(args.report_path), exist_ok=True)
    with open(args.report_path, "w", encoding="utf-8") as f:
        f.write(report_text)


if __name__ == "__main__":
    main()
