"""Grouped bar charts for mixer ablation results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

VARIANT_ORDER = [
    ("xlstm_full", "xLSTM FULL"),
    ("tf_no_pe", "Transformixer (no PE)"),
    ("tf_with_pe", "Transformixer (+ PE)"),
]
DATASETS = [("ECL", "ECL"), ("Traffic", "Traffic")]
HIGHLIGHT = {"tf_no_pe", "tf_with_pe"}


def load_ablation(csv_path: Path) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            dataset = "ECL" if row["dataset"] == "Electricity" else row["dataset"]
            out.setdefault(dataset, {})[row["variant"]] = {
                "mse": float(row["mse"]),
                "mae": float(row["mae"]),
            }
    return out


def _zoom_limits(values: list[float], metric: str) -> tuple[float, float]:
    vmin, vmax = min(values), max(values)
    span = max(vmax - vmin, 1e-4)
    pad = max(0.08 * span, 0.002 if metric == "mse" else 0.001)
    return vmin - pad, vmax + pad


def plot_grouped_bars(
    data: dict[str, dict[str, dict[str, float]]],
    metric: str,
    out_path: Path,
) -> None:
    n_variants = len(VARIANT_ORDER)
    x = np.arange(n_variants)
    width = 0.35
    colors = ["#4C72B0", "#DD8452"]
    all_values: list[float] = []

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, (dataset_key, dataset_label) in enumerate(DATASETS):
        values = [data[dataset_key][variant_id][metric] for variant_id, _ in VARIANT_ORDER]
        all_values.extend(values)
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, values, width, label=dataset_label, color=colors[i], zorder=2)
        for bar, (variant_id, _) in zip(bars, VARIANT_ORDER):
            if variant_id in HIGHLIGHT:
                bar.set_edgecolor("#222222")
                bar.set_linewidth(1.6)
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.0015 if metric == "mse" else 0.0008,
                f"{height:.4f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    y_min, y_max = _zoom_limits(all_values, metric)
    ax.set_ylim(y_min, y_max)

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in VARIANT_ORDER], rotation=15, ha="right")
    ax.set_ylabel(metric.upper())
    ax.set_title(
        f"Controlled mixer ablation ({metric.upper()}), lookback/horizon = 96\n"
        "(y-axis zoomed; NLinear-only omitted for scale)"
    )
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.6, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(bottom=0.22, top=0.82)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path} and {out_path.with_suffix('.pdf')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "models/transformixer/outputs/mixer_ablation_results.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "figures",
    )
    args = parser.parse_args()

    data = load_ablation(args.csv)
    plot_grouped_bars(data, "mse", args.out_dir / "ablation_mse.png")
    plot_grouped_bars(data, "mae", args.out_dir / "ablation_mae.png")


if __name__ == "__main__":
    main()
