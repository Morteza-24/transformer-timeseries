"""Bar charts for Transformixer paper results from experimental_results.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MODEL_ORDER = [
    "RLinear",
    "PatchTST",
    "iTransformer",
    "xLSTM-Mixer",
    "Transformixer",
    "Transformixer+PE",
]
DATASETS = ["ECL", "Traffic"]
HIGHLIGHT = {"Transformixer", "Transformixer+PE"}


def load_metrics(json_path: Path) -> dict[str, dict[str, dict[str, float]]]:
    rows = json.loads(json_path.read_text())
    out: dict[str, dict[str, dict[str, float]]] = {}
    for row in rows:
        out.setdefault(row["dataset"], {})[row["model"]] = row["metrics"]
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
    n_models = len(MODEL_ORDER)
    x = np.arange(n_models)
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    colors = ["#4C72B0", "#DD8452"]
    all_values: list[float] = []

    for i, dataset in enumerate(DATASETS):
        values = [data[dataset][m][metric] for m in MODEL_ORDER]
        all_values.extend(values)
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, values, width, label=dataset, color=colors[i], zorder=2)
        for bar, model in zip(bars, MODEL_ORDER):
            if model in HIGHLIGHT:
                bar.set_edgecolor("#222222")
                bar.set_linewidth(1.6)
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.0015 if metric == "mse" else 0.0008,
                f"{height:.4f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                rotation=90,
            )

    y_min, y_max = _zoom_limits(all_values, metric)
    ax.set_ylim(y_min, y_max)

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, rotation=22, ha="right")
    ax.set_ylabel(metric.upper())
    ax.set_title(
        f"Long-term forecasting ({metric.upper()}), lookback/horizon = 96\n"
        "(y-axis zoomed; Informer omitted for scale)"
    )
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.6, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path} and {out_path.with_suffix('.pdf')}")


def plot_relative_to_xlstm(
    data: dict[str, dict[str, dict[str, float]]],
    out_path: Path,
) -> None:
    """Percent MSE change of both Transformixer variants vs xLSTM-Mixer."""
    variants = [
        ("Transformixer", "no PE"),
        ("Transformixer+PE", "+ PE"),
    ]
    x = np.arange(len(DATASETS))
    width = 0.32

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    for i, (model, label) in enumerate(variants):
        deltas = []
        for dataset in DATASETS:
            base = data[dataset]["xLSTM-Mixer"]["mse"]
            ours = data[dataset][model]["mse"]
            deltas.append(100.0 * (ours - base) / base)
        offset = (i - 0.5) * width
        colors = ["#55A868" if d <= 0 else "#C44E52" for d in deltas]
        bars = ax.bar(x + offset, deltas, width, label=label, color=colors, zorder=2)
        for bar, delta in zip(bars, deltas):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                delta + (0.25 if delta >= 0 else -0.45),
                f"{delta:+.2f}%",
                ha="center",
                va="bottom" if delta >= 0 else "top",
                fontsize=8,
            )

    ax.axhline(0.0, color="#333333", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.set_ylabel(r"MSE change vs xLSTM-Mixer (%)")
    ax.set_title("Relative MSE: Transformixer variants")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.6, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path} and {out_path.with_suffix('.pdf')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "experimental_results.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "figures",
    )
    args = parser.parse_args()

    data = load_metrics(args.json)
    plot_grouped_bars(data, "mse", args.out_dir / "results_mse.png")
    plot_grouped_bars(data, "mae", args.out_dir / "results_mae.png")
    plot_relative_to_xlstm(data, args.out_dir / "relative_mse_vs_xlstm.png")


if __name__ == "__main__":
    main()
