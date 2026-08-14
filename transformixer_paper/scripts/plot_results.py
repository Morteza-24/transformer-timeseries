"""Bar charts for Transformixer paper results from experimental_results.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MODEL_ORDER = [
    "RLinear",
    "Informer",
    "PatchTST",
    "iTransformer",
    "xLSTM-Mixer",
    "Transformixer",
]
DATASETS = ["ECL", "Traffic"]
HIGHLIGHT = "Transformixer"


def load_metrics(json_path: Path) -> dict[str, dict[str, dict[str, float]]]:
    rows = json.loads(json_path.read_text())
    out: dict[str, dict[str, dict[str, float]]] = {}
    for row in rows:
        out.setdefault(row["dataset"], {})[row["model"]] = row["metrics"]
    return out


def plot_grouped_bars(
    data: dict[str, dict[str, dict[str, float]]],
    metric: str,
    out_path: Path,
) -> None:
    n_models = len(MODEL_ORDER)
    n_datasets = len(DATASETS)
    x = np.arange(n_models)
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    colors = ["#4C72B0", "#DD8452"]

    for i, dataset in enumerate(DATASETS):
        values = [data[dataset][m][metric] for m in MODEL_ORDER]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, values, width, label=dataset, color=colors[i], zorder=2)
        for bar, model in zip(bars, MODEL_ORDER):
            if model == HIGHLIGHT:
                bar.set_edgecolor("#222222")
                bar.set_linewidth(1.6)

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, rotation=20, ha="right")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Long-term forecasting ({metric.upper()}), lookback/horizon = 96")
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
    """Percent MSE change of Transformixer vs xLSTM-Mixer."""
    labels = []
    deltas = []
    for dataset in DATASETS:
        base = data[dataset]["xLSTM-Mixer"]["mse"]
        ours = data[dataset]["Transformixer"]["mse"]
        labels.append(dataset)
        deltas.append(100.0 * (ours - base) / base)

    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    colors = ["#C44E52" if d > 0 else "#55A868" for d in deltas]
    ax.bar(labels, deltas, color=colors, width=0.55, zorder=2)
    ax.axhline(0.0, color="#333333", linewidth=1.0)
    ax.set_ylabel(r"Transformixer MSE vs xLSTM-Mixer (%)")
    # ax.set_title("Relative MSE change")
    ax.grid(axis="y", linestyle=":", alpha=0.6, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for label, delta in zip(labels, deltas):
        ax.text(
            label,
            delta + (0.4 if delta >= 0 else -0.7),
            f"{delta:+.2f}%",
            ha="center",
            va="bottom" if delta >= 0 else "top",
            fontsize=9,
        )
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
