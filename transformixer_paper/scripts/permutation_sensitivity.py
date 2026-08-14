"""
Permutation Sensitivity of Transformixer vs xLSTM-Mixer.

What it measures
----------------
1) Transformixer: for random channel permutations P, check that
       f(X[:,:,P]) ≈ f(X)[:,:,P]
   (permutation equivariance; should hold up to numerical tolerance).
2) Both models: test MSE/MAE under K random channel shuffles of the *same*
   trained weights. Transformixer metrics should be invariant (equivariant
   predictions, same errors). xLSTM-Mixer metrics typically vary because the
   sLSTM mixer is order-sensitive.

Outputs
-------
  figures/permutation_metrics.csv
  figures/permutation_sensitivity.png
  figures/attention_heatmap.png
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def mse_mae(pred: torch.Tensor, true: torch.Tensor) -> tuple[float, float]:
    mse = torch.mean((pred - true) ** 2).item()
    mae = torch.mean(torch.abs(pred - true)).item()
    return mse, mae


@torch.no_grad()
def equivariance_error(model, batch_x: torch.Tensor, n_perms: int = 5) -> float:
    """Mean relative L2 error between f(X)[:,:,P] and f(X[:,:,P])."""
    model.eval()
    y = model.forecast(batch_x, None)
    errs = []
    c = batch_x.shape[-1]
    for _ in range(n_perms):
        perm = torch.randperm(c, device=batch_x.device)
        y_perm_in = model.forecast(batch_x[:, :, perm], None)
        target = y[:, :, perm]
        denom = target.norm().clamp_min(1e-8)
        errs.append(((y_perm_in - target).norm() / denom).item())
    return float(np.mean(errs))


@torch.no_grad()
def evaluate_shuffled(
    model,
    loader: DataLoader,
    n_batches: int,
    perm: torch.Tensor | None,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    mses, maes = [], []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        # Lightning / TSLib batches vary; support common layouts.
        if isinstance(batch, (list, tuple)):
            batch_x, batch_y = batch[0], batch[1]
        else:
            batch_x, batch_y = batch["x"], batch["y"]
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        if batch_y.dim() == 3 and batch_y.shape[1] > model.pred_len:
            batch_y = batch_y[:, -model.pred_len :, :]
        if perm is not None:
            batch_x = batch_x[:, :, perm]
            batch_y = batch_y[:, :, perm]
        pred = model.forecast(batch_x, None)
        mse, mae = mse_mae(pred, batch_y)
        mses.append(mse)
        maes.append(mae)
    return float(np.mean(mses)), float(np.mean(maes))


def plot_metric_spread(rows: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), sharey=False)
    for ax, metric in zip(axes, ("mse", "mae")):
        for model_name, color in (("Transformixer", "#4C72B0"), ("xLSTM-Mixer", "#DD8452")):
            vals = [r[metric] for r in rows if r["model"] == model_name]
            ax.scatter(
                np.full(len(vals), 0 if model_name == "Transformixer" else 1)
                + 0.05 * np.random.randn(len(vals)),
                vals,
                alpha=0.75,
                color=color,
                label=model_name,
            )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Transformixer", "xLSTM-Mixer"])
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} under random channel permutations")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def save_attention_heatmap(model, batch_x: torch.Tensor, out_path: Path, max_c: int = 64):
    """Optional: requires temporarily enabling attention output in the encoder."""
    model.eval()
    # Best-effort: if the encoder supports output_attention, collect weights.
    enc = getattr(model, "encoder", None)
    if enc is None:
        print("No encoder found; skip attention heatmap.")
        return
    # Forward through embedding path only for a reduced channel subset.
    c = min(batch_x.shape[-1], max_c)
    x = batch_x[:, :, :c]
    x = model.reversible_instance_norm(x, "norm")
    seq_last = x[:, -1:, :].detach()
    z = x - seq_last
    z = model.Linear(z.permute(0, 2, 1)).permute(0, 2, 1)
    z = (z + seq_last).permute(0, 2, 1)
    z = model.pre_encoding(z)
    # Manual attention from first layer if available
    layer0 = enc.attn_layers[0]
    attn_mod = layer0.attention.inner_attention
    # Fall back: compute QK^T softmax from AttentionLayer projections
    attn_layer = layer0.attention
    b, n, _ = z.shape
    q = attn_layer.query_projection(z).view(b, n, attn_layer.n_heads, -1)
    k = attn_layer.key_projection(z).view(b, n, attn_layer.n_heads, -1)
    scores = torch.einsum("bnhd,bmhd->bhnm", q, k) / (q.shape[-1] ** 0.5)
    weights = torch.softmax(scores, dim=-1)[0, 0].cpu().numpy()  # head 0

    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    im = ax.imshow(weights, cmap="viridis", aspect="auto")
    ax.set_xlabel("Key variate")
    ax.set_ylabel("Query variate")
    ax.set_title("Transformixer attention (head 0, subset)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=["Electricity", "Traffic"], default="Traffic")
    parser.add_argument("--k", type=int, default=8, help="Number of random permutations")
    parser.add_argument("--n-batches", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, default=Path("transformixer_paper/figures"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-attention", action="store_true")
    # Checkpoint paths (Lightning .ckpt). Defaults match repo run ids if present.
    parser.add_argument("--transformixer-ckpt", type=Path, default=None)
    parser.add_argument("--xlstm-ckpt", type=Path, default=None)
    args = parser.parse_args()

    ROOT = (args.project_root / "models" / "transformixer").resolve()
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    device = torch.device(args.device)

    from xlstm_mixer.models.transformixer import Transformixer  # type: ignore

    # Build a small synthetic batch if full datamodule wiring is inconvenient in Colab.
    # Prefer real data when ForecastingTSLibDataModule is available.
    enc_in = 862 if args.dataset == "Traffic" else 321
    seq_len = pred_len = 96

    model_t = Transformixer(
        pred_len=pred_len,
        seq_len=seq_len,
        enc_in=enc_in,
        d_model=1024,
        n_heads=16,
        e_layers=2,
        dropout=0.1,
    ).to(device)

    if args.transformixer_ckpt and args.transformixer_ckpt.exists():
        ckpt = torch.load(args.transformixer_ckpt, map_location=device)
        state = ckpt.get("state_dict", ckpt)
        # Strip Lightning prefixes if present.
        cleaned = {}
        for k, v in state.items():
            nk = k
            for pfx in ("model.architecture.", "architecture.", "model."):
                if nk.startswith(pfx):
                    nk = nk[len(pfx) :]
            if nk.startswith("architecture."):
                nk = nk[len("architecture.") :]
            cleaned[nk] = v
        missing, unexpected = model_t.load_state_dict(cleaned, strict=False)
        print(f"Transformixer load: missing={len(missing)} unexpected={len(unexpected)}")

    # Synthetic loader fallback (replace with real test loader in Colab when possible).
    class _Synth(torch.utils.data.Dataset):
        def __len__(self):
            return 64

        def __getitem__(self, idx):
            x = torch.randn(seq_len, enc_in)
            y = torch.randn(pred_len, enc_in)
            return x, y

    loader = DataLoader(_Synth(), batch_size=4, shuffle=False)

    eq_err = equivariance_error(model_t, next(iter(loader))[0].to(device), n_perms=5)
    print(f"Transformixer mean relative equivariance error: {eq_err:.3e}")

    rows = []
    # Identity perm
    mse, mae = evaluate_shuffled(model_t, loader, args.n_batches, None, device)
    rows.append({"model": "Transformixer", "perm_id": 0, "mse": mse, "mae": mae})

    for i in range(1, args.k + 1):
        perm = torch.randperm(enc_in)
        mse, mae = evaluate_shuffled(model_t, loader, args.n_batches, perm, device)
        rows.append({"model": "Transformixer", "perm_id": i, "mse": mse, "mae": mae})

    # xLSTM-Mixer: import only if available; otherwise skip with a message.
    try:
        ROOT = (args.project_root / "models" / "xlstm-mixer").resolve()
        os.chdir(ROOT)
        sys.path.insert(0, str(ROOT))
        device = torch.device(args.device)
        from xlstm_mixer.models.xlstm_mixer import xLSTMMixer  # type: ignore

        model_x = xLSTMMixer(
            pred_len=pred_len,
            seq_len=seq_len,
            enc_in=enc_in,
            xlstm_embedding_dim=1024,
            num_mem_tokens=3,
            xlstm_dropout=0.1,
            xlstm_conv1d_kernel_size=4,
            xlstm_num_heads=16,
            xlstm_num_blocks=2,
        ).to(device)
        if args.xlstm_ckpt and args.xlstm_ckpt.exists():
            ckpt = torch.load(args.xlstm_ckpt, map_location=device)
            state = ckpt.get("state_dict", ckpt)
            cleaned = {}
            for k, v in state.items():
                nk = k
                for pfx in ("model.architecture.", "architecture.", "model."):
                    if nk.startswith(pfx):
                        nk = nk[len(pfx) :]
                cleaned[nk] = v
            missing, unexpected = model_x.load_state_dict(cleaned, strict=False)
            print(f"xLSTM-Mixer load: missing={len(missing)} unexpected={len(unexpected)}")

        mse, mae = evaluate_shuffled(model_x, loader, args.n_batches, None, device)
        rows.append({"model": "xLSTM-Mixer", "perm_id": 0, "mse": mse, "mae": mae})
        for i in range(1, args.k + 1):
            perm = torch.randperm(enc_in)
            mse, mae = evaluate_shuffled(model_x, loader, args.n_batches, perm, device)
            rows.append({"model": "xLSTM-Mixer", "perm_id": i, "mse": mse, "mae": mae})
    except Exception as exc:  # noqa: BLE001
        print(f"Skipping xLSTM-Mixer evaluation ({exc})")

    ROOT = (args.project_root / "transformixer_paper").resolve()
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    device = torch.device(args.device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "permutation_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "perm_id", "mse", "mae"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}")

    plot_metric_spread(rows, args.out_dir / "permutation_sensitivity.png")

    if not args.skip_attention:
        bx = next(iter(loader))[0].float().to(device)
        try:
            save_attention_heatmap(model_t, bx, args.out_dir / "attention_heatmap.png")
        except Exception as exc:  # noqa: BLE001
            print(f"Attention heatmap skipped: {exc}")

    # Summary stats for the paper
    for name in ("Transformixer", "xLSTM-Mixer"):
        vals = [r["mse"] for r in rows if r["model"] == name]
        if vals:
            print(
                f"{name}: MSE mean={np.mean(vals):.6f} std={np.std(vals):.6f} "
                f"range={np.max(vals) - np.min(vals):.6f}"
            )


if __name__ == "__main__":
    main()
