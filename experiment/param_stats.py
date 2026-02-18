#!/usr/bin/env python
# /// script
# dependencies = [
#   "torch",
#   "transformers",
#   "numpy",
#   "tqdm",
#   "safetensors",
#   "huggingface_hub",
#   "wandb",
#   "pandas",
#   "matplotlib",
# ]
# ///

import argparse
import gc
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional, Set

import numpy as np
import torch
from tqdm import tqdm

from utils import (
    comparison_outdir,
    SmartLoader,
    resolve_device,
    resolve_dtype,
    extract_layer,
    classify_granular,
    stable_rank_and_spectral,
    empirical_rank,
    write_csv,
    init_wandb,
    log_csv_as_table,
    log_plots,
    finish_wandb,
)

# ---------------------------------------------------------------------------
# Plotting — generates per-group charts from the per_layer CSV
# ---------------------------------------------------------------------------

def plot_param_stats(per_layer_csv: str, outdir: str, title: str = None):
    """Generate param stats plots from a per-layer CSV file."""
    import pandas as pd
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)

    dataframe = pd.read_csv(per_layer_csv)
    print(f"[param_stats] Generating plots from {per_layer_csv} ({len(dataframe)} rows)")

    for group in ["attn", "mlp"]:
        sub = dataframe[dataframe["group"] == group].sort_values("layer")
        if sub.empty:
            continue

        # ---- Plot A: Relative Frobenius norm (layer locality) ----
        plt.figure(figsize=(8, 5))
        col = "dW_fro_layer_rel" if "dW_fro_layer_rel" in sub.columns else "dW_fro_layer"
        plt.plot(sub["layer"], sub[col], marker="o")
        plt.xlabel("Layer")
        ylabel = rf"$\|\Delta W\|_F / \|W\|_F$ ({group.upper()})" if col.endswith("_rel") else rf"$\|\Delta W\|_F$ per layer ({group.upper()})"
        plt.ylabel(ylabel)
        plt.title(title or f"Layer locality ({group})")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"layer_locality_{group}.png"))
        plt.close()

        # ---- Plot B: Stable rank ----
        plt.figure(figsize=(8, 5))
        plt.plot(sub["layer"], sub["mean_dW_stable_rank"], marker="o")
        plt.xlabel("Layer")
        plt.ylabel(rf"Mean stable rank of $\Delta W$ ({group.upper()})")
        plt.title(title or f"Edit dimensionality - Stable Rank ({group})")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"stable_rank_{group}.png"))
        plt.close()

        # ---- Plot C: Empirical rank ----
        if "mean_dW_empirical_rank" in sub.columns:
            plt.figure(figsize=(8, 5))
            plt.plot(sub["layer"], sub["mean_dW_empirical_rank"], marker="o", color="darkorange")
            plt.xlabel("Layer")
            plt.ylabel(rf"Mean empirical rank of $\Delta W$ ({group.upper()})")
            plt.title(title or f"Edit dimensionality - Empirical Rank ({group})")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"empirical_rank_{group}.png"))
            plt.close()

            # ---- Plot D: Comparison of both ranks ----
            fig, ax1 = plt.subplots(figsize=(10, 6))

            color = 'tab:blue'
            ax1.set_xlabel('Layer')
            ax1.set_ylabel(rf'Mean stable rank of $\Delta W$', color=color)
            ax1.plot(sub["layer"], sub["mean_dW_stable_rank"], marker="o", color=color, label="Stable Rank")
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.grid(alpha=0.3)

            ax2 = ax1.twinx()
            color = 'tab:orange'
            ax2.set_ylabel(rf'Mean empirical rank of $\Delta W$', color=color)
            ax2.plot(sub["layer"], sub["mean_dW_empirical_rank"], marker="s", color=color, label="Empirical Rank")
            ax2.tick_params(axis='y', labelcolor=color)

            plt.title(title or f"Edit dimensionality comparison ({group.upper()})")

            # Add legends from both axes
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"rank_comparison_{group}.png"))
            plt.close()

        # ---- Plot E: Spectral norm (worst-case amplification) ----
        spec_col = "max_dW_spectral_rel" if "max_dW_spectral_rel" in sub.columns else None
        if spec_col:
            plt.figure(figsize=(8, 5))
            plt.plot(sub["layer"], sub[spec_col], marker="o", color="tab:red")
            plt.xlabel("Layer")
            plt.ylabel(rf"$\sigma_1(\Delta W) / \sigma_1(W)$ ({group.upper()})")
            plt.title(title or f"Spectral norm — worst-case amplification ({group})")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"spectral_norm_{group}.png"))
            plt.close()

    print(f"[param_stats] ✓ All plots written to {outdir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", required=True, help="Baseline / before model path")
    parser.add_argument("--model-b", required=True, help="After model path")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--sr-iters", type=int, default=5)
    parser.add_argument("--empirical-rank", action="store_true", default=False,
                         help="Compute empirical rank via full SVD (slow, off by default)")
    parser.add_argument("--empirical-threshold", type=float, default=0.99,
                         help="Threshold for empirical rank (fraction of variance to capture, default: 0.99)")
    parser.add_argument("--outdir", default=None,
                         help="Output dir (default: auto-derived from model names)")
    parser.add_argument("--plot-outdir", default=None,
                         help="Plot dir (default: auto-derived from model names)")
    parser.add_argument("--title", default=None,
                         help="Title for generated plots")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    if args.outdir is None:
        args.outdir = comparison_outdir(args.model_a, args.model_b, suffix="param_stats")
    if args.plot_outdir is None:
        args.plot_outdir = comparison_outdir(args.model_a, args.model_b, suffix="param_plots")

    init_wandb("param_stats", args)

    # Set seed for reproducibility (vital for Power Iteration stability)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    print(f"[param_stats] Initializing SmartLoaders (streaming mode)")
    print(f"  Model A (baseline) : {args.model_a}")
    print(f"  Model B (target)   : {args.model_b}")
    print(f"  Device: {device}  |  Dtype: {dtype}")

    try:
        loader_a = SmartLoader(args.model_a)
        loader_b = SmartLoader(args.model_b)
    except FileNotFoundError as e:
        print(f"Error loading models: {e}")
        return

    # Get intersect of parameters
    names_a = loader_a.get_all_param_names()
    names_b = loader_b.get_all_param_names()
    
    # Filter for Weights only (skip biases, layernorms mostly by name convention for stats)
    # We stick to the logic: Must end in .weight
    all_names = sorted(list(names_a.intersection(names_b)))
    
    linear_names = []
    # Pre-scan to filter potential linear names
    for name in all_names:
        if name.endswith(".weight"):
            linear_names.append(name)

    rows = []
    per_layer = {}

    print(f"[param_stats] Scanning {len(linear_names)} weight matrices for Frobenius norm, stable rank & empirical rank...")
    
    for name in tqdm(linear_names, desc="Comparing weight matrices", unit="matrix"):
        # Load A
        Wa = loader_a.get_param(name, device, dtype)
        if Wa is None or Wa.ndim != 2:
            continue
            
        # Load B
        Wb = loader_b.get_param(name, device, dtype)
        if Wb is None:
            continue
            
        if Wa.shape != Wb.shape:
            print(f"Skipping {name}: shape mismatch {Wa.shape} vs {Wb.shape}")
            continue

        # Calc Stats
        dW = (Wb - Wa)
        
        layer = extract_layer(name)
        group = classify_granular(name)

        # Here's the key part of this file!
        dW_fro = float(dW.float().norm().item())
        W_fro = float(Wa.float().norm().item())
        dW_fro_rel = dW_fro / W_fro if W_fro > 0 else 0.0
        dW_sr, dW_spec = stable_rank_and_spectral(dW, iters=args.sr_iters)
        W_sr, W_spec = stable_rank_and_spectral(Wa, iters=args.sr_iters)
        dW_spec_rel = dW_spec / W_spec if W_spec > 0 else 0.0

        row = {
            "name": name,
            "layer": layer if layer is not None else -1,
            "group": group,
            "shape0": Wa.shape[0],
            "shape1": Wa.shape[1],
            "dW_fro": dW_fro,
            "W_fro": W_fro,
            "dW_fro_rel": dW_fro_rel,
            "dW_spectral": dW_spec,
            "W_spectral": W_spec,
            "dW_spectral_rel": dW_spec_rel,
            "dW_stable_rank": dW_sr,
            "W_stable_rank": W_sr,
        }

        if args.empirical_rank:
            dW_er = empirical_rank(dW, threshold=args.empirical_threshold)
            W_er = empirical_rank(Wa, threshold=args.empirical_threshold)
            row["dW_empirical_rank"] = dW_er
            row["W_empirical_rank"] = W_er

        rows.append(row)

        if layer is not None:
            key = (layer, group)
            defaults = {"sum_dW_fro_sq": 0.0, "sum_W_fro_sq": 0.0, "sum_dW_sr": 0.0,
                         "max_dW_spec": 0.0, "max_W_spec": 0.0, "count": 0}
            if args.empirical_rank:
                defaults["sum_dW_er"] = 0.0
            stats = per_layer.setdefault(key, defaults)
            stats["sum_dW_fro_sq"] += dW_fro * dW_fro
            stats["sum_W_fro_sq"] += W_fro * W_fro
            stats["sum_dW_sr"] += dW_sr
            stats["max_dW_spec"] = max(stats["max_dW_spec"], dW_spec)
            stats["max_W_spec"] = max(stats["max_W_spec"], W_spec)
            if args.empirical_rank:
                stats["sum_dW_er"] += dW_er
            stats["count"] += 1
            
        # Explicit delete to aid GC in loop
        del Wa
        del Wb
        del dW

    # Write Output
    os.makedirs(args.outdir, exist_ok=True)

    per_matrix_fields = ["name", "layer", "group", "shape0", "shape1",
                         "dW_fro", "W_fro", "dW_fro_rel",
                         "dW_spectral", "W_spectral", "dW_spectral_rel",
                         "dW_stable_rank", "W_stable_rank"]
    if args.empirical_rank:
        per_matrix_fields += ["dW_empirical_rank", "W_empirical_rank"]
    write_csv(
        os.path.join(args.outdir, "per_matrix.csv"),
        rows,
        per_matrix_fields,
    )

    per_layer_csv = os.path.join(args.outdir, "per_layer.csv")

    layer_rows = []
    for (layer, group), stats in sorted(per_layer.items(), key=lambda x: (x[0][0], x[0][1])):
        dW_fro_layer = float(np.sqrt(stats["sum_dW_fro_sq"]))
        W_fro_layer = float(np.sqrt(stats["sum_W_fro_sq"]))
        max_dW_spec = stats["max_dW_spec"]
        max_W_spec = stats["max_W_spec"]
        row = {
            "layer": layer,
            "group": group,
            "dW_fro_layer": dW_fro_layer,
            "W_fro_layer": W_fro_layer,
            "dW_fro_layer_rel": dW_fro_layer / W_fro_layer if W_fro_layer > 0 else 0.0,
            "max_dW_spectral": max_dW_spec,
            "max_W_spectral": max_W_spec,
            "max_dW_spectral_rel": max_dW_spec / max_W_spec if max_W_spec > 0 else 0.0,
            "mean_dW_stable_rank": stats["sum_dW_sr"] / max(stats["count"], 1),
            "count_mats": stats["count"],
        }
        if args.empirical_rank:
            row["mean_dW_empirical_rank"] = stats["sum_dW_er"] / max(stats["count"], 1)
        layer_rows.append(row)

    per_layer_fields = ["layer", "group",
                        "dW_fro_layer", "W_fro_layer", "dW_fro_layer_rel",
                        "max_dW_spectral", "max_W_spectral", "max_dW_spectral_rel",
                        "mean_dW_stable_rank", "count_mats"]
    if args.empirical_rank:
        per_layer_fields.insert(-1, "mean_dW_empirical_rank")
    write_csv(per_layer_csv, layer_rows, per_layer_fields)

    print(f"[param_stats] ✓ Wrote {len(rows)} per-matrix rows and {len(layer_rows)} per-layer rows to {args.outdir}")
    log_csv_as_table(os.path.join(args.outdir, "per_matrix.csv"), "per_matrix")
    log_csv_as_table(per_layer_csv, "per_layer")

    # Generate plots if requested
    if args.plot_outdir:
        plot_param_stats(per_layer_csv, args.plot_outdir, args.title)
        log_plots(args.plot_outdir, "param_plots")

    finish_wandb()

if __name__ == "__main__":
    main()
