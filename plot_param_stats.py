#!/usr/bin/env python
# /// script
# dependencies = [
#   "torch",
#   "pandas",
#   "matplotlib",
#   "wandb",
# ]
# ///

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

from utils import init_wandb, log_plots, finish_wandb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-layer-csv", required=True)
    ap.add_argument("--outdir", default="plots")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()
    init_wandb("plot_param_stats", args)

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.per_layer_csv)
    print(f"[plot_param_stats] Loaded {len(df)} rows from {args.per_layer_csv}")

    for group in ["attn", "mlp"]:
        sub = df[df["group"] == group].sort_values("layer")
        if sub.empty:
            continue

        print(f"[plot_param_stats] Generating plots for {group.upper()} group ({len(sub)} layers)...")

        # ---- Plot A: Relative Frobenius norm (layer locality) ----
        plt.figure(figsize=(8, 5))
        col = "dW_fro_layer_rel" if "dW_fro_layer_rel" in sub.columns else "dW_fro_layer"
        plt.plot(sub["layer"], sub[col], marker="o")
        plt.xlabel("Layer")
        ylabel = rf"$\|\Delta W\|_F / \|W\|_F$ ({group.upper()})" if col.endswith("_rel") else rf"$\|\Delta W\|_F$ per layer ({group.upper()})"
        plt.ylabel(ylabel)
        plt.title(args.title or f"Layer locality ({group})")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, f"layer_locality_{group}.png"))
        plt.close()

        # ---- Plot B: Stable rank ----
        plt.figure(figsize=(8, 5))
        plt.plot(sub["layer"], sub["mean_dW_stable_rank"], marker="o")
        plt.xlabel("Layer")
        plt.ylabel(rf"Mean stable rank of $\Delta W$ ({group.upper()})")
        plt.title(args.title or f"Edit dimensionality - Stable Rank ({group})")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, f"stable_rank_{group}.png"))
        plt.close()

        # ---- Plot C: Empirical rank ----
        if "mean_dW_empirical_rank" in sub.columns:
            plt.figure(figsize=(8, 5))
            plt.plot(sub["layer"], sub["mean_dW_empirical_rank"], marker="o", color="darkorange")
            plt.xlabel("Layer")
            plt.ylabel(rf"Mean empirical rank of $\Delta W$ ({group.upper()})")
            plt.title(args.title or f"Edit dimensionality - Empirical Rank ({group})")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(args.outdir, f"empirical_rank_{group}.png"))
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

            plt.title(args.title or f"Edit dimensionality comparison ({group.upper()})")

            # Add legends from both axes
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

            plt.tight_layout()
            plt.savefig(os.path.join(args.outdir, f"rank_comparison_{group}.png"))
            plt.close()

        # ---- Plot E: Spectral norm (worst-case amplification) ----
        spec_col = "max_dW_spectral_rel" if "max_dW_spectral_rel" in sub.columns else None
        if spec_col:
            plt.figure(figsize=(8, 5))
            plt.plot(sub["layer"], sub[spec_col], marker="o", color="tab:red")
            plt.xlabel("Layer")
            plt.ylabel(rf"$\sigma_1(\Delta W) / \sigma_1(W)$ ({group.upper()})")
            plt.title(args.title or f"Spectral norm — worst-case amplification ({group})")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(args.outdir, f"spectral_norm_{group}.png"))
            plt.close()

    print(f"[plot_param_stats] ✓ All plots written to {args.outdir}")
    log_plots(args.outdir, "param_plots")
    finish_wandb()

if __name__ == "__main__":
    main()