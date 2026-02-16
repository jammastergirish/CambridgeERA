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
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob

from utils import init_wandb, log_plots, finish_wandb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="outputs", help="Root outputs directory")
    ap.add_argument("--outdir", default="plots", help="Root plots directory")
    args = ap.parse_args()
    init_wandb("plot_activation_norms", args)

    # Find all activation_stats.csv files
    csv_files = glob.glob(os.path.join(args.indir, "**/activation_stats/activation_stats.csv"), recursive=True)

    if not csv_files:
        print(f"[plot_activation_norms] No activation_stats.csv files found in {args.indir}")
        return

    print(f"[plot_activation_norms] Found {len(csv_files)} comparison(s) to plot")

    for csv_path in csv_files:
        # Extract comparison name: outputs/<comparison>/activation_stats/activation_stats.csv
        parts = csv_path.split(os.sep)
        activation_stats_idx = parts.index("activation_stats")
        comparison_name = parts[activation_stats_idx - 1]
        title = comparison_name.replace("__to__", " → ").replace("_", "/")

        df = pd.read_csv(csv_path)
        df = df[df["layer"] != "ALL_MEAN"]
        df["layer"] = df["layer"].astype(int)

        # Output to plots/<comparison>/activation_plots/
        plot_outdir = os.path.join(args.outdir, comparison_name, "activation_plots")
        os.makedirs(plot_outdir, exist_ok=True)

        for split in ["forget", "retain"]:
            sub = df[df["split"] == split].sort_values("layer")
            if sub.empty:
                continue

            # Plot 1: Absolute norms — L1 and L2 side-by-side, model A vs model B
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            ax1.plot(sub["layer"], sub["model_a_norm_L1"], marker="o", linewidth=1.5, label="Model A (before)", color="tab:blue")
            ax1.plot(sub["layer"], sub["model_b_norm_L1"], marker="s", linewidth=1.5, label="Model B (after)", color="tab:orange")
            ax1.set_xlabel("Layer")
            ax1.set_ylabel(r"Mean $\|h\|_1$ per token")
            ax1.set_title(f"$L_1$ Activation Magnitude ({split})")
            ax1.legend()
            ax1.grid(alpha=0.3)

            ax2.plot(sub["layer"], sub["model_a_norm_L2"], marker="o", linewidth=1.5, label="Model A (before)", color="tab:blue")
            ax2.plot(sub["layer"], sub["model_b_norm_L2"], marker="s", linewidth=1.5, label="Model B (after)", color="tab:orange")
            ax2.set_xlabel("Layer")
            ax2.set_ylabel(r"Mean $\|h\|_2$ per token")
            ax2.set_title(f"$L_2$ Activation Magnitude ({split})")
            ax2.legend()
            ax2.grid(alpha=0.3)

            fig.suptitle(title, fontsize=11)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_outdir, f"activation_norms_{split}.png"))
            plt.close()

            # Plot 2: Activation diffs — L1 and L2 side-by-side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            ax1.plot(sub["layer"], sub["mean_dh_L1"], marker="o", linewidth=1.5, color="tab:green")
            ax1.set_xlabel("Layer")
            ax1.set_ylabel(r"Mean $\|\Delta h\|_1$ per token")
            ax1.set_title(f"Activation Diff $L_1$ Norm ({split})")
            ax1.grid(alpha=0.3)

            ax2.plot(sub["layer"], sub["mean_dh_L2"], marker="o", linewidth=1.5, color="tab:red")
            ax2.set_xlabel("Layer")
            ax2.set_ylabel(r"Mean $\|\Delta h\|_2$ per token")
            ax2.set_title(f"Activation Diff $L_2$ Norm ({split})")
            ax2.grid(alpha=0.3)

            fig.suptitle(title, fontsize=11)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_outdir, f"activation_diffs_{split}.png"))
            plt.close()

        print(f"[plot_activation_norms] ✓ Wrote activation plots to {plot_outdir}")

    print(f"\n[plot_activation_norms] Done — all activation plots complete.")
    log_plots(args.outdir, "activation_plots")
    finish_wandb()


if __name__ == "__main__":
    main()