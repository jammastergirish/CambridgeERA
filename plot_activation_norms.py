#!/usr/bin/env python
# /// script
# dependencies = [
#   "pandas",
#   "matplotlib"
# ]
# ///

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="outputs", help="Root outputs directory")
    ap.add_argument("--outdir", default="plots", help="Root plots directory")
    args = ap.parse_args()

    # Find all activation_stats.csv files
    csv_files = glob.glob(os.path.join(args.indir, "**/activation_stats/activation_stats.csv"), recursive=True)

    if not csv_files:
        print(f"No activation_stats.csv files found in {args.indir}")
        return

    for csv_path in csv_files:
        # Extract comparison name: outputs/<comparison>/activation_stats/activation_stats.csv
        parts = csv_path.split(os.sep)
        activation_stats_idx = parts.index("activation_stats")
        comparison_name = parts[activation_stats_idx - 1]

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

            # Plot 1: Absolute norms comparison (model_a vs model_b)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(sub["layer"], sub["model_a_norm_L2"], marker="o", linewidth=1.5, label="Model A (before)", color="tab:blue")
            ax.plot(sub["layer"], sub["model_b_norm_L2"], marker="s", linewidth=1.5, label="Model B (after)", color="tab:orange")
            ax.set_xlabel("Layer")
            ax.set_ylabel(r"Mean $\|h\|_2$ per token")
            ax.set_title(f"Activation Magnitude ({split})")
            ax.legend()
            ax.grid(alpha=0.3)
            fig.suptitle(comparison_name.replace("__to__", " → ").replace("_", "/"), fontsize=11)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_outdir, f"activation_norms_{split}.png"))
            plt.close()

            # Plot 2: Activation diffs (L1 and L2)
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

            fig.suptitle(comparison_name.replace("__to__", " → ").replace("_", "/"), fontsize=11)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_outdir, f"activation_diffs_{split}.png"))
            plt.close()

        print(f"Wrote: {plot_outdir}")

    print(f"\nActivation plots complete.")


if __name__ == "__main__":
    main()