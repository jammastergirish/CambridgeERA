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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-layer-csv", required=True)
    ap.add_argument("--outdir", default="plots")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.per_layer_csv)

    for group in ["attn", "mlp"]:
        sub = df[df["group"] == group].sort_values("layer")
        if sub.empty:
            continue

        # ---- Plot A: Frobenius norm (layer locality) ----
        plt.figure(figsize=(8, 5))
        plt.plot(sub["layer"], sub["dW_fro_layer"], marker="o")
        plt.xlabel("Layer")
        plt.ylabel(rf"$\|\Delta W\|_F$ per layer ({group.upper()})")
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

    print(f"Plots written to {args.outdir}")

if __name__ == "__main__":
    main()