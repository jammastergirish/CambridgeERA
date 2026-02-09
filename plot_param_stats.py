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
        plt.title(args.title or f"Edit dimensionality ({group})")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, f"stable_rank_{group}.png"))
        plt.close()

    print(f"Plots written to {args.outdir}")

if __name__ == "__main__":
    main()