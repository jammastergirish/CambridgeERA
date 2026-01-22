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
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", default="plots/activations")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = df[df["layer"] != "ALL_MEAN"]
    df["layer"] = df["layer"].astype(int)

    for split in ["forget", "retain"]:
        plt.figure(figsize=(9, 6))
        sub = df[df["split"] == split]

        for model in sub["model"].unique():
            m = sub[sub["model"] == model].sort_values("layer")
            plt.plot(
                m["layer"],
                m["mean_norm"],
                marker="o",
                linewidth=1.5,
                label=model.replace("EleutherAI/", "")
            )

        plt.xlabel("Layer")
        plt.ylabel("Mean hidden-state L2 norm")
        if split == "forget":
                subtitle = "WMDP-Bio prompts (forget distribution)"
        else:
            subtitle = "Wikitext samples (retain distribution)"

        plt.suptitle("Layer-wise activation magnitude", fontsize=14)
        plt.title(subtitle, fontsize=10)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, f"activation_norms_{split}.png"))
        plt.close()

    print(f"Activation plots written to {args.outdir}")

if __name__ == "__main__":
    main()