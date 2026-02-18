#!/usr/bin/env python
# /// script
# dependencies = [
#   "torch",
#   "transformers",
#   "numpy",
#   "matplotlib",
#   "scipy",
#   "wandb",
#   "pandas",
# ]
# ///

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import resolve_device, resolve_dtype, write_csv, classify_granular, init_wandb, log_csv_as_table, log_plots, finish_wandb
from param_stats import SmartLoader


def compute_null_space_projection(dW: torch.Tensor, rank_threshold: float = 0.99) -> dict:
    """
    Analyze how much of the weight change lies in the null space of the original weights.
    This helps understand if changes are "orthogonal" to the original function.
    """
    if dW.numel() == 0 or dW.ndim != 2:
        return {
            "null_proj_ratio": 0.0,
            "effective_rank": 0,
            "top10_variance_ratio": 0.0,
            "max_singular_value": 0.0,
            "singular_value_decay": 1.0,
        }

    # Compute singular values on device (GPU if available)
    s = torch.linalg.svdvals(dW.float())

    # Find effective rank (how many singular values capture threshold of variance)
    s_squared = s * s
    total_var = s_squared.sum().item()
    if total_var == 0:
        return {
            "null_proj_ratio": 0.0,
            "effective_rank": 0,
            "top10_variance_ratio": 0.0,
            "max_singular_value": 0.0,
            "singular_value_decay": 1.0,
        }

    cumsum = torch.cumsum(s_squared, dim=0)
    effective_rank = int(torch.searchsorted(cumsum, rank_threshold * total_var).item()) + 1

    # Compute how concentrated the changes are in top singular vectors
    top_k_variance = float(cumsum[min(9, len(s)-1)].item() / total_var) if len(s) > 0 else 0

    return {
        "effective_rank": effective_rank,
        "top10_variance_ratio": top_k_variance,
        "max_singular_value": float(s[0].item()) if len(s) > 0 else 0,
        "singular_value_decay": float((s[min(10, len(s)-1)] / (s[0] + 1e-10)).item()) if len(s) > 10 else 1.0,
    }


def analyze_subspace_alignment(Wa: torch.Tensor, Wb: torch.Tensor, k: int = 20) -> dict:
    """
    Analyze alignment between subspaces of original and fine-tuned weights.
    """
    if Wa.numel() == 0 or Wa.ndim != 2:
        return {}

    # SVD on device (GPU if available) — only need left singular vectors
    Ua, sa, _ = torch.linalg.svd(Wa.float(), full_matrices=False)
    Ub, sb, _ = torch.linalg.svd(Wb.float(), full_matrices=False)

    k = min(k, Ua.shape[1], Ub.shape[1])

    # Compute subspace alignment (Grassmann distance)
    Ua_k = Ua[:, :k]
    Ub_k = Ub[:, :k]

    # Compute alignment matrix
    M = Ua_k.T @ Ub_k
    alignment_singular_values = torch.linalg.svdvals(M)

    # Average alignment (higher = more aligned)
    avg_alignment = float(alignment_singular_values.mean().item())

    # Grassmann distance
    grassmann_dist = float(torch.sqrt(torch.clamp(k - (alignment_singular_values**2).sum(), min=0)).item())

    return {
        "subspace_alignment": avg_alignment,
        "grassmann_distance": grassmann_dist,
        "singular_value_ratio": float((sb[0] / (sa[0] + 1e-10)).item()) if len(sa) > 0 and len(sb) > 0 else 1.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-a", required=True, help="Baseline model path")
    ap.add_argument("--model-b", required=True, help="Fine-tuned model path")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--outdir", default="outputs/null_space_analysis")
    ap.add_argument("--num-samples", type=int, default=50, help="Number of weight matrices to sample")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    init_wandb("null_space_analysis", args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    print(f"[null_space_analysis] Loading model weights...")
    print(f"  Model A (baseline): {args.model_a}")
    print(f"  Model B (target)  : {args.model_b}")
    loader_a = SmartLoader(args.model_a)
    loader_b = SmartLoader(args.model_b)

    # Get weight names
    names_a = loader_a.get_all_param_names()
    names_b = loader_b.get_all_param_names()
    weight_names = sorted([n for n in names_a.intersection(names_b) if n.endswith('.weight')])

    # Sample weights for analysis
    if len(weight_names) > args.num_samples:
        weight_names = np.random.choice(weight_names, args.num_samples, replace=False).tolist()

    results = []
    _COMP_LABELS = ('qkv', 'proj', 'mlp_expand', 'mlp_contract')
    comp_results = {c: {"null_space": [], "alignment": []} for c in _COMP_LABELS}

    print(f"[null_space_analysis] Running SVD on {len(weight_names)} weight matrices (sampled from {args.num_samples} requested)...")
    for name in tqdm(weight_names, desc="SVD on weight matrices", unit="matrix"):
        Wa = loader_a.get_param(name, device, dtype)
        if Wa is None or Wa.ndim != 2:
            continue

        Wb = loader_b.get_param(name, device, dtype)
        if Wb is None or Wb.shape != Wa.shape:
            continue

        dW = Wb - Wa

        # Null space analysis
        null_analysis = compute_null_space_projection(dW)

        # Subspace alignment
        alignment = analyze_subspace_alignment(Wa, Wb)

        # Classify component (granular)
        component_type = classify_granular(name)

        result = {
            "name": name,
            "component": component_type,
            **null_analysis,
            **alignment,
        }
        results.append(result)

        # Aggregate by component type
        if component_type in comp_results:
            comp_results[component_type]["null_space"].append(null_analysis.get("top10_variance_ratio", 0))
            comp_results[component_type]["alignment"].append(alignment.get("subspace_alignment", 0))

        del Wa, Wb, dW

    # Save results
    os.makedirs(args.outdir, exist_ok=True)

    if results:
        fieldnames = list(results[0].keys())
        write_csv(
            os.path.join(args.outdir, "null_space_results.csv"),
            results,
            fieldnames,
        )

    # Create visualizations
    has_data = any(comp_results[c]["null_space"] for c in _COMP_LABELS)
    if has_data:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Null space concentration
        ax = axes[0]
        data_to_plot = []
        labels = []
        for c in _COMP_LABELS:
            if comp_results[c]["null_space"]:
                data_to_plot.append(comp_results[c]["null_space"])
                labels.append(c)

        bp = ax.boxplot(data_to_plot, labels=labels)
        ax.set_ylabel("Top-10 SV Variance Ratio")
        ax.set_title("Weight Change Concentration (Higher = More Low-Rank)")
        ax.grid(alpha=0.3)

        # Subspace alignment
        ax = axes[1]
        data_to_plot = []
        labels = []
        for c in _COMP_LABELS:
            if comp_results[c]["alignment"]:
                data_to_plot.append(comp_results[c]["alignment"])
                labels.append(c)

        bp = ax.boxplot(data_to_plot, labels=labels)
        ax.set_ylabel("Subspace Alignment")
        ax.set_title("Original vs Fine-tuned Subspace Alignment")
        ax.grid(alpha=0.3)

        plt.suptitle("Null Space and Subspace Analysis")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "null_space_visualization.png"))
        plt.close()

    # Print summary statistics
    print("\n[null_space_analysis] === Null Space Analysis Summary ===")
    for c in _COMP_LABELS:
        if comp_results[c]["null_space"]:
            print(f"{c} - Avg variance in top-10 SVs: {np.mean(comp_results[c]['null_space']):.3f}")
            print(f"{c} - Avg subspace alignment: {np.mean(comp_results[c]['alignment']):.3f}")

    print(f"\n[null_space_analysis] ✓ Results saved to {args.outdir}")
    log_csv_as_table(os.path.join(args.outdir, "null_space_results.csv"), "null_space_results")
    log_plots(args.outdir, "null_space")
    finish_wandb()


if __name__ == "__main__":
    main()