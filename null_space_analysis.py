#!/usr/bin/env python
# /// script
# dependencies = [
#   "torch",
#   "transformers",
#   "numpy",
#   "matplotlib",
#   "scipy",
# ]
# ///

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import null_space
from tqdm import tqdm

from utils import resolve_device, resolve_dtype, write_csv
from collect_param_stats import SmartLoader


def compute_null_space_projection(dW: torch.Tensor, rank_threshold: float = 0.99) -> dict:
    """
    Analyze how much of the weight change lies in the null space of the original weights.
    This helps understand if changes are "orthogonal" to the original function.
    """
    if dW.numel() == 0 or dW.ndim != 2:
        return {"null_proj_ratio": 0.0, "effective_rank": 0}

    # Convert to numpy for null space computation
    dW_np = dW.cpu().float().numpy()

    # Compute SVD of difference matrix
    U, s, Vt = np.linalg.svd(dW_np, full_matrices=False)

    # Find effective rank (how many singular values capture threshold of variance)
    s_squared = s * s
    total_var = np.sum(s_squared)
    if total_var == 0:
        return {"null_proj_ratio": 0.0, "effective_rank": 0}

    cumsum = np.cumsum(s_squared)
    effective_rank = np.searchsorted(cumsum, rank_threshold * total_var) + 1

    # Compute how concentrated the changes are in top singular vectors
    top_k_variance = cumsum[min(10, len(s)-1)] / total_var if len(s) > 0 else 0

    return {
        "effective_rank": int(effective_rank),
        "top10_variance_ratio": float(top_k_variance),
        "max_singular_value": float(s[0]) if len(s) > 0 else 0,
        "singular_value_decay": float(s[min(10, len(s)-1)] / (s[0] + 1e-10)) if len(s) > 10 else 1.0,
    }


def analyze_subspace_alignment(Wa: torch.Tensor, Wb: torch.Tensor, k: int = 20) -> dict:
    """
    Analyze alignment between subspaces of original and fine-tuned weights.
    """
    if Wa.numel() == 0 or Wa.ndim != 2:
        return {}

    # Get top-k singular vectors for both matrices
    Wa_np = Wa.cpu().float().numpy()
    Wb_np = Wb.cpu().float().numpy()

    # SVD to get principal subspaces
    Ua, sa, _ = np.linalg.svd(Wa_np, full_matrices=False)
    Ub, sb, _ = np.linalg.svd(Wb_np, full_matrices=False)

    k = min(k, Ua.shape[1], Ub.shape[1])

    # Compute subspace alignment (Grassmann distance)
    # This measures how aligned the top-k subspaces are
    Ua_k = Ua[:, :k]
    Ub_k = Ub[:, :k]

    # Compute alignment matrix
    M = Ua_k.T @ Ub_k
    alignment_singular_values = np.linalg.svd(M, compute_uv=False)

    # Average alignment (higher = more aligned)
    avg_alignment = np.mean(alignment_singular_values)

    # Grassmann distance
    grassmann_dist = np.sqrt(max(0, k - np.sum(alignment_singular_values**2)))

    return {
        "subspace_alignment": float(avg_alignment),
        "grassmann_distance": float(grassmann_dist),
        "singular_value_ratio": float(sb[0] / (sa[0] + 1e-10)) if len(sa) > 0 and len(sb) > 0 else 1.0,
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

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    print(f"Loading models...")
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
    mlp_results = {"null_space": [], "alignment": []}
    attn_results = {"null_space": [], "alignment": []}

    print(f"Analyzing {len(weight_names)} weight matrices...")
    for name in tqdm(weight_names, desc="Processing weights"):
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

        # Classify component
        is_mlp = 'mlp' in name.lower() or 'ffn' in name.lower()
        is_attn = 'attn' in name.lower() or 'attention' in name.lower()

        result = {
            "name": name,
            "component": "mlp" if is_mlp else ("attn" if is_attn else "other"),
            **null_analysis,
            **alignment,
        }
        results.append(result)

        # Aggregate by component type
        if is_mlp:
            mlp_results["null_space"].append(null_analysis.get("top10_variance_ratio", 0))
            mlp_results["alignment"].append(alignment.get("subspace_alignment", 0))
        elif is_attn:
            attn_results["null_space"].append(null_analysis.get("top10_variance_ratio", 0))
            attn_results["alignment"].append(alignment.get("subspace_alignment", 0))

        del Wa, Wb, dW

    # Save results
    os.makedirs(args.outdir, exist_ok=True)

    write_csv(
        os.path.join(args.outdir, "null_space_results.csv"),
        results,
        list(results[0].keys()) if results else [],
    )

    # Create visualizations
    if mlp_results["null_space"] or attn_results["null_space"]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Null space concentration
        ax = axes[0]
        data_to_plot = []
        labels = []
        if mlp_results["null_space"]:
            data_to_plot.append(mlp_results["null_space"])
            labels.append("MLP")
        if attn_results["null_space"]:
            data_to_plot.append(attn_results["null_space"])
            labels.append("Attention")

        bp = ax.boxplot(data_to_plot, labels=labels)
        ax.set_ylabel("Top-10 SV Variance Ratio")
        ax.set_title("Weight Change Concentration (Higher = More Low-Rank)")
        ax.grid(alpha=0.3)

        # Subspace alignment
        ax = axes[1]
        data_to_plot = []
        labels = []
        if mlp_results["alignment"]:
            data_to_plot.append(mlp_results["alignment"])
            labels.append("MLP")
        if attn_results["alignment"]:
            data_to_plot.append(attn_results["alignment"])
            labels.append("Attention")

        bp = ax.boxplot(data_to_plot, labels=labels)
        ax.set_ylabel("Subspace Alignment")
        ax.set_title("Original vs Fine-tuned Subspace Alignment")
        ax.grid(alpha=0.3)

        plt.suptitle("Null Space and Subspace Analysis")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "null_space_visualization.png"))
        plt.close()

    # Print summary statistics
    print("\n=== Null Space Analysis Summary ===")
    if mlp_results["null_space"]:
        print(f"MLP - Avg variance in top-10 SVs: {np.mean(mlp_results['null_space']):.3f}")
        print(f"MLP - Avg subspace alignment: {np.mean(mlp_results['alignment']):.3f}")
    if attn_results["null_space"]:
        print(f"Attention - Avg variance in top-10 SVs: {np.mean(attn_results['null_space']):.3f}")
        print(f"Attention - Avg subspace alignment: {np.mean(attn_results['alignment']):.3f}")

    print(f"\nResults saved to {args.outdir}")


if __name__ == "__main__":
    main()