#!/usr/bin/env python
# /// script
# dependencies = [
#   "torch",
#   "numpy",
#   "matplotlib",
#   "tqdm",
#   "safetensors",
#   "huggingface_hub",
#   "wandb",
#   "pandas",
# ]
# ///

"""
Analyze MLP parameter update alignment with original weight nullspace.
Tests if updates primarily affect new directions orthogonal to original column space.
"""

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

from utils import resolve_device, resolve_dtype, write_csv, extract_layer, classify_coarse, init_wandb, log_csv_as_table, log_plots, finish_wandb
from collect_param_stats import SmartLoader


def compute_nullspace_alignment(W_orig, dW, rank_threshold=0.99):
    """
    Analyze how the update dW aligns with the nullspace of original weights W_orig.

    Returns:
        dict: Alignment metrics
    """
    if W_orig.ndim != 2 or dW.ndim != 2:
        return None

    # Keep on device (GPU if available), cast to float32 for numerical stability
    W = W_orig.float()
    dW_f = dW.float()

    # Compute SVD of original weights (need full U for nullspace projection)
    U, s, Vt = torch.linalg.svd(W, full_matrices=True)

    # Find effective rank (for nullspace boundary)
    s_squared = s * s
    total_var = s_squared.sum().item()
    if total_var == 0:
        return None

    cumsum = torch.cumsum(s_squared, dim=0)
    effective_rank = int(torch.searchsorted(cumsum, rank_threshold * total_var).item()) + 1
    effective_rank = min(effective_rank, len(s))

    # Split into column space and approximate nullspace
    n_rows, n_cols = W.shape

    if effective_rank >= min(n_rows, n_cols):
        # Full rank - no nullspace
        nullspace_dim = 0
        colspace_proj_norm = float(dW_f.norm().item())
        nullspace_proj_norm = 0.0
    else:
        # Project dW onto column space and nullspace
        U_colspace = U[:, :effective_rank]
        U_nullspace = U[:, effective_rank:]

        # Project update onto column space
        dW_colspace = U_colspace @ (U_colspace.T @ dW_f)
        colspace_proj_norm = float(dW_colspace.norm().item())

        # Project update onto nullspace
        dW_nullspace = U_nullspace @ (U_nullspace.T @ dW_f)
        nullspace_proj_norm = float(dW_nullspace.norm().item())

        nullspace_dim = U_nullspace.shape[1]

    # Total update norm
    total_norm = float(dW_f.norm().item())

    # Compute alignment ratios
    if total_norm > 0:
        colspace_ratio = colspace_proj_norm / total_norm
        nullspace_ratio = nullspace_proj_norm / total_norm
    else:
        colspace_ratio = 0.0
        nullspace_ratio = 0.0

    # Also analyze row space (for decoder/output matrices)
    sr = torch.linalg.svdvals(W.T)
    sr_squared = sr * sr
    total_var_r = sr_squared.sum().item()
    if total_var_r > 0:
        cumsum_r = torch.cumsum(sr_squared, dim=0)
        effective_rank_r = int(torch.searchsorted(cumsum_r, rank_threshold * total_var_r).item()) + 1
        effective_rank_r = min(effective_rank_r, len(sr))
    else:
        effective_rank_r = 0

    # Compute rank change
    W_new = W + dW_f
    s_new = torch.linalg.svdvals(W_new)
    s_new_squared = s_new * s_new
    total_var_new = s_new_squared.sum().item()
    if total_var_new > 0:
        cumsum_new = torch.cumsum(s_new_squared, dim=0)
        effective_rank_new = int(torch.searchsorted(cumsum_new, rank_threshold * total_var_new).item()) + 1
    else:
        effective_rank_new = 0

    return {
        "original_eff_rank": int(effective_rank),
        "updated_eff_rank": int(effective_rank_new),
        "rank_increase": int(effective_rank_new - effective_rank),
        "colspace_projection_ratio": float(colspace_ratio),
        "nullspace_projection_ratio": float(nullspace_ratio),
        "nullspace_dimension": int(nullspace_dim),
        "update_norm": float(total_norm),
        "original_norm": float(W.norm().item()),
        "relative_update_size": float(total_norm / (W.norm().item() + 1e-10)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-a", required=True, help="Baseline model")
    ap.add_argument("--model-b", required=True, help="Unlearned/finetuned model")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--rank-threshold", type=float, default=0.99)
    ap.add_argument("--outdir", default="outputs/mlp_nullspace_alignment")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    init_wandb("mlp_nullspace_alignment", args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    print(f"[mlp_nullspace_alignment] Loading model weights...")
    print(f"  Model A (baseline): {args.model_a}")
    print(f"  Model B (target)  : {args.model_b}")

    loader_a = SmartLoader(args.model_a)
    loader_b = SmartLoader(args.model_b)

    # Get weight names - focus on MLP weights
    names_a = loader_a.get_all_param_names()
    names_b = loader_b.get_all_param_names()

    # Filter for MLP weights only
    mlp_names = []
    for name in sorted(names_a.intersection(names_b)):
        if name.endswith('.weight'):
            if classify_coarse(name) == 'mlp' or 'fc' in name.lower():
                mlp_names.append(name)

    print(f"[mlp_nullspace_alignment] Found {len(mlp_names)} MLP weight matrices")

    results = []
    layer_aggregated = {}

    print("[mlp_nullspace_alignment] Computing SVD + nullspace projections...")
    for name in tqdm(mlp_names, desc="Analyzing MLP nullspace alignment", unit="matrix"):
        # Load weights
        Wa = loader_a.get_param(name, device, dtype)
        if Wa is None or Wa.ndim != 2:
            continue

        Wb = loader_b.get_param(name, device, dtype)
        if Wb is None or Wb.shape != Wa.shape:
            continue

        # Compute update
        dW = Wb - Wa

        # Analyze nullspace alignment
        metrics = compute_nullspace_alignment(Wa, dW, args.rank_threshold)

        if metrics is None:
            continue

        # Add metadata
        layer = extract_layer(name)
        metrics["name"] = name
        metrics["layer"] = layer if layer is not None else -1
        metrics["shape"] = f"{Wa.shape[0]}x{Wa.shape[1]}"

        # Determine if encoder (input projection) or decoder (output projection)
        if 'up_proj' in name or 'out_proj' in name or 'o_proj' in name or 'fc2' in name:
            metrics["type"] = "decoder"
        else:
            metrics["type"] = "encoder"

        results.append(metrics)

        # Aggregate by layer
        if layer is not None:
            if layer not in layer_aggregated:
                layer_aggregated[layer] = {
                    "colspace_ratios": [],
                    "nullspace_ratios": [],
                    "rank_increases": [],
                    "encoder_nullspace": [],
                    "decoder_nullspace": [],
                }

            layer_aggregated[layer]["colspace_ratios"].append(metrics["colspace_projection_ratio"])
            layer_aggregated[layer]["nullspace_ratios"].append(metrics["nullspace_projection_ratio"])
            layer_aggregated[layer]["rank_increases"].append(metrics["rank_increase"])

            if metrics["type"] == "encoder":
                layer_aggregated[layer]["encoder_nullspace"].append(metrics["nullspace_projection_ratio"])
            else:
                layer_aggregated[layer]["decoder_nullspace"].append(metrics["nullspace_projection_ratio"])

        del Wa, Wb, dW

    # Compute layer-wise statistics
    layer_results = []
    for layer in sorted(layer_aggregated.keys()):
        stats = layer_aggregated[layer]
        layer_results.append({
            "layer": layer,
            "avg_colspace_ratio": np.mean(stats["colspace_ratios"]),
            "avg_nullspace_ratio": np.mean(stats["nullspace_ratios"]),
            "avg_rank_increase": np.mean(stats["rank_increases"]),
            "encoder_nullspace_ratio": np.mean(stats["encoder_nullspace"]) if stats["encoder_nullspace"] else 0,
            "decoder_nullspace_ratio": np.mean(stats["decoder_nullspace"]) if stats["decoder_nullspace"] else 0,
            "num_matrices": len(stats["colspace_ratios"]),
        })

    # Save results
    os.makedirs(args.outdir, exist_ok=True)

    write_csv(
        os.path.join(args.outdir, "mlp_nullspace_metrics.csv"),
        results,
        ["name", "layer", "type", "shape", "original_eff_rank", "updated_eff_rank",
         "rank_increase", "colspace_projection_ratio", "nullspace_projection_ratio",
         "nullspace_dimension", "update_norm", "original_norm", "relative_update_size"]
    )

    write_csv(
        os.path.join(args.outdir, "layer_nullspace_summary.csv"),
        layer_results,
        ["layer", "avg_colspace_ratio", "avg_nullspace_ratio", "avg_rank_increase",
         "encoder_nullspace_ratio", "decoder_nullspace_ratio", "num_matrices"]
    )

    # Create visualizations
    if layer_results:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        layers = [r["layer"] for r in layer_results]

        # Plot 1: Nullspace vs Colspace projection
        ax = axes[0, 0]
        ax.plot(layers, [r["avg_nullspace_ratio"] for r in layer_results], 'o-',
                label="Nullspace", color='red')
        ax.plot(layers, [r["avg_colspace_ratio"] for r in layer_results], 's-',
                label="Column Space", color='blue')
        ax.set_xlabel("Layer")
        ax.set_ylabel("Projection Ratio")
        ax.set_title("Update Alignment (Higher Nullspace = Off-manifold)")
        ax.legend()
        ax.grid(alpha=0.3)

        # Plot 2: Encoder vs Decoder nullspace alignment
        ax = axes[0, 1]
        ax.plot(layers, [r["encoder_nullspace_ratio"] for r in layer_results], 'o-',
                label="Encoder (Input)", color='green')
        ax.plot(layers, [r["decoder_nullspace_ratio"] for r in layer_results], 's-',
                label="Decoder (Output)", color='purple')
        ax.set_xlabel("Layer")
        ax.set_ylabel("Nullspace Projection Ratio")
        ax.set_title("Encoder vs Decoder Nullspace Alignment")
        ax.legend()
        ax.grid(alpha=0.3)

        # Plot 3: Rank increase
        ax = axes[0, 2]
        ax.plot(layers, [r["avg_rank_increase"] for r in layer_results], 'o-', color='orange')
        ax.set_xlabel("Layer")
        ax.set_ylabel("Average Rank Increase")
        ax.set_title("Effective Rank Change (W → W + ΔW)")
        ax.grid(alpha=0.3)

        # Plot 4: Scatter plot of nullspace vs rank increase
        ax = axes[1, 0]
        nullspace_all = [r["nullspace_projection_ratio"] for r in results]
        rank_inc_all = [r["rank_increase"] for r in results]
        ax.scatter(nullspace_all, rank_inc_all, alpha=0.5, s=20)
        ax.set_xlabel("Nullspace Projection Ratio")
        ax.set_ylabel("Rank Increase")
        ax.set_title("Nullspace Alignment vs Rank Increase")
        ax.grid(alpha=0.3)

        # Plot 5: Distribution of projection ratios
        ax = axes[1, 1]
        nullspace_ratios = [r["nullspace_projection_ratio"] for r in results]
        colspace_ratios = [r["colspace_projection_ratio"] for r in results]

        ax.hist(nullspace_ratios, bins=30, alpha=0.5, label="Nullspace", color='red')
        ax.hist(colspace_ratios, bins=30, alpha=0.5, label="Column Space", color='blue')
        ax.set_xlabel("Projection Ratio")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Projection Ratios")
        ax.legend()
        ax.grid(alpha=0.3)

        # Plot 6: Summary statistics
        ax = axes[1, 2]
        ax.axis('off')

        avg_nullspace = np.mean(nullspace_ratios)
        avg_colspace = np.mean(colspace_ratios)
        avg_rank_inc = np.mean([r["rank_increase"] for r in results])

        # Find layers with highest nullspace alignment
        top_nullspace_layers = sorted(layer_results,
                                     key=lambda x: x["avg_nullspace_ratio"],
                                     reverse=True)[:3]

        summary_text = f"""
        MLP Nullspace Alignment Summary:

        Overall Statistics:
        - Avg Nullspace Projection: {avg_nullspace:.3f}
        - Avg Colspace Projection: {avg_colspace:.3f}
        - Nullspace/Colspace Ratio: {avg_nullspace/(avg_colspace+1e-10):.2f}x
        - Avg Rank Increase: {avg_rank_inc:.1f}

        Top Nullspace-Aligned Layers:
        {chr(10).join([f'  Layer {l["layer"]}: {l["avg_nullspace_ratio"]:.3f}' for l in top_nullspace_layers])}

        Interpretation:
        {'✓ Updates primarily off-manifold' if avg_nullspace > 0.6 else '✗ Updates mostly on-manifold'}
        {'✓ Rank expansion observed' if avg_rank_inc > 5 else '✗ Minimal rank change'}
        """

        ax.text(0.05, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center')

        plt.suptitle(f"MLP Nullspace Alignment Analysis\n{args.model_a} → {args.model_b}")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "mlp_nullspace_alignment.png"), dpi=150)
        plt.close()

    # Save summary JSON
    summary = {
        "model_a": args.model_a,
        "model_b": args.model_b,
        "avg_nullspace_ratio": float(np.mean(nullspace_ratios)) if results else 0,
        "avg_colspace_ratio": float(np.mean(colspace_ratios)) if results else 0,
        "avg_rank_increase": float(np.mean([r["rank_increase"] for r in results])) if results else 0,
        "primarily_off_manifold": bool(np.mean(nullspace_ratios) > 0.6) if results else False,
        "total_matrices_analyzed": len(results),
    }

    with open(os.path.join(args.outdir, "nullspace_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[mlp_nullspace_alignment] ✓ Results saved to {args.outdir}")
    print(f"[mlp_nullspace_alignment] Average nullspace projection: {summary['avg_nullspace_ratio']:.3f}")
    print(f"[mlp_nullspace_alignment] Average colspace projection: {summary['avg_colspace_ratio']:.3f}")
    print(f"[mlp_nullspace_alignment] Updates are {'primarily off-manifold' if summary['primarily_off_manifold'] else 'mostly on-manifold'}")
    log_csv_as_table(os.path.join(args.outdir, "mlp_nullspace_metrics.csv"), "mlp_nullspace_metrics")
    log_csv_as_table(os.path.join(args.outdir, "layer_nullspace_summary.csv"), "layer_nullspace_summary")
    log_plots(args.outdir, "mlp_nullspace")
    finish_wandb()


if __name__ == "__main__":
    main()