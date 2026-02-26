#!/usr/bin/env python
# /// script
# dependencies = [
#   "torch",
#   "transformers",
#   "numpy",
#   "matplotlib",
#   "pandas",
#   "tqdm",
#   "wandb",
# ]
# ///

"""
Distance-from-initialization analysis for unlearning models.

Inspired by subliminal learning Experiment 3b, this tracks how far unlearned
models drift from their original pretrained initialization. Key insights:

From subliminal learning:
- Peak performance occurs at specific distance from initialization (≈4-5 units)
- Too close to init = insufficient learning
- Too far from init = alignment basin exit → brittleness
- The distance metric reveals the "Goldilocks zone" for stable learning

For unlearning applications:
- Robust methods should stay within optimal distance from pretrained weights
- Brittle methods may show excessive drift or insufficient change
- Distance correlates with tamper-resistance (harder to fine-tune back)

This analysis enhances the existing weight comparison with temporal dynamics
and provides direct measurement of the alignment basin concept.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import (
    SmartLoader,
    comparison_outdir,
    init_wandb,
    finish_wandb,
    log_csv_as_table,
    log_plots,
    resolve_device,
    resolve_dtype,
    write_csv,
    extract_layer,
    classify_granular,
    stable_rank_and_spectral,
)


# ---------------------------------------------------------------------------
# Distance computation utilities
# ---------------------------------------------------------------------------

def compute_parameter_distances(
    init_params: Dict[str, torch.Tensor],
    target_params: Dict[str, torch.Tensor],
    granular: bool = True
) -> Dict[str, Dict]:
    """
    Compute distance metrics between initialization and target parameters.

    Returns hierarchical structure:
    - Per-matrix distances
    - Per-component aggregates
    - Per-layer aggregates
    - Global totals
    """
    results = {
        "per_matrix": [],
        "per_component": defaultdict(list),
        "per_layer": defaultdict(list),
        "global": {}
    }

    total_distance_sq = 0.0
    total_params = 0

    for param_name, init_param in init_params.items():
        if param_name not in target_params:
            continue

        target_param = target_params[param_name]

        # Ensure same device and dtype for computation
        if init_param.device != target_param.device:
            target_param = target_param.to(init_param.device)
        if init_param.dtype != target_param.dtype:
            target_param = target_param.to(init_param.dtype)

        # Compute parameter-level distances
        diff = target_param - init_param

        # L2 distance (Frobenius norm for matrices)
        l2_distance = torch.norm(diff, p=2).item()

        # Relative L2 distance
        init_norm = torch.norm(init_param, p=2).item()
        rel_l2_distance = l2_distance / init_norm if init_norm > 1e-8 else 0.0

        # Cosine similarity (1 = unchanged direction)
        init_flat = init_param.flatten()
        target_flat = target_param.flatten()
        cosine_sim = torch.nn.functional.cosine_similarity(
            init_flat.unsqueeze(0), target_flat.unsqueeze(0)
        ).item()

        # Element-wise statistics
        n_params = diff.numel()
        mean_abs_change = torch.abs(diff).mean().item()
        std_change = diff.std().item()
        max_abs_change = torch.abs(diff).max().item()

        # Record per-matrix results
        layer_idx = extract_layer(param_name)
        component = classify_granular(param_name) if granular else None

        matrix_result = {
            "param_name": param_name,
            "layer": layer_idx,
            "component": component,
            "n_params": n_params,
            "l2_distance": l2_distance,
            "rel_l2_distance": rel_l2_distance,
            "cosine_similarity": cosine_sim,
            "mean_abs_change": mean_abs_change,
            "std_change": std_change,
            "max_abs_change": max_abs_change,
            "init_norm": init_norm,
        }

        results["per_matrix"].append(matrix_result)

        # Accumulate for aggregates
        total_distance_sq += l2_distance ** 2
        total_params += n_params

        # Group by component and layer
        if component is not None:
            results["per_component"][component].append(matrix_result)
        if layer_idx is not None:
            layer_key = f"layer_{layer_idx}"
            if component:
                layer_key += f"_{component}"
            results["per_layer"][layer_key].append(matrix_result)

    # Compute global aggregates
    results["global"]["total_l2_distance"] = np.sqrt(total_distance_sq)
    results["global"]["total_params"] = total_params
    results["global"]["mean_l2_distance"] = np.sqrt(total_distance_sq / len(results["per_matrix"]))

    return results


def aggregate_distance_results(distance_dict: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convert distance results to DataFrames for analysis and plotting."""

    # Per-matrix DataFrame
    per_matrix_df = pd.DataFrame(distance_dict["per_matrix"])

    # Per-component aggregates
    component_aggregates = []
    for component, matrices in distance_dict["per_component"].items():
        if not matrices:
            continue

        # Aggregate metrics across matrices in this component
        l2_distances = [m["l2_distance"] for m in matrices]
        rel_distances = [m["rel_l2_distance"] for m in matrices]
        cosine_sims = [m["cosine_similarity"] for m in matrices]

        component_aggregates.append({
            "component": component,
            "n_matrices": len(matrices),
            "total_l2_distance": np.sqrt(sum(d**2 for d in l2_distances)),
            "mean_l2_distance": np.mean(l2_distances),
            "std_l2_distance": np.std(l2_distances),
            "mean_rel_l2_distance": np.mean(rel_distances),
            "mean_cosine_similarity": np.mean(cosine_sims),
            "total_params": sum(m["n_params"] for m in matrices)
        })

    per_component_df = pd.DataFrame(component_aggregates)

    # Per-layer aggregates
    layer_aggregates = []
    for layer_key, matrices in distance_dict["per_layer"].items():
        if not matrices:
            continue

        # Extract layer number and component
        parts = layer_key.split("_")
        layer_num = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
        component = "_".join(parts[2:]) if len(parts) > 2 else "all"

        l2_distances = [m["l2_distance"] for m in matrices]

        layer_aggregates.append({
            "layer": layer_num,
            "component": component,
            "layer_key": layer_key,
            "n_matrices": len(matrices),
            "layer_l2_distance": np.sqrt(sum(d**2 for d in l2_distances)),
            "mean_l2_distance": np.mean(l2_distances),
            "total_params": sum(m["n_params"] for m in matrices)
        })

    per_layer_df = pd.DataFrame(layer_aggregates)

    return per_matrix_df, per_component_df, per_layer_df


# ---------------------------------------------------------------------------
# Visualization functions
# ---------------------------------------------------------------------------

def create_distance_plots(
    per_matrix_df: pd.DataFrame,
    per_component_df: pd.DataFrame,
    per_layer_df: pd.DataFrame,
    global_stats: Dict,
    outdir: str,
    title: str = "Distance from Initialization"
):
    """Create comprehensive distance analysis plots."""
    os.makedirs(outdir, exist_ok=True)

    # Main figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"{title}", fontsize=16)

    # Plot 1: Per-component total distances
    if not per_component_df.empty:
        comp_data = per_component_df.sort_values("total_l2_distance", ascending=False)
        bars = axes[0, 0].bar(range(len(comp_data)), comp_data["total_l2_distance"])
        axes[0, 0].set_xticks(range(len(comp_data)))
        axes[0, 0].set_xticklabels(comp_data["component"], rotation=45, ha='right')
        axes[0, 0].set_ylabel("Total L2 Distance")
        axes[0, 0].set_title("Distance by Component Type")
        axes[0, 0].grid(True, alpha=0.3)

        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{height:.2f}', ha='center', va='bottom', fontsize=10)

    # Plot 2: Layer-wise distance progression (aggregated across components)
    if not per_layer_df.empty:
        # Aggregate by layer number across all components
        layer_totals = per_layer_df.groupby("layer")["layer_l2_distance"].sum().reset_index()
        layer_totals = layer_totals.sort_values("layer")

        axes[0, 1].plot(layer_totals["layer"], layer_totals["layer_l2_distance"],
                       marker='o', linewidth=2, markersize=6)
        axes[0, 1].set_xlabel("Layer")
        axes[0, 1].set_ylabel("Total L2 Distance")
        axes[0, 1].set_title("Distance by Layer")
        axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Cosine similarity distribution
    if not per_matrix_df.empty:
        cosine_sims = per_matrix_df["cosine_similarity"].dropna()
        axes[1, 0].hist(cosine_sims, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(cosine_sims.mean(), color='red', linestyle='--',
                          label=f'Mean: {cosine_sims.mean():.3f}')
        axes[1, 0].set_xlabel("Cosine Similarity")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_title("Parameter Direction Preservation")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Distance vs Parameter Count (log scale)
    if not per_matrix_df.empty:
        scatter_data = per_matrix_df.dropna(subset=["l2_distance", "n_params"])
        if not scatter_data.empty:
            axes[1, 1].scatter(scatter_data["n_params"], scatter_data["l2_distance"],
                              alpha=0.6, s=30)
            axes[1, 1].set_xscale('log')
            axes[1, 1].set_yscale('log')
            axes[1, 1].set_xlabel("Number of Parameters")
            axes[1, 1].set_ylabel("L2 Distance")
            axes[1, 1].set_title("Distance vs Matrix Size")
            axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "distance_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Additional plot: Component-layer heatmap
    if not per_layer_df.empty and "component" in per_layer_df.columns:
        # Create pivot table for heatmap
        pivot_data = per_layer_df.pivot_table(
            values="layer_l2_distance",
            index="component",
            columns="layer",
            aggfunc="sum",
            fill_value=0
        )

        if not pivot_data.empty:
            fig, ax = plt.subplots(figsize=(14, 8))
            im = ax.imshow(pivot_data.values, aspect='auto', cmap='viridis', interpolation='nearest')

            ax.set_xlabel("Layer")
            ax.set_ylabel("Component")
            ax.set_title(f"{title} - Layer × Component Heatmap")
            ax.set_xticks(range(len(pivot_data.columns)))
            ax.set_xticklabels(pivot_data.columns)
            ax.set_yticks(range(len(pivot_data.index)))
            ax.set_yticklabels(pivot_data.index)

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("L2 Distance")

            plt.tight_layout()
            plt.savefig(os.path.join(outdir, "distance_heatmap.png"), dpi=300, bbox_inches='tight')
            plt.close()

    print(f"Distance analysis plots saved to {outdir}")


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def run_distance_analysis(
    init_model_path: str,
    target_model_path: str,
    outdir: str,
    device: str = "auto",
    dtype: str = "auto"
) -> Dict:
    """
    Run distance-from-initialization analysis.

    Args:
        init_model_path: Path to initialization/pretrained model
        target_model_path: Path to target (unlearned) model
        outdir: Output directory for results
        device: Device to use for computation
        dtype: Data type for computation

    Returns:
        Dictionary containing analysis results
    """
    device = resolve_device(device)
    dtype = resolve_dtype(dtype)

    os.makedirs(outdir, exist_ok=True)

    print(f"Loading models for distance analysis...")
    print(f"  Initialization model: {init_model_path}")
    print(f"  Target model: {target_model_path}")

    # Load models
    loader_init = SmartLoader(init_model_path, device=device, dtype=dtype)
    loader_target = SmartLoader(target_model_path, device=device, dtype=dtype)

    init_params = {}
    target_params = {}

    # Load all parameters
    print("Loading initialization model parameters...")
    with loader_init:
        for name, param in loader_init.model.named_parameters():
            init_params[name] = param.clone().cpu()

    print("Loading target model parameters...")
    with loader_target:
        for name, param in loader_target.model.named_parameters():
            target_params[name] = param.clone().cpu()

    # Compute distance metrics
    print("Computing distance metrics...")
    distance_results = compute_parameter_distances(init_params, target_params)

    # Convert to DataFrames
    per_matrix_df, per_component_df, per_layer_df = aggregate_distance_results(distance_results)

    # Save raw data
    per_matrix_df.to_csv(os.path.join(outdir, "distance_per_matrix.csv"), index=False)
    per_component_df.to_csv(os.path.join(outdir, "distance_per_component.csv"), index=False)
    per_layer_df.to_csv(os.path.join(outdir, "distance_per_layer.csv"), index=False)

    # Save summary statistics
    summary = {
        "global_stats": distance_results["global"],
        "component_summary": per_component_df.to_dict("records") if not per_component_df.empty else [],
        "key_metrics": {
            "total_l2_distance": distance_results["global"]["total_l2_distance"],
            "mean_cosine_similarity": per_matrix_df["cosine_similarity"].mean() if not per_matrix_df.empty else 0,
            "max_layer_distance": per_layer_df["layer_l2_distance"].max() if not per_layer_df.empty else 0,
            "most_changed_component": per_component_df.loc[per_component_df["total_l2_distance"].idxmax(), "component"] if not per_component_df.empty else None
        }
    }

    with open(os.path.join(outdir, "distance_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Create visualizations
    plot_outdir = os.path.join(outdir, "plots")
    model_name = os.path.basename(target_model_path)
    create_distance_plots(
        per_matrix_df, per_component_df, per_layer_df,
        distance_results["global"], plot_outdir,
        title=f"Distance from Init: {model_name}"
    )

    print(f"Distance analysis complete. Results saved to {outdir}")
    print(f"Key findings:")
    print(f"  Total L2 distance from initialization: {summary['key_metrics']['total_l2_distance']:.4f}")
    print(f"  Mean cosine similarity: {summary['key_metrics']['mean_cosine_similarity']:.4f}")
    print(f"  Most changed component: {summary['key_metrics']['most_changed_component']}")

    return summary


# ---------------------------------------------------------------------------
# CLI interface
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Distance-from-initialization analysis")
    parser.add_argument("--init-model", required=True,
                       help="Initialization/pretrained model path")
    parser.add_argument("--target-model", required=True,
                       help="Target (unlearned) model path")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--device", default="auto", help="Device to use")
    parser.add_argument("--dtype", default="auto", help="Data type")
    parser.add_argument("--wandb-project", help="W&B project name")
    parser.add_argument("--wandb-name", help="W&B run name")

    args = parser.parse_args()

    # Initialize W&B if specified
    wandb_run = None
    if args.wandb_project:
        wandb_run = init_wandb(args.wandb_project, args.wandb_name or "distance_analysis")

    try:
        summary = run_distance_analysis(
            init_model_path=args.init_model,
            target_model_path=args.target_model,
            outdir=args.outdir,
            device=args.device,
            dtype=args.dtype,
        )

        # Log results to W&B
        if wandb_run:
            import wandb
            wandb.log(summary["key_metrics"])
            log_csv_as_table("distance_per_matrix", os.path.join(args.outdir, "distance_per_matrix.csv"))
            log_csv_as_table("distance_per_component", os.path.join(args.outdir, "distance_per_component.csv"))
            log_plots(os.path.join(args.outdir, "plots"))

    finally:
        if wandb_run:
            finish_wandb()


if __name__ == "__main__":
    main()