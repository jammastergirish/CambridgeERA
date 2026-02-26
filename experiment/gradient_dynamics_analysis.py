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
#   "datasets",
# ]
# ///

"""
Gradient dynamics analysis during unlearning training.

Inspired by subliminal learning research (Experiment 2b), this tracks per-batch
gradient norms across layers during unlearning to reveal optimization stability
patterns. Different unlearning methods and hyperparameters should show
characteristic gradient flow signatures.

Key insights from subliminal learning:
- Real MNIST produced 13× larger gradient norms than uniform noise → instability
- Large, variable gradient norms indicate the model is being pushed outside
  its alignment basin → brittleness
- Stable, controlled gradient flow correlates with robust learning

For unlearning, we expect:
- Robust methods (CB-LAT, filtering) → controlled, stable gradient norms
- Brittle methods (simple GA) → large, variable, potentially exploding gradients
- Optimal hyperparameters → consistent gradient magnitudes across training
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
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import (
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
)


# ---------------------------------------------------------------------------
# Gradient tracking utilities
# ---------------------------------------------------------------------------

def get_parameter_groups(model: nn.Module) -> Dict[str, List[Tuple[str, nn.Parameter]]]:
    """Group parameters by component type for granular gradient tracking."""
    groups = defaultdict(list)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        layer_idx = extract_layer(name)
        component = classify_granular(name)

        if layer_idx is not None and component is not None:
            group_key = f"layer_{layer_idx}_{component}"
            groups[group_key].append((name, param))
        else:
            # Handle non-layer parameters (embeddings, final layer norm, etc.)
            if "embed" in name.lower():
                groups["embeddings"].append((name, param))
            elif "norm" in name.lower() and "final" in name.lower():
                groups["final_norm"].append((name, param))
            elif "lm_head" in name.lower():
                groups["lm_head"].append((name, param))
            else:
                groups["other"].append((name, param))

    return groups


def compute_gradient_norms(parameter_groups: Dict[str, List[Tuple[str, nn.Parameter]]]) -> Dict[str, float]:
    """Compute L2 gradient norms for each parameter group."""
    norms = {}

    total_norm_sq = 0.0

    for group_name, params in parameter_groups.items():
        group_norm_sq = 0.0

        for name, param in params:
            if param.grad is not None:
                param_norm_sq = param.grad.data.norm(dtype=torch.float32).item() ** 2
                group_norm_sq += param_norm_sq
                total_norm_sq += param_norm_sq

        norms[group_name] = np.sqrt(group_norm_sq)

    norms["total"] = np.sqrt(total_norm_sq)
    return norms


# ---------------------------------------------------------------------------
# Data loading for gradient analysis
# ---------------------------------------------------------------------------

def load_analysis_datasets(max_samples: int = 1000) -> Tuple[List[str], List[str]]:
    """Load forget and retain datasets for gradient analysis."""
    # Read the local datasets created by create_datasets.py
    forget_texts = []
    retain_texts = []

    # Load forget set (WMDP-Bio questions)
    try:
        with open("forget.txt", "r") as f:
            forget_texts = [line.strip() for line in f if line.strip()][:max_samples]
    except FileNotFoundError:
        print("Warning: forget.txt not found, using WMDP-Bio from HuggingFace")
        dataset = load_dataset("cais/wmdp-bio", split="test")
        forget_texts = [item["question"] for item in dataset][:max_samples]

    # Load retain set (WikiText-2)
    try:
        with open("retain.txt", "r") as f:
            retain_texts = [line.strip() for line in f if line.strip()][:max_samples]
    except FileNotFoundError:
        print("Warning: retain.txt not found, using WikiText-2 from HuggingFace")
        dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")
        retain_texts = [item["text"] for item in dataset if len(item["text"]) > 50][:max_samples]

    return forget_texts, retain_texts


# ---------------------------------------------------------------------------
# Simulated training with gradient tracking
# ---------------------------------------------------------------------------

def simulate_unlearning_with_gradients(
    model: nn.Module,
    tokenizer,
    forget_texts: List[str],
    retain_texts: List[str],
    method: str = "ga",
    lr: float = 1e-5,
    num_steps: int = 100,
    batch_size: int = 4,
    max_length: int = 512,
    device: str = "cuda",
) -> pd.DataFrame:
    """
    Simulate unlearning training while tracking gradient norms.

    This doesn't actually update the model weights - it just computes gradients
    to analyze the optimization dynamics that would occur during unlearning.
    """
    model.eval()  # Don't update batch norm, dropout, etc.
    parameter_groups = get_parameter_groups(model)

    gradient_records = []

    # Create optimizer for gradient computation (don't actually step)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for step in tqdm(range(num_steps), desc="Tracking gradients"):
        # Alternate between forget and retain batches
        if step % 2 == 0:
            texts = np.random.choice(forget_texts, size=min(batch_size, len(forget_texts)), replace=False)
            split = "forget"
        else:
            texts = np.random.choice(retain_texts, size=min(batch_size, len(retain_texts)), replace=False)
            split = "retain"

        # Tokenize batch
        inputs = tokenizer(
            list(texts),
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        # Forward pass
        with torch.enable_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

            # Apply method-specific loss modification
            if method == "ga" and split == "forget":
                loss = -loss  # Gradient ascent on forget set
            elif method == "ga_simple" and split == "forget":
                loss = -loss
                if split == "retain":
                    continue  # Skip retain loss for GA simple

            # Backward pass to compute gradients
            optimizer.zero_grad()
            loss.backward()

            # Record gradient norms
            gradient_norms = compute_gradient_norms(parameter_groups)

            for group_name, norm in gradient_norms.items():
                gradient_records.append({
                    "step": step,
                    "split": split,
                    "group": group_name,
                    "gradient_norm": norm,
                    "loss": loss.item(),
                    "method": method
                })

    return pd.DataFrame(gradient_records)


# ---------------------------------------------------------------------------
# Analysis and visualization
# ---------------------------------------------------------------------------

def analyze_gradient_patterns(df: pd.DataFrame) -> Dict:
    """Analyze gradient patterns for stability and optimization health."""
    summary = {}

    # Overall statistics
    total_df = df[df["group"] == "total"]
    summary["mean_gradient_norm"] = total_df["gradient_norm"].mean()
    summary["std_gradient_norm"] = total_df["gradient_norm"].std()
    summary["max_gradient_norm"] = total_df["gradient_norm"].max()
    summary["cv_gradient_norm"] = summary["std_gradient_norm"] / summary["mean_gradient_norm"]

    # Split-specific analysis
    for split in ["forget", "retain"]:
        split_df = total_df[total_df["split"] == split]
        if len(split_df) > 0:
            summary[f"{split}_mean_norm"] = split_df["gradient_norm"].mean()
            summary[f"{split}_std_norm"] = split_df["gradient_norm"].std()

    # Stability indicators
    summary["exploding_gradients"] = (total_df["gradient_norm"] > 10.0).sum()
    summary["vanishing_gradients"] = (total_df["gradient_norm"] < 1e-6).sum()

    # Component-wise analysis
    component_stats = {}
    for group in df["group"].unique():
        if group != "total":
            group_df = df[df["group"] == group]
            component_stats[group] = {
                "mean_norm": group_df["gradient_norm"].mean(),
                "max_norm": group_df["gradient_norm"].max(),
                "std_norm": group_df["gradient_norm"].std()
            }

    summary["component_stats"] = component_stats

    return summary


def create_gradient_plots(df: pd.DataFrame, outdir: str, title: str = "Gradient Dynamics"):
    """Create comprehensive gradient analysis plots."""
    os.makedirs(outdir, exist_ok=True)

    # 1. Total gradient norm over training steps
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f"{title} - Gradient Dynamics Analysis", fontsize=16)

    # Plot 1: Total gradient norm timeline
    total_df = df[df["group"] == "total"]
    for split in ["forget", "retain"]:
        split_data = total_df[total_df["split"] == split]
        if len(split_data) > 0:
            axes[0, 0].plot(split_data["step"], split_data["gradient_norm"],
                           label=f"{split} set", alpha=0.7, marker='o', markersize=2)

    axes[0, 0].set_xlabel("Training Step")
    axes[0, 0].set_ylabel("Gradient Norm")
    axes[0, 0].set_title("Total Gradient Norm Over Training")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Gradient norm distribution
    axes[0, 1].hist(total_df["gradient_norm"], bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(total_df["gradient_norm"].mean(), color='red', linestyle='--',
                       label=f'Mean: {total_df["gradient_norm"].mean():.3f}')
    axes[0, 1].set_xlabel("Gradient Norm")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Gradient Norm Distribution")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Component-wise gradient norms (mean)
    component_df = df[df["group"] != "total"].groupby("group")["gradient_norm"].mean().sort_values(ascending=False)
    axes[1, 0].bar(range(len(component_df)), component_df.values)
    axes[1, 0].set_xticks(range(len(component_df)))
    axes[1, 0].set_xticklabels(component_df.index, rotation=45, ha='right')
    axes[1, 0].set_ylabel("Mean Gradient Norm")
    axes[1, 0].set_title("Mean Gradient Norm by Component")
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Forget vs Retain gradient comparison
    forget_norms = total_df[total_df["split"] == "forget"]["gradient_norm"]
    retain_norms = total_df[total_df["split"] == "retain"]["gradient_norm"]

    if len(forget_norms) > 0 and len(retain_norms) > 0:
        axes[1, 1].boxplot([forget_norms, retain_norms], labels=["Forget", "Retain"])
        axes[1, 1].set_ylabel("Gradient Norm")
        axes[1, 1].set_title("Gradient Norms: Forget vs Retain")
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "gradient_dynamics.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Additional plot: Layer-wise gradient heatmap
    layer_df = df[df["group"].str.contains("layer_")].copy()
    if len(layer_df) > 0:
        # Extract layer numbers and components
        layer_df["layer_num"] = layer_df["group"].str.extract(r'layer_(\d+)').astype(int)
        layer_df["component"] = layer_df["group"].str.extract(r'layer_\d+_(.+)')

        # Create pivot table for heatmap
        pivot_data = layer_df.groupby(["layer_num", "component"])["gradient_norm"].mean().unstack()

        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(pivot_data.T, aspect='auto', cmap='viridis', interpolation='nearest')

        ax.set_xlabel("Layer")
        ax.set_ylabel("Component")
        ax.set_title(f"{title} - Layer-wise Gradient Norms (Mean)")
        ax.set_xticks(range(len(pivot_data.index)))
        ax.set_xticklabels(pivot_data.index)
        ax.set_yticks(range(len(pivot_data.columns)))
        ax.set_yticklabels(pivot_data.columns)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Mean Gradient Norm")

        # Add text annotations
        for i in range(len(pivot_data.columns)):
            for j in range(len(pivot_data.index)):
                text = ax.text(j, i, f'{pivot_data.iloc[j, i]:.2f}',
                              ha="center", va="center", color="white", fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "gradient_heatmap_layers.png"), dpi=300, bbox_inches='tight')
        plt.close()


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def run_gradient_dynamics_analysis(
    model_a_path: str,
    model_b_path: str,
    outdir: str,
    method: str = "ga",
    num_steps: int = 100,
    device: str = "auto",
    dtype: str = "auto",
    max_samples: int = 500,
) -> None:
    """
    Run gradient dynamics analysis comparing two models.

    This simulates unlearning training on both models to compare their
    optimization dynamics and gradient flow patterns.
    """
    device = resolve_device(device)
    dtype = resolve_dtype(dtype)

    os.makedirs(outdir, exist_ok=True)

    # Load datasets
    print("Loading analysis datasets...")
    forget_texts, retain_texts = load_analysis_datasets(max_samples)
    print(f"Loaded {len(forget_texts)} forget texts, {len(retain_texts)} retain texts")

    # Load models and tokenizer
    print(f"Loading models...")
    print(f"  Model A (baseline): {model_a_path}")
    print(f"  Model B (target): {model_b_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_a_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_a = AutoModelForCausalLM.from_pretrained(
        model_a_path, torch_dtype=dtype, device_map=device
    )
    model_b = AutoModelForCausalLM.from_pretrained(
        model_b_path, torch_dtype=dtype, device_map=device
    )

    # Run gradient analysis on both models
    results = []

    for model, model_name in [(model_a, "baseline"), (model_b, "target")]:
        print(f"Analyzing gradient dynamics for {model_name}...")

        df = simulate_unlearning_with_gradients(
            model=model,
            tokenizer=tokenizer,
            forget_texts=forget_texts,
            retain_texts=retain_texts,
            method=method,
            num_steps=num_steps,
            device=device,
        )

        df["model"] = model_name
        results.append(df)

    # Combine results
    combined_df = pd.concat(results, ignore_index=True)

    # Save raw data
    write_csv(combined_df, os.path.join(outdir, "gradient_dynamics.csv"))

    # Analyze patterns for each model
    summary = {}
    for model_name in ["baseline", "target"]:
        model_df = combined_df[combined_df["model"] == model_name]
        summary[model_name] = analyze_gradient_patterns(model_df)

    # Save summary
    with open(os.path.join(outdir, "gradient_analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Create plots for each model
    plot_outdir = os.path.join(outdir, "plots")
    for model_name in ["baseline", "target"]:
        model_df = combined_df[combined_df["model"] == model_name]
        create_gradient_plots(
            model_df,
            plot_outdir,
            title=f"{model_name.capitalize()} Model"
        )

    # Comparative analysis
    print("Creating comparative analysis...")

    # Compare gradient stability between models
    baseline_total = combined_df[(combined_df["model"] == "baseline") & (combined_df["group"] == "total")]
    target_total = combined_df[(combined_df["model"] == "target") & (combined_df["group"] == "total")]

    comparison = {
        "baseline_vs_target": {
            "mean_gradient_ratio": target_total["gradient_norm"].mean() / baseline_total["gradient_norm"].mean(),
            "stability_ratio": (target_total["gradient_norm"].std() / target_total["gradient_norm"].mean()) /
                             (baseline_total["gradient_norm"].std() / baseline_total["gradient_norm"].mean()),
            "max_gradient_ratio": target_total["gradient_norm"].max() / baseline_total["gradient_norm"].max()
        }
    }

    summary["comparison"] = comparison

    # Update summary file
    with open(os.path.join(outdir, "gradient_analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Gradient dynamics analysis complete. Results saved to {outdir}")
    print(f"Key findings:")
    print(f"  Baseline mean gradient norm: {summary['baseline']['mean_gradient_norm']:.4f}")
    print(f"  Target mean gradient norm: {summary['target']['mean_gradient_norm']:.4f}")
    print(f"  Gradient magnitude ratio: {comparison['baseline_vs_target']['mean_gradient_ratio']:.4f}")
    print(f"  Stability ratio: {comparison['baseline_vs_target']['stability_ratio']:.4f}")


# ---------------------------------------------------------------------------
# CLI interface
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Gradient dynamics analysis during simulated unlearning")
    parser.add_argument("--model-a", required=True, help="Baseline model path")
    parser.add_argument("--model-b", required=True, help="Target model path")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--method", default="ga", choices=["ga", "ga_simple", "dpo", "npo"],
                       help="Unlearning method to simulate")
    parser.add_argument("--num-steps", type=int, default=100, help="Number of gradient steps to simulate")
    parser.add_argument("--device", default="auto", help="Device to use")
    parser.add_argument("--dtype", default="auto", help="Data type")
    parser.add_argument("--max-samples", type=int, default=500, help="Max samples per dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--wandb-project", help="W&B project name")
    parser.add_argument("--wandb-name", help="W&B run name")

    args = parser.parse_args()

    # Initialize W&B if specified
    wandb_run = None
    if args.wandb_project:
        wandb_run = init_wandb(args.wandb_project, args.wandb_name or "gradient_dynamics")

    # Set random seed
    import random
    import torch
    import numpy as np
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    try:
        run_gradient_dynamics_analysis(
            model_a_path=args.model_a,
            model_b_path=args.model_b,
            outdir=args.outdir,
            method=args.method,
            num_steps=args.num_steps,
            device=args.device,
            dtype=args.dtype,
            max_samples=args.max_samples,
        )

        # Log results to W&B
        if wandb_run:
            log_csv_as_table("gradient_dynamics", os.path.join(args.outdir, "gradient_dynamics.csv"))
            log_plots(os.path.join(args.outdir, "plots"))

    finally:
        if wandb_run:
            finish_wandb()


if __name__ == "__main__":
    main()