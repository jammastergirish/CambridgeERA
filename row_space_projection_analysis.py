#!/usr/bin/env python
# /// script
# dependencies = [
#   "torch",
#   "transformers",
#   "numpy",
#   "matplotlib",
#   "tqdm",
#   "safetensors",
#   "huggingface_hub",
# ]
# ///

"""
Analyze how pre-activations of forget/retain data project onto parameter update row space.
Tests if forget data activations align more with update directions than retain data.
"""

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import tempfile
import shutil

from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import resolve_device, resolve_dtype, write_csv, extract_layer
from collect_param_stats import SmartLoader


class ActivationCapture:
    """Hook to capture pre-MLP activations."""

    def __init__(self):
        self.activations = []
        self.hooks = []

    def capture_hook(self, module, input, output):
        # input is a tuple, first element is the pre-activation
        if isinstance(input, tuple):
            act = input[0]
        else:
            act = input
        self.activations.append(act.detach().cpu())

    def register_hooks(self, model, layer_indices):
        """Register hooks on specified MLP layers."""
        self.clear()

        # Handle different model architectures
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            # LLaMA-style
            for idx in layer_indices:
                if idx < len(model.model.layers):
                    mlp = model.model.layers[idx].mlp
                    hook = mlp.register_forward_hook(self.capture_hook)
                    self.hooks.append(hook)
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            # GPT-style
            for idx in layer_indices:
                if idx < len(model.transformer.h):
                    mlp = model.transformer.h[idx].mlp
                    hook = mlp.register_forward_hook(self.capture_hook)
                    self.hooks.append(hook)
        elif hasattr(model, 'layers'):
            # Generic transformer
            for idx in layer_indices:
                if idx < len(model.layers):
                    if hasattr(model.layers[idx], 'mlp'):
                        mlp = model.layers[idx].mlp
                        hook = mlp.register_forward_hook(self.capture_hook)
                        self.hooks.append(hook)

    def clear(self):
        """Remove hooks and clear activations."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = []


def get_mlp_weight_updates(loader_a, loader_b, layer_idx, device, dtype):
    """Get MLP weight updates for a specific layer."""
    updates = {}

    # Try different naming conventions
    possible_names = [
        f"model.layers.{layer_idx}.mlp.gate_proj.weight",
        f"model.layers.{layer_idx}.mlp.up_proj.weight",
        f"model.layers.{layer_idx}.mlp.down_proj.weight",
        f"transformer.h.{layer_idx}.mlp.c_fc.weight",
        f"transformer.h.{layer_idx}.mlp.c_proj.weight",
        f"layers.{layer_idx}.mlp.fc1.weight",
        f"layers.{layer_idx}.mlp.fc2.weight",
    ]

    for name in possible_names:
        # Try to load from both models
        Wa = loader_a.get_param(name, device, dtype)
        if Wa is None or Wa.ndim != 2:
            continue

        Wb = loader_b.get_param(name, device, dtype)
        if Wb is None or Wb.shape != Wa.shape:
            continue

        # Compute update
        dW = (Wb - Wa).cpu().float().numpy()

        # Store update
        if 'gate' in name or 'up' in name or 'fc1' in name or 'c_fc' in name:
            updates["encoder"] = dW
        else:
            updates["decoder"] = dW

    return updates


def compute_row_space_projection(activations, dW, top_k=20):
    """
    Compute how activations project onto the row space of weight updates.

    Args:
        activations: Pre-MLP activations [batch, seq, hidden]
        dW: Weight update matrix [out_dim, in_dim]
        top_k: Number of top singular vectors to use

    Returns:
        dict: Projection metrics
    """
    if len(activations) == 0 or dW is None:
        return None

    # Flatten activations
    acts_flat = np.vstack([a.reshape(-1, a.shape[-1]) for a in activations])

    # Compute SVD of weight update (row space)
    # dW.T gives us row vectors as columns
    U, s, Vt = np.linalg.svd(dW.T, full_matrices=False)

    # U contains the right singular vectors (row space basis)
    k = min(top_k, U.shape[1])
    U_top = U[:, :k]

    # Project activations onto row space
    proj = acts_flat @ U_top
    proj_norm = np.linalg.norm(proj, axis=1).mean()

    # Original activation norm
    orig_norm = np.linalg.norm(acts_flat, axis=1).mean()

    # Projection ratio
    proj_ratio = proj_norm / (orig_norm + 1e-10)

    # Also compute variance explained by projection
    proj_var = np.var(proj)
    orig_var = np.var(acts_flat)
    var_ratio = proj_var / (orig_var + 1e-10)

    # Compute alignment with individual singular vectors
    alignments = []
    for i in range(min(5, k)):  # Top 5 singular vectors
        align = np.abs(acts_flat @ U[:, i]).mean()
        alignments.append(float(align))

    return {
        "projection_norm": float(proj_norm),
        "original_norm": float(orig_norm),
        "projection_ratio": float(proj_ratio),
        "variance_ratio": float(var_ratio),
        "top_alignments": alignments,
        "num_samples": len(acts_flat),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-a", required=True, help="Baseline model")
    ap.add_argument("--model-b", required=True, help="Unlearned/finetuned model")
    ap.add_argument("--forget-text", default="data/forget.txt")
    ap.add_argument("--retain-text", default="data/retain.txt")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--max-samples", type=int, default=500,
                   help="Max texts per split to process (default: 500)")
    ap.add_argument("--layers-to-analyze", type=str, default=None,
                   help="Comma-separated layer indices (default: every 4th layer)")
    ap.add_argument("--outdir", default="outputs/row_space_projection")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    # Load texts
    print("[row_space_projection] Loading forget/retain texts...")
    with open(args.forget_text, "r") as f:
        forget_texts = [line.strip() for line in f if line.strip()][:200]  # Limit for speed
    with open(args.retain_text, "r") as f:
        retain_texts = [line.strip() for line in f if line.strip()][:200]

    # Cap sample count
    if len(forget_texts) > args.max_samples:
        forget_texts = forget_texts[:args.max_samples]
    if len(retain_texts) > args.max_samples:
        retain_texts = retain_texts[:args.max_samples]

    print(f"[row_space_projection] Loaded {len(forget_texts)} forget texts, {len(retain_texts)} retain texts (max-samples={args.max_samples})")

    # Load tokenizer and models
    print(f"[row_space_projection] Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_a)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model for activation capture
    model_a = AutoModelForCausalLM.from_pretrained(args.model_a, torch_dtype=dtype).to(device)
    model_a.eval()

    # Load weight loaders for parameter differences
    loader_a = SmartLoader(args.model_a)
    loader_b = SmartLoader(args.model_b)

    # Determine layers to analyze
    if hasattr(model_a, 'gpt_neox') and hasattr(model_a.gpt_neox, 'layers'):
        num_layers = len(model_a.gpt_neox.layers)
    elif hasattr(model_a, 'model') and hasattr(model_a.model, 'layers'):
        num_layers = len(model_a.model.layers)
    elif hasattr(model_a, 'transformer') and hasattr(model_a.transformer, 'h'):
        num_layers = len(model_a.transformer.h)
    else:
        print("Warning: Could not determine number of layers")
        num_layers = 32  # Default guess

    if args.layers_to_analyze:
        layers = [int(x) for x in args.layers_to_analyze.split(",")]
    else:
        # Analyze every 4th layer
        layers = list(range(0, num_layers, 4))

    print(f"[row_space_projection] Will analyze layers: {layers}")

    results = []
    os.makedirs(args.outdir, exist_ok=True)

    # Create activation capturer
    capturer = ActivationCapture()

    # Process each layer
    for layer_idx in tqdm(layers, desc="Projecting onto row space", unit="layer"):
        print(f"\nLayer {layer_idx}:")

        # Get MLP weight updates for this layer
        print("  Loading weight updates...")
        updates = get_mlp_weight_updates(loader_a, loader_b, layer_idx, device, dtype)

        if not updates:
            print(f"  No MLP weights found for layer {layer_idx}")
            continue

        # Register hooks for this layer
        capturer.register_hooks(model_a, [layer_idx])

        # Collect forget activations
        print("  Collecting forget activations...")
        capturer.clear()
        capturer.activations = []

        for i in range(0, len(forget_texts), args.batch_size):
            batch = forget_texts[i:i+args.batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True,
                             truncation=True, max_length=args.max_length).to(device)

            with torch.no_grad():
                _ = model_a(**inputs)

        forget_acts = capturer.activations.copy()

        # Collect retain activations
        print("  Collecting retain activations...")
        capturer.clear()
        capturer.activations = []

        for i in range(0, len(retain_texts), args.batch_size):
            batch = retain_texts[i:i+args.batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True,
                             truncation=True, max_length=args.max_length).to(device)

            with torch.no_grad():
                _ = model_a(**inputs)

        retain_acts = capturer.activations.copy()

        # Compute projections for encoder weights
        if "encoder" in updates:
            print("  Computing encoder projections...")
            forget_proj = compute_row_space_projection(forget_acts, updates["encoder"])
            retain_proj = compute_row_space_projection(retain_acts, updates["encoder"])

            if forget_proj and retain_proj:
                result = {
                    "layer": layer_idx,
                    "weight_type": "encoder",
                    "forget_proj_ratio": forget_proj["projection_ratio"],
                    "retain_proj_ratio": retain_proj["projection_ratio"],
                    "forget_proj_norm": forget_proj["projection_norm"],
                    "retain_proj_norm": retain_proj["projection_norm"],
                    "projection_diff": forget_proj["projection_ratio"] - retain_proj["projection_ratio"],
                    "forget_stronger": forget_proj["projection_ratio"] > retain_proj["projection_ratio"],
                }
                results.append(result)

                # Save detailed projections
                np.savez_compressed(
                    os.path.join(args.outdir, f"projections_layer_{layer_idx}_encoder.npz"),
                    forget_alignments=forget_proj["top_alignments"],
                    retain_alignments=retain_proj["top_alignments"],
                    dW_shape=updates["encoder"].shape,
                )

        # Clear hooks
        capturer.clear()

    # Save results
    if results:
        write_csv(
            os.path.join(args.outdir, "row_space_projections.csv"),
            results,
            ["layer", "weight_type", "forget_proj_ratio", "retain_proj_ratio",
             "forget_proj_norm", "retain_proj_norm", "projection_diff", "forget_stronger"]
        )

        # Compute layer-wise summary
        layer_summary = {}
        for r in results:
            layer = r["layer"]
            if layer not in layer_summary:
                layer_summary[layer] = {
                    "forget_ratios": [],
                    "retain_ratios": [],
                    "diffs": [],
                }
            layer_summary[layer]["forget_ratios"].append(r["forget_proj_ratio"])
            layer_summary[layer]["retain_ratios"].append(r["retain_proj_ratio"])
            layer_summary[layer]["diffs"].append(r["projection_diff"])

        layer_results = []
        for layer in sorted(layer_summary.keys()):
            layer_results.append({
                "layer": layer,
                "avg_forget_proj": np.mean(layer_summary[layer]["forget_ratios"]),
                "avg_retain_proj": np.mean(layer_summary[layer]["retain_ratios"]),
                "avg_diff": np.mean(layer_summary[layer]["diffs"]),
            })

        write_csv(
            os.path.join(args.outdir, "layer_projection_summary.csv"),
            layer_results,
            ["layer", "avg_forget_proj", "avg_retain_proj", "avg_diff"]
        )

        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        layers_plot = [r["layer"] for r in layer_results]

        # Plot 1: Projection ratios by layer
        ax = axes[0, 0]
        ax.plot(layers_plot, [r["avg_forget_proj"] for r in layer_results], 'o-',
                label="Forget", color='red')
        ax.plot(layers_plot, [r["avg_retain_proj"] for r in layer_results], 's-',
                label="Retain", color='blue')
        ax.set_xlabel("Layer")
        ax.set_ylabel("Projection Ratio")
        ax.set_title("Activation Projection onto Update Row Space")
        ax.legend()
        ax.grid(alpha=0.3)

        # Plot 2: Projection difference
        ax = axes[0, 1]
        diffs = [r["avg_diff"] for r in layer_results]
        colors = ['red' if d > 0 else 'blue' for d in diffs]
        ax.bar(layers_plot, diffs, color=colors, alpha=0.6)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Forget - Retain Projection")
        ax.set_title("Differential Projection (Positive = Forget More Affected)")
        ax.grid(alpha=0.3)

        # Plot 3: Scatter of all measurements
        ax = axes[1, 0]
        forget_all = [r["forget_proj_ratio"] for r in results]
        retain_all = [r["retain_proj_ratio"] for r in results]
        ax.scatter(retain_all, forget_all, alpha=0.6, s=50)
        ax.plot([0, max(retain_all+forget_all)], [0, max(retain_all+forget_all)],
                'k--', alpha=0.3, label="Equal projection")
        ax.set_xlabel("Retain Projection Ratio")
        ax.set_ylabel("Forget Projection Ratio")
        ax.set_title("Forget vs Retain Projections")
        ax.legend()
        ax.grid(alpha=0.3)

        # Plot 4: Summary statistics
        ax = axes[1, 1]
        ax.axis('off')

        avg_forget_proj = np.mean([r["forget_proj_ratio"] for r in results])
        avg_retain_proj = np.mean([r["retain_proj_ratio"] for r in results])
        forget_stronger_count = sum(1 for r in results if r["forget_stronger"])

        summary_text = f"""
        Row Space Projection Summary:

        Average Projections:
        - Forget: {avg_forget_proj:.3f}
        - Retain: {avg_retain_proj:.3f}
        - Ratio: {avg_forget_proj / (avg_retain_proj + 1e-10):.2f}x

        Layer Analysis:
        - Forget stronger in {forget_stronger_count}/{len(results)} cases
        - Max difference at layer {layers_plot[np.argmax(diffs)]}
        - Avg difference: {np.mean(diffs):.3f}

        Interpretation:
        {'✓ Forget more affected by updates' if avg_forget_proj > avg_retain_proj * 1.2 else '✗ Similar affect on both'}
        {'✓ Selective modification' if np.mean(diffs) > 0.05 else '✗ Non-selective updates'}
        """

        ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center')

        plt.suptitle(f"Row Space Projection Analysis\n{args.model_a} → {args.model_b}")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "row_space_projections.png"), dpi=150)
        plt.close()

        # Save summary JSON
        summary = {
            "model_a": args.model_a,
            "model_b": args.model_b,
            "avg_forget_projection": float(avg_forget_proj),
            "avg_retain_projection": float(avg_retain_proj),
            "projection_ratio": float(avg_forget_proj / (avg_retain_proj + 1e-10)),
            "forget_more_affected": bool(avg_forget_proj > avg_retain_proj * 1.2),
            "layers_analyzed": len(layers),
        }

        with open(os.path.join(args.outdir, "row_space_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n[row_space_projection] ✓ Results saved to {args.outdir}")
        print(f"[row_space_projection] Forget projection: {avg_forget_proj:.3f}")
        print(f"[row_space_projection] Retain projection: {avg_retain_proj:.3f}")
        print(f"[row_space_projection] Selectivity: {summary['projection_ratio']:.2f}x more aligned with forget data")


if __name__ == "__main__":
    main()