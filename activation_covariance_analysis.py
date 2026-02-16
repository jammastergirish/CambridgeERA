#!/usr/bin/env python
# /// script
# dependencies = [
#   "torch",
#   "transformers",
#   "numpy",
#   "matplotlib",
#   "tqdm",
#   "scipy",
#   "wandb",
#   "pandas",
# ]
# ///

"""
Analyze activation covariance structure changes between forget and retain datasets.
Examines how singular value spectra change after unlearning.
"""

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from scipy.stats import wasserstein_distance

from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import resolve_device, resolve_dtype, write_csv, init_wandb, log_csv_as_table, log_plots, finish_wandb


def get_activations_batch(model, tokenizer, texts, layer_idx, device, max_length=512, batch_size=8):
    """Extract activations at a specific layer for given texts."""
    all_activations = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True,
                          truncation=True, max_length=max_length).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Get activations from specified layer
            hidden_states = outputs.hidden_states[layer_idx]
            # Flatten batch and sequence dimensions
            batch_acts = hidden_states.reshape(-1, hidden_states.shape[-1])
            all_activations.append(batch_acts.cpu().float().numpy())

    return np.vstack(all_activations)


def compute_covariance_metrics(activations, top_k=50):
    """Compute covariance matrix and analyze its spectrum."""
    # Center the activations
    mean = activations.mean(axis=0)
    centered = activations - mean

    # Compute covariance
    n_samples = centered.shape[0]
    cov = (centered.T @ centered) / (n_samples - 1)

    # Compute eigenvalues (singular values squared)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending

    # Normalize to get explained variance ratio
    total_var = eigenvalues.sum()
    explained_var_ratio = eigenvalues / (total_var + 1e-10)

    # Compute effective rank (number of eigenvalues to explain 99% variance)
    cumsum = np.cumsum(explained_var_ratio)
    effective_rank = np.searchsorted(cumsum, 0.99) + 1

    # Top-k concentration
    top_k_var = explained_var_ratio[:top_k].sum() if len(eigenvalues) >= top_k else explained_var_ratio.sum()

    # Spectral entropy (measure of spread)
    # Normalize eigenvalues to probabilities
    probs = eigenvalues / (eigenvalues.sum() + 1e-10)
    probs = probs[probs > 1e-10]  # Remove zeros for log
    spectral_entropy = -np.sum(probs * np.log(probs))

    return {
        "eigenvalues": eigenvalues[:min(100, len(eigenvalues))],  # Keep top 100
        "explained_var_ratio": explained_var_ratio[:min(100, len(explained_var_ratio))],
        "effective_rank": int(effective_rank),
        "top_k_concentration": float(top_k_var),
        "spectral_entropy": float(spectral_entropy),
        "max_eigenvalue": float(eigenvalues[0]) if len(eigenvalues) > 0 else 0,
        "trace": float(total_var),
    }


def compare_spectra(spectrum_a, spectrum_b):
    """Compare two eigenvalue spectra."""
    eig_a = spectrum_a["eigenvalues"]
    eig_b = spectrum_b["eigenvalues"]

    # Pad to same length
    max_len = max(len(eig_a), len(eig_b))
    eig_a_pad = np.pad(eig_a, (0, max_len - len(eig_a)), constant_values=0)
    eig_b_pad = np.pad(eig_b, (0, max_len - len(eig_b)), constant_values=0)

    # Wasserstein distance between spectra
    w_distance = wasserstein_distance(eig_a_pad, eig_b_pad)

    # Relative change in top eigenvalues
    top_10_change = np.abs(eig_b_pad[:10] - eig_a_pad[:10]).mean() / (eig_a_pad[:10].mean() + 1e-10)

    return {
        "wasserstein_distance": float(w_distance),
        "top_10_relative_change": float(top_10_change),
        "effective_rank_change": spectrum_b["effective_rank"] - spectrum_a["effective_rank"],
        "entropy_change": spectrum_b["spectral_entropy"] - spectrum_a["spectral_entropy"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-a", required=True, help="Baseline model")
    ap.add_argument("--model-b", required=True, help="Unlearned/finetuned model")
    ap.add_argument("--forget-text", default="data/forget.txt")
    ap.add_argument("--retain-text", default="data/retain.txt")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--max-samples", type=int, default=500,
                   help="Max texts per split to process (default: 500)")
    ap.add_argument("--layers-to-analyze", type=str, default=None,
                   help="Comma-separated layer indices to analyze (default: all)")
    ap.add_argument("--outdir", default="outputs/activation_covariance")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    init_wandb("activation_covariance", args)

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    # Load texts
    print("[activation_covariance] Loading forget/retain texts...")
    with open(args.forget_text, "r") as f:
        forget_texts = [line.strip() for line in f if line.strip()]
    with open(args.retain_text, "r") as f:
        retain_texts = [line.strip() for line in f if line.strip()]

    # Cap sample count
    if len(forget_texts) > args.max_samples:
        forget_texts = forget_texts[:args.max_samples]
    if len(retain_texts) > args.max_samples:
        retain_texts = retain_texts[:args.max_samples]

    print(f"[activation_covariance] Using {len(forget_texts)} forget texts, {len(retain_texts)} retain texts (max-samples={args.max_samples})")

    # Load models
    print(f"[activation_covariance] Loading model A (baseline): {args.model_a}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_a)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_a = AutoModelForCausalLM.from_pretrained(args.model_a, torch_dtype=dtype).to(device)
    model_a.eval()

    print(f"[activation_covariance] Loading model B (target): {args.model_b}")
    model_b = AutoModelForCausalLM.from_pretrained(args.model_b, torch_dtype=dtype).to(device)
    model_b.eval()

    # Get layers to analyze
    if hasattr(model_a, 'gpt_neox') and hasattr(model_a.gpt_neox, 'layers'):
        num_layers = len(model_a.gpt_neox.layers)
    elif hasattr(model_a, 'model') and hasattr(model_a.model, 'layers'):
        num_layers = len(model_a.model.layers)
    elif hasattr(model_a, 'transformer') and hasattr(model_a.transformer, 'h'):
        num_layers = len(model_a.transformer.h)
    elif hasattr(model_a, 'encoder') and hasattr(model_a.encoder, 'layer'):
        num_layers = len(model_a.encoder.layer)
    else:
        print("Warning: Could not determine number of layers, defaulting to 32")
        num_layers = 32

    if args.layers_to_analyze:
        layers = [int(x) for x in args.layers_to_analyze.split(",")]
    else:
        # Analyze every 4th layer to reduce computation
        layers = list(range(0, num_layers + 1, 4))

    results = []
    os.makedirs(args.outdir, exist_ok=True)

    # Analyze each layer
    print(f"[activation_covariance] Analyzing covariance spectra across {len(layers)} layers...")
    for layer_idx in tqdm(layers, desc="Analyzing covariance spectra", unit="layer"):
        print(f"\n[activation_covariance] Layer {layer_idx}:")

        # Get activations for both models and datasets
        print("  Extracting forget activations (both models)...")
        forget_acts_a = get_activations_batch(model_a, tokenizer, forget_texts,
                                             layer_idx, device, args.max_length, args.batch_size)
        forget_acts_b = get_activations_batch(model_b, tokenizer, forget_texts,
                                             layer_idx, device, args.max_length, args.batch_size)

        print("  Extracting retain activations (both models)...")
        retain_acts_a = get_activations_batch(model_a, tokenizer, retain_texts,
                                             layer_idx, device, args.max_length, args.batch_size)
        retain_acts_b = get_activations_batch(model_b, tokenizer, retain_texts,
                                             layer_idx, device, args.max_length, args.batch_size)

        # Compute covariance spectra
        print("  Computing & comparing covariance spectra...")
        forget_spectrum_a = compute_covariance_metrics(forget_acts_a)
        forget_spectrum_b = compute_covariance_metrics(forget_acts_b)
        retain_spectrum_a = compute_covariance_metrics(retain_acts_a)
        retain_spectrum_b = compute_covariance_metrics(retain_acts_b)

        # Compare spectra
        forget_comparison = compare_spectra(forget_spectrum_a, forget_spectrum_b)
        retain_comparison = compare_spectra(retain_spectrum_a, retain_spectrum_b)

        # Store results
        result = {
            "layer": layer_idx,
            "forget_eff_rank_a": forget_spectrum_a["effective_rank"],
            "forget_eff_rank_b": forget_spectrum_b["effective_rank"],
            "retain_eff_rank_a": retain_spectrum_a["effective_rank"],
            "retain_eff_rank_b": retain_spectrum_b["effective_rank"],
            "forget_entropy_a": forget_spectrum_a["spectral_entropy"],
            "forget_entropy_b": forget_spectrum_b["spectral_entropy"],
            "retain_entropy_a": retain_spectrum_a["spectral_entropy"],
            "retain_entropy_b": retain_spectrum_b["spectral_entropy"],
            "forget_wasserstein": forget_comparison["wasserstein_distance"],
            "retain_wasserstein": retain_comparison["wasserstein_distance"],
            "forget_top10_change": forget_comparison["top_10_relative_change"],
            "retain_top10_change": retain_comparison["top_10_relative_change"],
        }
        results.append(result)

        # Save detailed spectra for this layer
        np.savez_compressed(
            os.path.join(args.outdir, f"spectra_layer_{layer_idx}.npz"),
            forget_eig_a=forget_spectrum_a["eigenvalues"],
            forget_eig_b=forget_spectrum_b["eigenvalues"],
            retain_eig_a=retain_spectrum_a["eigenvalues"],
            retain_eig_b=retain_spectrum_b["eigenvalues"],
            forget_var_a=forget_spectrum_a["explained_var_ratio"],
            forget_var_b=forget_spectrum_b["explained_var_ratio"],
            retain_var_a=retain_spectrum_a["explained_var_ratio"],
            retain_var_b=retain_spectrum_b["explained_var_ratio"],
        )

    # Save results CSV
    write_csv(
        os.path.join(args.outdir, "covariance_metrics.csv"),
        results,
        ["layer", "forget_eff_rank_a", "forget_eff_rank_b", "retain_eff_rank_a", "retain_eff_rank_b",
         "forget_entropy_a", "forget_entropy_b", "retain_entropy_a", "retain_entropy_b",
         "forget_wasserstein", "retain_wasserstein", "forget_top10_change", "retain_top10_change"]
    )

    # Create visualizations
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))

    layers_plot = [r["layer"] for r in results]

    # Plot 1: Effective rank comparison
    ax = axes[0, 0]
    ax.plot(layers_plot, [r["forget_eff_rank_a"] for r in results], 'o-', label="Forget (A)", color='red', alpha=0.5)
    ax.plot(layers_plot, [r["forget_eff_rank_b"] for r in results], 's-', label="Forget (B)", color='darkred')
    ax.plot(layers_plot, [r["retain_eff_rank_a"] for r in results], 'o-', label="Retain (A)", color='blue', alpha=0.5)
    ax.plot(layers_plot, [r["retain_eff_rank_b"] for r in results], 's-', label="Retain (B)", color='darkblue')
    ax.set_xlabel("Layer")
    ax.set_ylabel("Effective Rank")
    ax.set_title("Effective Rank (99% variance)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 2: Spectral entropy
    ax = axes[0, 1]
    ax.plot(layers_plot, [r["forget_entropy_b"] - r["forget_entropy_a"] for r in results],
            'o-', label="Forget Δ", color='red')
    ax.plot(layers_plot, [r["retain_entropy_b"] - r["retain_entropy_a"] for r in results],
            's-', label="Retain Δ", color='blue')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Δ Spectral Entropy (B - A)")
    ax.set_title("Change in Spectral Entropy")
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 3: Wasserstein distance
    ax = axes[0, 2]
    ax.plot(layers_plot, [r["forget_wasserstein"] for r in results], 'o-', label="Forget", color='red')
    ax.plot(layers_plot, [r["retain_wasserstein"] for r in results], 's-', label="Retain", color='blue')
    ax.set_xlabel("Layer")
    ax.set_ylabel("Wasserstein Distance")
    ax.set_title("Spectrum Change (A → B)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 4: Top-10 eigenvalue changes
    ax = axes[1, 0]
    ax.plot(layers_plot, [r["forget_top10_change"] for r in results], 'o-', label="Forget", color='red')
    ax.plot(layers_plot, [r["retain_top10_change"] for r in results], 's-', label="Retain", color='blue')
    ax.set_xlabel("Layer")
    ax.set_ylabel("Relative Change")
    ax.set_title("Top-10 Eigenvalue Change")
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 5-8: Spectra visualizations for selected layers
    selected_layers = [0, len(layers)//2, -1]  # First, middle, last
    for i, sel_idx in enumerate(selected_layers):
        if sel_idx < 0:
            sel_idx = len(layers) + sel_idx
        if sel_idx >= len(layers):
            continue

        layer = layers[sel_idx]
        ax = axes[1 + i//2, 1 + i%2]

        # Load saved spectra
        spectra_file = os.path.join(args.outdir, f"spectra_layer_{layer}.npz")
        if os.path.exists(spectra_file):
            data = np.load(spectra_file)

            # Plot top 50 eigenvalues
            k = 50
            x = np.arange(1, k+1)

            ax.semilogy(x, data["forget_eig_a"][:k], 'o-', label="Forget (A)",
                       color='red', alpha=0.5, markersize=3)
            ax.semilogy(x, data["forget_eig_b"][:k], 's-', label="Forget (B)",
                       color='darkred', markersize=3)
            ax.semilogy(x, data["retain_eig_a"][:k], 'o-', label="Retain (A)",
                       color='blue', alpha=0.5, markersize=3)
            ax.semilogy(x, data["retain_eig_b"][:k], 's-', label="Retain (B)",
                       color='darkblue', markersize=3)

            ax.set_xlabel("Eigenvalue Index")
            ax.set_ylabel("Eigenvalue (log scale)")
            ax.set_title(f"Layer {layer} Spectrum")
            ax.legend()
            ax.grid(alpha=0.3)

    # Plot 9: Summary statistics
    ax = axes[2, 2]
    ax.axis('off')

    # Compute summary
    avg_forget_wasserstein = np.mean([r["forget_wasserstein"] for r in results])
    avg_retain_wasserstein = np.mean([r["retain_wasserstein"] for r in results])
    avg_forget_rank_change = np.mean([r["forget_eff_rank_b"] - r["forget_eff_rank_a"] for r in results])
    avg_retain_rank_change = np.mean([r["retain_eff_rank_b"] - r["retain_eff_rank_a"] for r in results])

    summary_text = f"""
    Covariance Analysis Summary:

    Spectrum Changes (Wasserstein):
    - Forget: {avg_forget_wasserstein:.3f}
    - Retain: {avg_retain_wasserstein:.3f}
    - Ratio: {avg_forget_wasserstein / (avg_retain_wasserstein + 1e-10):.2f}x

    Effective Rank Changes:
    - Forget: {avg_forget_rank_change:+.1f}
    - Retain: {avg_retain_rank_change:+.1f}

    Interpretation:
    Forget {'more' if avg_forget_wasserstein > avg_retain_wasserstein else 'less'} affected than Retain
    {'✓ Selective modification' if avg_forget_wasserstein > 1.5 * avg_retain_wasserstein else '✗ Non-selective changes'}
    """

    ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center')

    plt.suptitle(f"Activation Covariance Analysis\n{args.model_a} → {args.model_b}")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "covariance_analysis.png"), dpi=150)
    plt.close()

    # Save summary JSON
    summary = {
        "model_a": args.model_a,
        "model_b": args.model_b,
        "avg_forget_wasserstein": float(avg_forget_wasserstein),
        "avg_retain_wasserstein": float(avg_retain_wasserstein),
        "forget_more_affected": bool(avg_forget_wasserstein > avg_retain_wasserstein),
        "selective_ratio": float(avg_forget_wasserstein / (avg_retain_wasserstein + 1e-10)),
    }

    with open(os.path.join(args.outdir, "covariance_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[activation_covariance] ✓ Results saved to {args.outdir}")
    print(f"[activation_covariance] Forget spectrum change (Wasserstein): {avg_forget_wasserstein:.3f}")
    print(f"[activation_covariance] Retain spectrum change (Wasserstein): {avg_retain_wasserstein:.3f}")
    print(f"[activation_covariance] Selectivity ratio: {summary['selective_ratio']:.2f}x")
    log_csv_as_table(os.path.join(args.outdir, "covariance_metrics.csv"), "covariance_metrics")
    log_plots(args.outdir, "activation_covariance")
    finish_wandb()


if __name__ == "__main__":
    main()