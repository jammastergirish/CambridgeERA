#!/usr/bin/env python
# /// script
# dependencies = [
#   "torch",
#   "transformers",
#   "numpy",
#   "matplotlib",
#   "scikit-learn",
#   "tqdm",
#   "wandb",
#   "pandas",
# ]
# ///

"""
Analyze activation separation between forget and retain datasets.
Measures how well-separated the internal representations are after unlearning.
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import json

from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import resolve_device, resolve_dtype, write_csv, init_wandb, log_csv_as_table, log_plots, finish_wandb


def get_activations(model, tokenizer, texts, layer_idx, device, max_length=512, batch_size=8):
    """Extract activations at a specific layer for given texts."""
    activations = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True,
                          truncation=True, max_length=max_length).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Get activations from specified layer
            hidden_states = outputs.hidden_states[layer_idx]
            # Average pool over sequence length
            pooled = hidden_states.mean(dim=1)
            activations.append(pooled.cpu().float().numpy())

    return np.vstack(activations)


def compute_separation_metrics(forget_acts, retain_acts):
    """Compute various separation metrics between two activation sets."""

    # Centroid distance (cosine)
    forget_centroid = forget_acts.mean(axis=0)
    retain_centroid = retain_acts.mean(axis=0)

    # Normalize for cosine distance
    forget_norm = forget_centroid / (np.linalg.norm(forget_centroid) + 1e-8)
    retain_norm = retain_centroid / (np.linalg.norm(retain_centroid) + 1e-8)
    cosine_distance = 1 - np.dot(forget_norm, retain_norm)

    # Euclidean distance between centroids
    euclidean_distance = np.linalg.norm(forget_centroid - retain_centroid)

    # Linear discriminability (using LDA)
    X = np.vstack([forget_acts, retain_acts])
    y = np.array([0] * len(forget_acts) + [1] * len(retain_acts))

    # Shuffle for train/test split
    indices = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    X_train, X_test = X[indices[:split]], X[indices[split:]]
    y_train, y_test = y[indices[:split]], y[indices[split:]]

    try:
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train, y_train)
        y_pred_proba = lda.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
    except:
        auc_score = 0.5  # Random baseline if LDA fails

    # Within-cluster vs between-cluster variance ratio
    forget_var = np.var(forget_acts, axis=0).mean()
    retain_var = np.var(retain_acts, axis=0).mean()
    within_cluster_var = (forget_var + retain_var) / 2

    between_cluster_var = np.var(np.vstack([forget_centroid, retain_centroid]), axis=0).mean()
    variance_ratio = between_cluster_var / (within_cluster_var + 1e-8)

    return {
        "cosine_distance": float(cosine_distance),
        "euclidean_distance": float(euclidean_distance),
        "linear_discriminability_auc": float(auc_score),
        "variance_ratio": float(variance_ratio),
        "forget_centroid_norm": float(np.linalg.norm(forget_centroid)),
        "retain_centroid_norm": float(np.linalg.norm(retain_centroid)),
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
    ap.add_argument("--outdir", default="outputs/activation_separation")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    init_wandb("activation_separation", args)

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    # Load texts
    print("[activation_separation] Loading forget/retain texts...")
    with open(args.forget_text, "r") as f:
        forget_texts = [line.strip() for line in f if line.strip()][:500]
    with open(args.retain_text, "r") as f:
        retain_texts = [line.strip() for line in f if line.strip()][:500]

    # Cap sample count
    if len(forget_texts) > args.max_samples:
        forget_texts = forget_texts[:args.max_samples]
    if len(retain_texts) > args.max_samples:
        retain_texts = retain_texts[:args.max_samples]

    print(f"[activation_separation] Loaded {len(forget_texts)} forget texts, {len(retain_texts)} retain texts (max-samples={args.max_samples})")

    # Load models
    print(f"[activation_separation] Loading model A (baseline): {args.model_a}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_a)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_a = AutoModelForCausalLM.from_pretrained(args.model_a, torch_dtype=dtype).to(device)
    model_a.eval()

    print(f"[activation_separation] Loading model B (target): {args.model_b}")
    model_b = AutoModelForCausalLM.from_pretrained(args.model_b, torch_dtype=dtype).to(device)
    model_b.eval()

    # Get number of layers
    if hasattr(model_a, 'gpt_neox') and hasattr(model_a.gpt_neox, 'layers'):
        # GPT-NeoX style
        num_layers = len(model_a.gpt_neox.layers)
    elif hasattr(model_a, 'model') and hasattr(model_a.model, 'layers'):
        # LLaMA style
        num_layers = len(model_a.model.layers)
    elif hasattr(model_a, 'transformer') and hasattr(model_a.transformer, 'h'):
        # GPT-2 style
        num_layers = len(model_a.transformer.h)
    elif hasattr(model_a, 'encoder') and hasattr(model_a.encoder, 'layer'):
        # BERT style
        num_layers = len(model_a.encoder.layer)
    else:
        raise ValueError(f"Could not determine number of layers for model architecture")

    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)

    results_a = []
    results_b = []

    # Analyze each layer
    print(f"[activation_separation] Analyzing {num_layers + 1} layers (including embedding layer)...")
    for layer_idx in tqdm(range(num_layers + 1), desc="Analyzing layer separation", unit="layer"):  # +1 for embeddings
        print(f"  Layer {layer_idx}/{num_layers}: Extracting forget & retain activations for both models...", flush=True)
        # Get activations for model A
        forget_acts_a = get_activations(model_a, tokenizer, forget_texts,
                                        layer_idx, device, args.max_length, args.batch_size)
        retain_acts_a = get_activations(model_a, tokenizer, retain_texts,
                                        layer_idx, device, args.max_length, args.batch_size)

        # Get activations for model B
        forget_acts_b = get_activations(model_b, tokenizer, forget_texts,
                                        layer_idx, device, args.max_length, args.batch_size)
        retain_acts_b = get_activations(model_b, tokenizer, retain_texts,
                                        layer_idx, device, args.max_length, args.batch_size)

        # Compute separation metrics
        metrics_a = compute_separation_metrics(forget_acts_a, retain_acts_a)
        metrics_b = compute_separation_metrics(forget_acts_b, retain_acts_b)

        metrics_a["layer"] = layer_idx
        metrics_b["layer"] = layer_idx

        results_a.append(metrics_a)
        results_b.append(metrics_b)

        # Save intermediate activations for detailed analysis
        if layer_idx % 4 == 0:  # Save every 4th layer to manage disk space
            np.savez_compressed(
                os.path.join(args.outdir, f"activations_layer_{layer_idx}.npz"),
                forget_a=forget_acts_a,
                retain_a=retain_acts_a,
                forget_b=forget_acts_b,
                retain_b=retain_acts_b,
            )

    # Save results
    write_csv(
        os.path.join(args.outdir, "separation_metrics_model_a.csv"),
        results_a,
        ["layer", "cosine_distance", "euclidean_distance", "linear_discriminability_auc",
         "variance_ratio", "forget_centroid_norm", "retain_centroid_norm"]
    )

    write_csv(
        os.path.join(args.outdir, "separation_metrics_model_b.csv"),
        results_b,
        ["layer", "cosine_distance", "euclidean_distance", "linear_discriminability_auc",
         "variance_ratio", "forget_centroid_norm", "retain_centroid_norm"]
    )

    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    layers = [r["layer"] for r in results_a]

    # Plot 1: Cosine distance
    ax = axes[0, 0]
    ax.plot(layers, [r["cosine_distance"] for r in results_a], 'o-', label="Model A (baseline)")
    ax.plot(layers, [r["cosine_distance"] for r in results_b], 's-', label="Model B (unlearned)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Distance")
    ax.set_title("Forget/Retain Centroid Separation (Cosine)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 2: Linear discriminability
    ax = axes[0, 1]
    ax.plot(layers, [r["linear_discriminability_auc"] for r in results_a], 'o-', label="Model A")
    ax.plot(layers, [r["linear_discriminability_auc"] for r in results_b], 's-', label="Model B")
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUC Score")
    ax.set_title("Linear Discriminability (Forget vs Retain)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 3: Variance ratio
    ax = axes[0, 2]
    ax.plot(layers, [r["variance_ratio"] for r in results_a], 'o-', label="Model A")
    ax.plot(layers, [r["variance_ratio"] for r in results_b], 's-', label="Model B")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Between/Within Variance Ratio")
    ax.set_title("Cluster Separation (Variance Ratio)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 4: Change in cosine distance
    ax = axes[1, 0]
    delta_cosine = [results_b[i]["cosine_distance"] - results_a[i]["cosine_distance"]
                    for i in range(len(results_a))]
    ax.plot(layers, delta_cosine, 'o-', color='green')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Δ Cosine Distance (B - A)")
    ax.set_title("Change in Separation After Unlearning")
    ax.grid(alpha=0.3)

    # Plot 5: Centroid norms
    ax = axes[1, 1]
    ax.plot(layers, [r["forget_centroid_norm"] for r in results_a], 'o-',
            label="Forget (A)", color='red', alpha=0.5)
    ax.plot(layers, [r["retain_centroid_norm"] for r in results_a], 'o-',
            label="Retain (A)", color='blue', alpha=0.5)
    ax.plot(layers, [r["forget_centroid_norm"] for r in results_b], 's-',
            label="Forget (B)", color='darkred')
    ax.plot(layers, [r["retain_centroid_norm"] for r in results_b], 's-',
            label="Retain (B)", color='darkblue')
    ax.set_xlabel("Layer")
    ax.set_ylabel("Centroid L2 Norm")
    ax.set_title("Activation Magnitudes")
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 6: Summary statistics
    ax = axes[1, 2]
    ax.axis('off')

    # Compute average changes
    avg_cosine_change = np.mean(delta_cosine)
    avg_auc_a = np.mean([r["linear_discriminability_auc"] for r in results_a])
    avg_auc_b = np.mean([r["linear_discriminability_auc"] for r in results_b])

    summary_text = f"""
    Summary Statistics:

    Model A (Baseline):
    - Avg Linear Discriminability: {avg_auc_a:.3f}
    - Avg Cosine Distance: {np.mean([r["cosine_distance"] for r in results_a]):.3f}

    Model B (Unlearned):
    - Avg Linear Discriminability: {avg_auc_b:.3f}
    - Avg Cosine Distance: {np.mean([r["cosine_distance"] for r in results_b]):.3f}

    Change (B - A):
    - Avg Δ Cosine Distance: {avg_cosine_change:.3f}
    - Avg Δ AUC: {avg_auc_b - avg_auc_a:.3f}

    Interpretation:
    {'✓ Increased separation' if avg_cosine_change > 0.05 else '✗ Minimal separation change'}
    {'✓ More discriminable' if avg_auc_b > avg_auc_a + 0.05 else '✗ Similar discriminability'}
    """

    ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center')

    plt.suptitle(f"Activation Separation Analysis\n{args.model_a} → {args.model_b}")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "activation_separation_analysis.png"), dpi=150)
    plt.close()

    # Save summary JSON
    summary = {
        "model_a": args.model_a,
        "model_b": args.model_b,
        "avg_cosine_change": float(avg_cosine_change),
        "avg_auc_change": float(avg_auc_b - avg_auc_a),
        "max_separation_layer": int(layers[np.argmax(delta_cosine)]),
        "max_separation_value": float(max(delta_cosine)),
    }

    with open(os.path.join(args.outdir, "separation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[activation_separation] ✓ Results saved to {args.outdir}")
    print(f"[activation_separation] Average separation change (cosine): {avg_cosine_change:.3f}")
    print(f"[activation_separation] Maximum separation at layer {summary['max_separation_layer']}: {summary['max_separation_value']:.3f}")
    log_csv_as_table(os.path.join(args.outdir, "separation_metrics.csv"), "separation_metrics")
    log_plots(args.outdir, "activation_separation")
    finish_wandb()


if __name__ == "__main__":
    main()