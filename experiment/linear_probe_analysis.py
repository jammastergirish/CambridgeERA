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
Linear probe analysis: locate where forget-set knowledge is encoded.

For a single model, trains per-layer logistic-regression probes that
classify forget vs retain activations.  Selectivity = probe accuracy
minus majority-class baseline, measuring how strongly the forget/retain
distinction is linearly readable at each layer.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import (
    model_outdir,
    resolve_device,
    resolve_dtype,
    write_csv,
    init_wandb,
    log_csv_as_table,
    log_plots,
    finish_wandb,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_activations(
    model,
    tokenizer,
    texts: List[str],
    layer_index: int,
    device: str,
    max_length: int = 512,
    batch_size: int = 8,
) -> np.ndarray:
    """Last-token hidden states at *layer_index* for a list of texts."""
    all_activations: List[np.ndarray] = []
    for batch_start in range(0, len(texts), batch_size):
        batch_texts = texts[batch_start : batch_start + batch_size]
        inputs = tokenizer(
            batch_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[layer_index]        # (B, T, D)
            attention_mask = inputs["attention_mask"]           # (B, T)
            sequence_lengths = attention_mask.sum(dim=1) - 1    # index of last real token
            last_token = hidden[
                torch.arange(hidden.size(0), device=device), sequence_lengths
            ]  # (B, D)
            all_activations.append(last_token.cpu().float().numpy())
    return np.vstack(all_activations)


def get_num_layers(model) -> int:
    """Auto-detect transformer layer count across common architectures."""
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return len(model.gpt_neox.layers)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return len(model.transformer.h)
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        return len(model.encoder.layer)
    raise ValueError("Could not determine number of layers for model architecture")


# ---------------------------------------------------------------------------
# Core probe training (testable, no I/O)
# ---------------------------------------------------------------------------

_PROBE_FIELDNAMES = [
    "layer",
    "test_accuracy",
    "train_accuracy",
    "selectivity",
    "auc",
    "majority_baseline",
]


def train_probe(
    forget_features: np.ndarray,
    retain_features: np.ndarray,
    regularisation_strength: float = 1.0,
    max_iterations: int = 1000,
    seed: int = 42,
    layer_index: int = 0,
) -> Dict[str, float]:
    """Train a logistic-regression probe to classify forget vs retain.

    Args:
        forget_features: (N_forget, D) activation array.
        retain_features: (N_retain, D) activation array.
        regularisation_strength: Inverse regularisation (sklearn *C*).
        max_iterations: Solver iteration limit.
        seed: Random state for reproducibility.
        layer_index: Added to the seed for the train/test split.

    Returns a dict with test_accuracy, train_accuracy, selectivity, auc,
    and majority_baseline.
    """
    num_forget = len(forget_features)
    num_retain = len(retain_features)
    labels = np.array([0] * num_forget + [1] * num_retain)  # 0=forget, 1=retain
    majority_baseline = max(num_forget, num_retain) / (num_forget + num_retain)

    combined_features = np.vstack([forget_features, retain_features])

    # Reproducible train/test split (80/20)
    rng = np.random.RandomState(seed + layer_index)
    shuffled_indices = rng.permutation(len(combined_features))
    split_point = int(0.8 * len(combined_features))

    train_features = combined_features[shuffled_indices[:split_point]]
    test_features = combined_features[shuffled_indices[split_point:]]
    train_labels = labels[shuffled_indices[:split_point]]
    test_labels = labels[shuffled_indices[split_point:]]

    # Train probe
    probe = LogisticRegression(
        C=regularisation_strength, max_iter=max_iterations,
        solver="lbfgs", random_state=seed,
    )
    probe.fit(train_features, train_labels)

    train_accuracy = float(accuracy_score(train_labels, probe.predict(train_features)))
    test_accuracy = float(accuracy_score(test_labels, probe.predict(test_features)))
    selectivity = test_accuracy - majority_baseline

    # AUC (may be undefined if only one class in test split)
    try:
        predicted_probabilities = probe.predict_proba(test_features)[:, 1]
        auc_score = float(roc_auc_score(test_labels, predicted_probabilities))
    except Exception:
        auc_score = 0.5

    return {
        "test_accuracy": round(test_accuracy, 4),
        "train_accuracy": round(train_accuracy, 4),
        "selectivity": round(float(selectivity), 4),
        "auc": round(auc_score, 4),
        "majority_baseline": round(float(majority_baseline), 4),
    }


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------

def plot_probe_results(
    results: List[Dict],
    outdir: str,
    title: Optional[str] = None,
) -> None:
    """Create the 1×2 panel of probe accuracy/selectivity and AUC plots."""
    layers = [r["layer"] for r in results]
    majority_baseline = results[0]["majority_baseline"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Accuracy + Selectivity (twin y-axis)
    axis = axes[0]
    axis.plot(
        layers, [r["test_accuracy"] for r in results], "o-",
        color="tab:blue", label="Test accuracy",
    )
    axis.plot(
        layers, [r["train_accuracy"] for r in results], "s--",
        alpha=0.5, color="tab:cyan", label="Train accuracy",
    )
    axis.axhline(
        majority_baseline, color="gray", ls="--", alpha=0.6,
        label=f"Majority baseline ({majority_baseline:.2f})",
    )
    axis.set_xlabel("Layer")
    axis.set_ylabel("Accuracy")
    axis.grid(alpha=0.3)

    selectivity_axis = axis.twinx()
    selectivity_values = [r["selectivity"] for r in results]
    bar_colors = ["green" if s > 0 else "red" for s in selectivity_values]
    selectivity_axis.bar(layers, selectivity_values, color=bar_colors, alpha=0.25, label="Selectivity")
    selectivity_axis.axhline(0, color="gray", ls=":", alpha=0.4)
    selectivity_axis.set_ylabel("Selectivity (acc − baseline)")

    # Combined legend from both axes
    handles_main, labels_main = axis.get_legend_handles_labels()
    handles_sel, labels_sel = selectivity_axis.get_legend_handles_labels()
    axis.legend(handles_main + handles_sel, labels_main + labels_sel, loc="best", fontsize=8)
    axis.set_title("Probe Accuracy & Selectivity by Layer")

    # 2. AUC
    axis = axes[1]
    axis.plot(layers, [r["auc"] for r in results], "o-", color="purple")
    axis.axhline(0.5, color="gray", ls="--", alpha=0.5, label="Chance (0.5)")
    axis.set_xlabel("Layer")
    axis.set_ylabel("ROC AUC")
    axis.set_title("Probe AUC by Layer")
    axis.legend()
    axis.grid(alpha=0.3)

    plt.suptitle(title or "Linear Probe Analysis", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "linear_probe_analysis.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Per-layer linear-probe analysis for a single model.",
    )
    parser.add_argument("--model", required=True,
                        help="Model name or path (e.g. EleutherAI/deep-ignorance-unfiltered)")
    parser.add_argument("--forget-text", default="data/forget.txt")
    parser.add_argument("--retain-text", default="data/retain.txt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=500,
                        help="Max texts per split (default: 500)")
    parser.add_argument("--outdir", default=None,
                        help="Output dir (default: outputs/<model>/linear_probes)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--C", type=float, default=1.0,
                        help="Logistic-regression inverse regularisation strength")
    parser.add_argument("--max-iter", type=int, default=1000,
                        help="Solver max iterations for logistic regression")
    parser.add_argument("--title", default=None, help="Title for plots")
    args = parser.parse_args()

    if args.outdir is None:
        args.outdir = model_outdir(args.model, suffix="linear_probes")

    init_wandb("linear_probe", args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    # Load data
    print("[linear_probe] Loading forget/retain texts...")
    with open(args.forget_text) as fh:
        forget_texts = [line.strip() for line in fh if line.strip()][:args.max_samples]
    with open(args.retain_text) as fh:
        retain_texts = [line.strip() for line in fh if line.strip()][:args.max_samples]
    print(f"[linear_probe] {len(forget_texts)} forget, {len(retain_texts)} retain samples")

    # Load model
    print(f"[linear_probe] Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()

    num_layers = get_num_layers(model)
    total_layers = num_layers + 1  # +1 for embedding layer (hidden_states[0])
    print(f"[linear_probe] {total_layers} layers (incl. embedding layer)")

    os.makedirs(args.outdir, exist_ok=True)

    # Per-layer probing
    results: List[Dict] = []
    print(f"[linear_probe] Training probes (C={args.C}, max_iter={args.max_iter})...")

    for layer_index in tqdm(range(total_layers), desc="Probing layers", unit="layer"):
        forget_activations = get_activations(
            model, tokenizer, forget_texts, layer_index, device, args.max_length, args.batch_size,
        )
        retain_activations = get_activations(
            model, tokenizer, retain_texts, layer_index, device, args.max_length, args.batch_size,
        )

        metrics = train_probe(
            forget_activations, retain_activations,
            regularisation_strength=args.C,
            max_iterations=args.max_iter,
            seed=args.seed,
            layer_index=layer_index,
        )
        metrics["layer"] = layer_index
        results.append(metrics)

    # Save CSV
    write_csv(os.path.join(args.outdir, "probe_results.csv"), results, _PROBE_FIELDNAMES)

    # Summary JSON
    best = max(results, key=lambda r: r["selectivity"])
    summary = {
        "model": args.model,
        "num_layers": total_layers,
        "probe_C": args.C,
        "probe_max_iter": args.max_iter,
        "majority_baseline": best["majority_baseline"],
        "best_layer": best["layer"],
        "best_selectivity": best["selectivity"],
        "best_accuracy": best["test_accuracy"],
        "best_auc": best["auc"],
    }
    with open(os.path.join(args.outdir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    # Plots
    plot_probe_results(results, args.outdir, title=args.title)

    print(f"\n[linear_probe] ✓ Results saved to {args.outdir}")
    print(f"[linear_probe] Best selectivity: layer {best['layer']} "
          f"(selectivity={best['selectivity']:.4f}, acc={best['test_accuracy']:.4f})")
    log_csv_as_table(os.path.join(args.outdir, "probe_results.csv"), "probe_results")
    log_plots(args.outdir, "linear_probe")
    finish_wandb()


if __name__ == "__main__":
    main()
