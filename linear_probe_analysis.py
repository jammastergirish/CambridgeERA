#!/usr/bin/env python
# /// script
# dependencies = [
#   "torch",
#   "transformers",
#   "numpy",
#   "matplotlib",
#   "scikit-learn",
#   "tqdm",
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

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import resolve_device, resolve_dtype, write_csv


# ---- helpers ----------------------------------------------------------------

def get_activations(model, tokenizer, texts, layer_idx, device,
                    max_length=512, batch_size=8):
    """Last-token hidden states at *layer_idx* for a list of texts."""
    activations = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
            hidden = out.hidden_states[layer_idx]       # (B, T, D)
            # Use last non-padding token (causal LMs accumulate context left-to-right)
            attn_mask = inputs["attention_mask"]         # (B, T)
            seq_lens = attn_mask.sum(dim=1) - 1         # index of last real token
            last_tok = hidden[torch.arange(hidden.size(0), device=device), seq_lens]  # (B, D)
            activations.append(last_tok.cpu().float().numpy())
    return np.vstack(activations)


def get_num_layers(model):
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


# ---- main -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Per-layer linear-probe analysis for a single model."
    )
    ap.add_argument("--model", required=True,
                    help="Model name or path (e.g. EleutherAI/deep-ignorance-unfiltered)")
    ap.add_argument("--forget-text", default="data/forget.txt")
    ap.add_argument("--retain-text", default="data/retain.txt")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--max-samples", type=int, default=500,
                    help="Max texts per split (default: 500)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    # Probe hyperparameters
    ap.add_argument("--C", type=float, default=1.0,
                    help="Logistic-regression inverse regularisation strength")
    ap.add_argument("--max-iter", type=int, default=1000,
                    help="Solver max iterations for logistic regression")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    # ---- load data ----------------------------------------------------------
    print(f"[linear_probe] Loading forget/retain texts...")
    with open(args.forget_text) as f:
        forget_texts = [l.strip() for l in f if l.strip()]
    with open(args.retain_text) as f:
        retain_texts = [l.strip() for l in f if l.strip()]

    forget_texts = forget_texts[:args.max_samples]
    retain_texts = retain_texts[:args.max_samples]
    print(f"[linear_probe] {len(forget_texts)} forget, {len(retain_texts)} retain samples")

    # ---- load model ---------------------------------------------------------
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

    # ---- labels & majority baseline -----------------------------------------
    n_forget = len(forget_texts)
    n_retain = len(retain_texts)
    y = np.array([0] * n_forget + [1] * n_retain)   # 0=forget, 1=retain
    majority_baseline = max(n_forget, n_retain) / (n_forget + n_retain)

    # ---- per-layer probing --------------------------------------------------
    results = []
    print(f"[linear_probe] Training probes (C={args.C}, max_iter={args.max_iter})...")

    for layer_idx in tqdm(range(total_layers), desc="Probing layers", unit="layer"):
        # Extract activations
        forget_acts = get_activations(model, tokenizer, forget_texts,
                                      layer_idx, device, args.max_length, args.batch_size)
        retain_acts = get_activations(model, tokenizer, retain_texts,
                                      layer_idx, device, args.max_length, args.batch_size)
        X = np.vstack([forget_acts, retain_acts])

        # Reproducible train/test split (80/20)
        rng = np.random.RandomState(args.seed + layer_idx)
        idx = rng.permutation(len(X))
        split = int(0.8 * len(X))
        X_train, X_test = X[idx[:split]], X[idx[split:]]
        y_train, y_test = y[idx[:split]], y[idx[split:]]

        # Train probe
        clf = LogisticRegression(C=args.C, max_iter=args.max_iter,
                                 solver="lbfgs", random_state=args.seed)
        clf.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, clf.predict(X_train))
        test_acc = accuracy_score(y_test, clf.predict(X_test))
        selectivity = test_acc - majority_baseline

        # AUC (may be undefined if only one class in test split)
        try:
            proba = clf.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, proba)
        except Exception:
            auc = 0.5

        results.append({
            "layer": layer_idx,
            "test_accuracy": round(float(test_acc), 4),
            "train_accuracy": round(float(train_acc), 4),
            "selectivity": round(float(selectivity), 4),
            "auc": round(float(auc), 4),
            "majority_baseline": round(float(majority_baseline), 4),
        })

    # ---- save CSV -----------------------------------------------------------
    fieldnames = ["layer", "test_accuracy", "train_accuracy",
                  "selectivity", "auc", "majority_baseline"]
    write_csv(os.path.join(args.outdir, "probe_results.csv"), results, fieldnames)

    # ---- summary JSON -------------------------------------------------------
    best = max(results, key=lambda r: r["selectivity"])
    summary = {
        "model": args.model,
        "num_layers": total_layers,
        "probe_C": args.C,
        "probe_max_iter": args.max_iter,
        "majority_baseline": majority_baseline,
        "best_layer": best["layer"],
        "best_selectivity": best["selectivity"],
        "best_accuracy": best["test_accuracy"],
        "best_auc": best["auc"],
    }
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- plots --------------------------------------------------------------
    layers = [r["layer"] for r in results]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Accuracy
    ax = axes[0]
    ax.plot(layers, [r["test_accuracy"] for r in results], "o-", label="Test accuracy")
    ax.plot(layers, [r["train_accuracy"] for r in results], "s--", alpha=0.5,
            label="Train accuracy")
    ax.axhline(majority_baseline, color="gray", ls="--", alpha=0.6,
               label=f"Majority baseline ({majority_baseline:.2f})")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.set_title("Probe Accuracy by Layer")
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. Selectivity
    ax = axes[1]
    colors = ["green" if s > 0 else "red" for s in [r["selectivity"] for r in results]]
    ax.bar(layers, [r["selectivity"] for r in results], color=colors, alpha=0.7)
    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Selectivity (acc − baseline)")
    ax.set_title("Probe Selectivity by Layer")
    ax.grid(alpha=0.3)

    # 3. AUC
    ax = axes[2]
    ax.plot(layers, [r["auc"] for r in results], "o-", color="purple")
    ax.axhline(0.5, color="gray", ls="--", alpha=0.5, label="Chance (0.5)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("ROC AUC")
    ax.set_title("Probe AUC by Layer")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle(f"Linear Probe Analysis — {args.model}", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "linear_probe_analysis.png"), dpi=150)
    plt.close()

    print(f"\n[linear_probe] ✓ Results saved to {args.outdir}")
    print(f"[linear_probe] Best selectivity: layer {best['layer']} "
          f"(selectivity={best['selectivity']:.4f}, acc={best['test_accuracy']:.4f})")


if __name__ == "__main__":
    main()
