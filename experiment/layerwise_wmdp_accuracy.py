#!/usr/bin/env python
# /// script
# dependencies = [
#   "torch",
#   "transformers",
#   "numpy",
#   "matplotlib",
#   "datasets",
#   "tqdm",
#   "wandb",
#   "pandas",
# ]
# ///

"""
Layer-wise WMDP-Bio accuracy via logit lens or tuned lens.

For a single model, evaluates WMDP-Bio multiple-choice accuracy at every
transformer layer.  This reveals *where* hazardous knowledge becomes
accessible — and whether unlearning methods actually erase it or just
hide it from the final output head.

Two modes:
  logit  — project intermediate hidden states through the model's own
           final LayerNorm + unembedding head (lm_head).  Zero training cost.
  tuned  — train a per-layer affine transform (nn.Linear) to map hidden
           states → vocab logits.  More accurate at early layers but
           requires a training pass on held-out WMDP data.
"""

import argparse
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import resolve_device, resolve_dtype, write_csv, init_wandb, log_csv_as_table, log_plots, finish_wandb


# ---- helpers ----------------------------------------------------------------

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


def get_final_ln(model):
    """Find the model's final layer norm (applied before lm_head)."""
    # GPT-NeoX: model.gpt_neox.final_layer_norm
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "final_layer_norm"):
        return model.gpt_neox.final_layer_norm
    # LLaMA / Mistral: model.model.norm
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    # GPT-2: model.transformer.ln_f
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f
    return None


def logit_lens_project(hidden_states, model):
    """Project hidden states to vocab logits using the model's own head."""
    ln = get_final_ln(model)
    if ln is not None:
        hidden_states = ln(hidden_states)
    return model.lm_head(hidden_states)


def load_wmdp_bio(max_samples=None):
    """Load WMDP-Bio multiple-choice dataset from HuggingFace.

    Returns list of dicts with keys: question, choices, answer (int index).
    """
    ds = load_dataset("cais/wmdp", "wmdp-bio", split="test")
    items = []
    for ex in ds:
        q = ex.get("question", ex.get("prompt", ""))
        choices = ex.get("choices", [])
        answer = ex.get("answer", 0)
        if q and choices:
            items.append({"question": q, "choices": choices, "answer": int(answer)})
    if max_samples:
        items = items[:max_samples]
    return items


def score_mcq_at_layer(model, tokenizer, items, layer_idx, device,
                       max_length=512, project_fn=None):
    """Evaluate MCQ accuracy at a specific layer.

    For each question, appends each answer choice and computes the average
    per-token log-prob of the choice continuation.  Picks the highest.

    Args:
        project_fn: callable(hidden_states) → logits.  If None, uses
            the model's own output logits (final layer only).

    Returns:
        accuracy (float), num_correct (int), total (int)
    """
    correct = 0
    total = 0

    for item in items:
        q = item["question"]
        choices = item["choices"]
        answer_idx = item["answer"]

        choice_scores = []
        for choice in choices:
            # Format: "Question: ... Answer: <choice>"
            text = f"{q} {choice}"
            enc = tokenizer(text, return_tensors="pt", max_length=max_length,
                            truncation=True).to(device)
            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]

            with torch.no_grad():
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                output_hidden_states=True)

                if project_fn is not None:
                    hidden = outputs.hidden_states[layer_idx]  # (1, T, D)
                    logits = project_fn(hidden)
                else:
                    logits = outputs.logits  # final layer

            # Compute log-prob of the choice tokens
            # Tokenize just the choice to find how many tokens it adds
            choice_enc = tokenizer(f" {choice}", add_special_tokens=False)
            choice_len = len(choice_enc["input_ids"])
            if choice_len == 0:
                choice_scores.append(float("-inf"))
                continue

            # Score: average log-prob over the last `choice_len` tokens
            seq_len = input_ids.size(1)
            start = max(seq_len - choice_len - 1, 0)
            end = seq_len - 1  # logits are shifted by 1

            log_probs = F.log_softmax(logits[0, start:end, :], dim=-1)
            target_ids = input_ids[0, start + 1:end + 1]
            token_lps = log_probs[torch.arange(log_probs.size(0)), target_ids]
            avg_lp = token_lps.mean().item()
            choice_scores.append(avg_lp)

        if choice_scores:
            predicted = int(np.argmax(choice_scores))
            if predicted == answer_idx:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total


# ---- tuned lens training ----------------------------------------------------

class TunedLensProbe(torch.nn.Module):
    """Per-layer affine transform: hidden_dim → vocab_size."""
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, hidden_states):
        return self.linear(hidden_states)


def train_tuned_lens(model, tokenizer, train_texts, layer_idx, device,
                     hidden_dim, vocab_size, max_length=512, batch_size=4,
                     lr=1e-3, epochs=3):
    """Train a tuned lens probe for a single layer.

    Uses causal LM loss: the probe must predict the next token from the
    intermediate hidden state at the given layer.
    """
    probe = TunedLensProbe(hidden_dim, vocab_size).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    probe.train()
    for _epoch in range(epochs):
        np.random.shuffle(train_texts)
        for i in range(0, len(train_texts), batch_size):
            batch = train_texts[i:i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", max_length=max_length,
                            truncation=True, padding=True).to(device)
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
                hidden = out.hidden_states[layer_idx]  # (B, T, D)

            logits = probe(hidden)[:, :-1, :].contiguous()
            labels = enc["input_ids"][:, 1:].contiguous()
            mask = enc["attention_mask"][:, 1:].contiguous().float()

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), reduction="none"
            )
            loss = (loss * mask.view(-1)).sum() / mask.sum().clamp(min=1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    probe.eval()
    return probe


# ---- main -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Layer-wise WMDP-Bio accuracy via logit/tuned lens."
    )
    ap.add_argument("--model", required=True,
                    help="Model name or path")
    ap.add_argument("--lens", choices=["logit", "tuned"], default="logit",
                    help="Lens type: logit (default) or tuned")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--max-samples", type=int, default=500,
                    help="Max WMDP questions to evaluate (default: 500)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    # Tuned lens hyperparams
    ap.add_argument("--tuned-lr", type=float, default=1e-3,
                    help="Learning rate for tuned lens probes")
    ap.add_argument("--tuned-epochs", type=int, default=3,
                    help="Training epochs for tuned lens probes")
    ap.add_argument("--tuned-train-frac", type=float, default=0.3,
                    help="Fraction of WMDP data used to train tuned lens (rest for eval)")
    args = ap.parse_args()
    init_wandb("wmdp_lens", args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    # ---- load data ----------------------------------------------------------
    print(f"[wmdp_lens] Loading WMDP-Bio dataset...")
    all_items = load_wmdp_bio(max_samples=args.max_samples)
    print(f"[wmdp_lens] {len(all_items)} MCQ items loaded")

    # Split for tuned lens training if needed
    if args.lens == "tuned":
        np.random.shuffle(all_items)
        n_train = int(len(all_items) * args.tuned_train_frac)
        train_items = all_items[:n_train]
        eval_items = all_items[n_train:]
        # Build training texts from train MCQ items (question + correct answer)
        train_texts = [f"{it['question']} {it['choices'][it['answer']]}"
                       for it in train_items]
        print(f"[wmdp_lens] Tuned lens: {n_train} train, {len(eval_items)} eval")
    else:
        eval_items = all_items

    # ---- load model ---------------------------------------------------------
    print(f"[wmdp_lens] Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()

    num_layers = get_num_layers(model)
    total_layers = num_layers + 1  # +1 for embedding layer
    hidden_dim = model.config.hidden_size
    vocab_size = model.config.vocab_size
    print(f"[wmdp_lens] {total_layers} layers, hidden_dim={hidden_dim}, vocab_size={vocab_size}")

    os.makedirs(args.outdir, exist_ok=True)

    # ---- evaluate per layer -------------------------------------------------
    results = []
    print(f"[wmdp_lens] Evaluating {args.lens} lens across {total_layers} layers...")

    for layer_idx in tqdm(range(total_layers), desc=f"{args.lens} lens", unit="layer"):
        if args.lens == "logit":
            project_fn = lambda h, m=model: logit_lens_project(h, m)
        else:
            # Train a tuned lens probe for this layer
            print(f"\n  Training tuned lens for layer {layer_idx}...")
            probe = train_tuned_lens(
                model, tokenizer, train_texts, layer_idx, device,
                hidden_dim, vocab_size, args.max_length, args.batch_size,
                args.tuned_lr, args.tuned_epochs,
            )
            project_fn = lambda h, p=probe: p(h)

        acc, n_correct, n_total = score_mcq_at_layer(
            model, tokenizer, eval_items, layer_idx, device,
            args.max_length, project_fn=project_fn,
        )

        results.append({
            "layer": layer_idx,
            "accuracy": round(float(acc), 4),
            "correct": n_correct,
            "total": n_total,
        })

    # ---- final layer reference (using model's own logits, no lens) ----------
    final_acc, final_correct, final_total = score_mcq_at_layer(
        model, tokenizer, eval_items, -1, device,
        args.max_length, project_fn=None,
    )
    print(f"\n[wmdp_lens] Final-layer accuracy (native): {final_acc:.4f} ({final_correct}/{final_total})")

    # ---- save CSV -----------------------------------------------------------
    fieldnames = ["layer", "accuracy", "correct", "total"]
    write_csv(os.path.join(args.outdir, "wmdp_lens_results.csv"), results, fieldnames)

    # ---- summary JSON -------------------------------------------------------
    best = max(results, key=lambda r: r["accuracy"])
    summary = {
        "model": args.model,
        "lens": args.lens,
        "num_layers": total_layers,
        "max_samples": args.max_samples,
        "final_layer_accuracy": final_acc,
        "best_layer": best["layer"],
        "best_layer_accuracy": best["accuracy"],
    }
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- plots --------------------------------------------------------------
    layers = [r["layer"] for r in results]
    accs = [r["accuracy"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Absolute accuracy by layer
    ax = axes[0]
    ax.plot(layers, accs, "o-", color="tab:blue", label=f"{args.lens} lens")
    ax.axhline(final_acc, color="tab:orange", ls="--", alpha=0.7,
               label=f"Final layer ({final_acc:.3f})")
    ax.axhline(0.25, color="gray", ls=":", alpha=0.5,
               label="Random chance (0.25)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("WMDP-Bio Accuracy")
    ax.set_title("WMDP Accuracy by Layer")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 2. Delta from final layer
    ax = axes[1]
    deltas = [a - final_acc for a in accs]
    colors = ["green" if d >= 0 else "red" for d in deltas]
    ax.bar(layers, deltas, color=colors, alpha=0.6)
    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Δ Accuracy (layer − final)")
    ax.set_title("Accuracy Delta from Final Layer")
    ax.grid(alpha=0.3)

    plt.suptitle(f"Layer-wise WMDP-Bio Accuracy — {args.model} ({args.lens} lens)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "wmdp_lens_analysis.png"), dpi=150)
    plt.close()

    print(f"\n[wmdp_lens] ✓ Results saved to {args.outdir}")
    print(f"[wmdp_lens] Best layer: {best['layer']} (accuracy={best['accuracy']:.4f})")
    log_csv_as_table(os.path.join(args.outdir, "wmdp_lens_results.csv"), "wmdp_lens_results")
    log_plots(args.outdir, "wmdp_lens")
    finish_wandb()


if __name__ == "__main__":
    main()
