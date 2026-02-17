#!/usr/bin/env python
# /// script
# requires-python = ">=3.11"
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
MMLU evaluation — general-capabilities benchmark.

Evaluates a model's multiple-choice accuracy on MMLU subjects.
Used as Step 0 in the pipeline to quickly identify models that have
catastrophically collapsed during unlearning.

Usage:
  uv run experiment/eval_mmlu.py \
    --model EleutherAI/deep-ignorance-unfiltered \
    --outdir outputs/EleutherAI_deep-ignorance-unfiltered/mmlu

  # Quick sanity check (20 questions)
  uv run experiment/eval_mmlu.py \
    --model EleutherAI/deep-ignorance-unfiltered \
    --max-samples 20 --outdir /tmp/mmlu_test
"""

import argparse
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import resolve_device, resolve_dtype, write_csv, init_wandb, log_csv_as_table, log_plots, finish_wandb


# ---- data -------------------------------------------------------------------

def load_mmlu(max_samples=None, seed=42):
    """Load MMLU multiple-choice dataset from HuggingFace.

    Returns list of dicts with keys: question, choices, answer (int index), subject.
    Samples uniformly across all subjects.
    """
    ds = load_dataset("cais/mmlu", "all", split="test")
    items = []
    for ex in ds:
        q = ex.get("question", "")
        choices = ex.get("choices", [])
        answer = ex.get("answer", 0)
        subject = ex.get("subject", "unknown")
        if q and choices:
            items.append({
                "question": q,
                "choices": choices,
                "answer": int(answer),
                "subject": subject,
            })

    # Shuffle and subsample
    rng = np.random.RandomState(seed)
    rng.shuffle(items)
    if max_samples and max_samples < len(items):
        items = items[:max_samples]
    return items


# ---- scoring ----------------------------------------------------------------

@torch.no_grad()
def score_mcq(model, tokenizer, items, device, max_length=512):
    """Evaluate MCQ accuracy at the final layer.

    For each question, formats as 'Question: ... Answer: <choice>' for each
    choice, computes average per-token log-prob of the choice continuation,
    and picks the highest.

    Returns:
        list of per-item dicts with keys: subject, correct (bool), predicted, answer
    """
    results = []

    for item in tqdm(items, desc="Scoring MMLU", unit="q"):
        q = item["question"]
        choices = item["choices"]
        answer_idx = item["answer"]
        subject = item["subject"]

        choice_scores = []
        for choice in choices:
            text = f"{q} {choice}"
            enc = tokenizer(text, return_tensors="pt", max_length=max_length,
                            truncation=True).to(device)
            input_ids = enc["input_ids"]

            outputs = model(input_ids=input_ids)
            logits = outputs.logits

            # Tokenize just the choice to find how many tokens it adds
            choice_enc = tokenizer(f" {choice}", add_special_tokens=False)
            choice_len = len(choice_enc["input_ids"])
            if choice_len == 0:
                choice_scores.append(float("-inf"))
                continue

            # Average log-prob over the choice tokens
            seq_len = input_ids.size(1)
            start = max(seq_len - choice_len - 1, 0)
            end = seq_len - 1  # logits shifted by 1

            log_probs = F.log_softmax(logits[0, start:end, :], dim=-1)
            target_ids = input_ids[0, start + 1:end + 1]
            token_lps = log_probs[torch.arange(log_probs.size(0)), target_ids]
            avg_lp = token_lps.mean().item()
            choice_scores.append(avg_lp)

        if choice_scores:
            predicted = int(np.argmax(choice_scores))
            results.append({
                "subject": subject,
                "correct": predicted == answer_idx,
                "predicted": predicted,
                "answer": answer_idx,
            })

    return results


# ---- main -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="MMLU evaluation for general capabilities.")
    ap.add_argument("--model", required=True, help="HuggingFace model ID")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--max-samples", type=int, default=1000,
                    help="Max questions to evaluate (default: 1000, sampled across subjects)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    init_wandb("mmlu", args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    # ---- load data ----------------------------------------------------------
    print(f"[mmlu] Loading MMLU dataset...")
    items = load_mmlu(max_samples=args.max_samples, seed=args.seed)
    subjects = sorted(set(it["subject"] for it in items))
    print(f"[mmlu] {len(items)} questions across {len(subjects)} subjects")

    # ---- load model ---------------------------------------------------------
    print(f"[mmlu] Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()

    # ---- score --------------------------------------------------------------
    item_results = score_mcq(model, tokenizer, items, device, args.max_length)

    # ---- aggregate per subject ----------------------------------------------
    subject_stats = {}
    for r in item_results:
        s = r["subject"]
        if s not in subject_stats:
            subject_stats[s] = {"correct": 0, "total": 0}
        subject_stats[s]["total"] += 1
        if r["correct"]:
            subject_stats[s]["correct"] += 1

    overall_correct = sum(s["correct"] for s in subject_stats.values())
    overall_total = sum(s["total"] for s in subject_stats.values())
    overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0.0

    print(f"\n[mmlu] Overall accuracy: {overall_accuracy:.4f} ({overall_correct}/{overall_total})")

    # ---- save CSV -----------------------------------------------------------
    os.makedirs(args.outdir, exist_ok=True)

    csv_rows = []
    for subj in sorted(subject_stats.keys()):
        st = subject_stats[subj]
        acc = st["correct"] / st["total"] if st["total"] > 0 else 0.0
        csv_rows.append({
            "subject": subj,
            "accuracy": round(acc, 4),
            "correct": st["correct"],
            "total": st["total"],
        })

    write_csv(
        os.path.join(args.outdir, "mmlu_results.csv"),
        csv_rows,
        ["subject", "accuracy", "correct", "total"],
    )

    # ---- summary JSON -------------------------------------------------------
    summary = {
        "model": args.model,
        "max_samples": args.max_samples,
        "num_subjects": len(subject_stats),
        "overall_accuracy": round(overall_accuracy, 4),
        "overall_correct": overall_correct,
        "overall_total": overall_total,
        "per_subject": {
            subj: round(st["correct"] / st["total"], 4, ) if st["total"] > 0 else 0.0
            for subj, st in sorted(subject_stats.items())
        },
    }
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- plot ---------------------------------------------------------------
    subj_names = [r["subject"] for r in csv_rows]
    subj_accs = [r["accuracy"] for r in csv_rows]

    fig, ax = plt.subplots(figsize=(max(10, len(subj_names) * 0.35), 6))

    colors = ["#2ecc71" if a >= 0.5 else "#e74c3c" if a < 0.30 else "#f39c12"
              for a in subj_accs]
    bars = ax.barh(range(len(subj_names)), subj_accs, color=colors, alpha=0.8)
    ax.set_yticks(range(len(subj_names)))
    ax.set_yticklabels(subj_names, fontsize=7)
    ax.set_xlabel("Accuracy")
    ax.set_xlim(0, 1.0)
    ax.axvline(0.25, color="gray", ls=":", alpha=0.5, label="Random chance (0.25)")
    ax.axvline(overall_accuracy, color="tab:blue", ls="--", alpha=0.7,
               label=f"Overall ({overall_accuracy:.3f})")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title(f"MMLU Accuracy — {args.model}\n(n={overall_total})")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "mmlu_accuracy.png"), dpi=150)
    plt.close()

    print(f"\n[mmlu] ✓ Results saved to {args.outdir}")
    log_csv_as_table(os.path.join(args.outdir, "mmlu_results.csv"), "mmlu_results")
    log_plots(args.outdir, "mmlu")
    finish_wandb()


if __name__ == "__main__":
    main()
