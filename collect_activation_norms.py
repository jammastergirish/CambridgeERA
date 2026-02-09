#!/usr/bin/env python
# /// script
# dependencies = [
#   "torch",
#   "transformers",
#   "numpy",
#   "tqdm",
# ]
# ///

import argparse
import csv
import os
import shutil
import tempfile
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import resolve_device, resolve_dtype


def read_lines(path: str, max_samples: int) -> List[str]:
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return lines[:max_samples]


@torch.inference_mode()
def cache_hidden_states(
    model_id: str,
    texts: List[str],
    cache_dir: str,
    device: str,
    dtype: torch.dtype,
    max_length: int,
    batch_size: int,
    use_fp16_cache: bool = False,
) -> Dict[str, List[float]]:
    """
    Run model on texts, cache hidden states to disk, and return absolute norms.
    Returns dict with "mean_norm_L1" and "mean_norm_L2" per layer.
    """
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, low_cpu_mem_usage=True)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, low_cpu_mem_usage=True)

    model.eval()
    model.to(device)

    # Accumulators for absolute norms
    sum_L1 = None
    sum_L2 = None
    total_tokens = 0.0

    for batch_idx, i in enumerate(tqdm(range(0, len(texts), batch_size), desc=f"Caching {model_id}")):
        batch = texts[i : i + batch_size]
        inp = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inp = {k: v.to(device) for k, v in inp.items()}
        out = model(**inp, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states

        if hs is None:
            raise RuntimeError("No hidden_states returned; cannot compute activation norms.")

        if sum_L1 is None:
            sum_L1 = [0.0] * len(hs)
            sum_L2 = [0.0] * len(hs)

        mask = inp["attention_mask"].float()
        token_count = float(mask.sum().item())
        total_tokens += token_count

        # Compute absolute norms
        for layer_idx, h in enumerate(hs):
            hf = h.float()
            L1_norms = hf.abs().sum(dim=-1)  # [B, T]
            L2_norms = torch.linalg.vector_norm(hf, ord=2, dim=-1)  # [B, T]
            sum_L1[layer_idx] += float((L1_norms * mask).sum().item())
            sum_L2[layer_idx] += float((L2_norms * mask).sum().item())

        # Save hidden states and mask for this batch (for diff computation later)
        if use_fp16_cache:
            # Use half precision to save disk space and I/O time
            batch_data = {
                "hidden_states": [h.cpu().half() for h in hs],  # Convert to fp16 for storage
                "attention_mask": inp["attention_mask"].cpu(),
            }
        else:
            # Keep full precision
            batch_data = {
                "hidden_states": [h.cpu() for h in hs],
                "attention_mask": inp["attention_mask"].cpu(),
            }
        torch.save(batch_data, os.path.join(cache_dir, f"batch_{batch_idx}.pt"))

    # Cleanup model from memory
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {
        "mean_norm_L1": [s / total_tokens if total_tokens else 0.0 for s in sum_L1],
        "mean_norm_L2": [s / total_tokens if total_tokens else 0.0 for s in sum_L2],
        "num_layers": len(sum_L2),
    }


@torch.inference_mode()
def compute_activation_diffs(
    model_id: str,
    texts: List[str],
    cache_dir: str,
    device: str,
    dtype: torch.dtype,
    max_length: int,
    batch_size: int,
    num_layers: int,
) -> Dict[str, List[float]]:
    """
    Run model_b on texts, load cached hidden states from model_a, compute diffs.
    Returns absolute norms for model_b AND diff norms.
    """
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, low_cpu_mem_usage=True)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, low_cpu_mem_usage=True)

    model.eval()
    model.to(device)

    # Accumulators
    sum_abs_L1 = [0.0] * num_layers  # Absolute L1 norms for model_b
    sum_abs_L2 = [0.0] * num_layers  # Absolute L2 norms for model_b
    sum_diff_L1 = [0.0] * num_layers  # Diff L1 norms
    sum_diff_L2 = [0.0] * num_layers  # Diff L2 norms
    total_tokens = 0.0

    for batch_idx, i in enumerate(tqdm(range(0, len(texts), batch_size), desc=f"Diffing {model_id}")):
        batch = texts[i : i + batch_size]
        inp = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inp = {k: v.to(device) for k, v in inp.items()}
        out = model(**inp, output_hidden_states=True, use_cache=False)
        hs_b = out.hidden_states

        # Load cached hidden states from model_a
        cached = torch.load(os.path.join(cache_dir, f"batch_{batch_idx}.pt"), weights_only=True)
        hs_a = cached["hidden_states"]
        mask_a = cached["attention_mask"].float()

        mask = mask_a.to(device)
        token_count = float(mask.sum().item())
        total_tokens += token_count

        for layer_idx in range(num_layers):
            h_a = hs_a[layer_idx].to(device=device, dtype=torch.float32)
            h_b = hs_b[layer_idx].float()

            # Handle potential shape mismatch
            min_len = min(h_a.shape[1], h_b.shape[1])
            h_a = h_a[:, :min_len, :]
            h_b = h_b[:, :min_len, :]
            layer_mask = mask[:, :min_len]

            # Absolute norms for model_b
            abs_L1 = h_b.abs().sum(dim=-1)
            abs_L2 = torch.linalg.vector_norm(h_b, ord=2, dim=-1)
            sum_abs_L1[layer_idx] += float((abs_L1 * layer_mask).sum().item())
            sum_abs_L2[layer_idx] += float((abs_L2 * layer_mask).sum().item())

            # Diff norms
            dh = h_b - h_a
            L1_norms = dh.abs().sum(dim=-1)
            L2_norms = torch.linalg.vector_norm(dh, ord=2, dim=-1)

            sum_diff_L1[layer_idx] += float((L1_norms * layer_mask).sum().item())
            sum_diff_L2[layer_idx] += float((L2_norms * layer_mask).sum().item())

    # Cleanup
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {
        "mean_norm_L1": [s / total_tokens if total_tokens else 0.0 for s in sum_abs_L1],
        "mean_norm_L2": [s / total_tokens if total_tokens else 0.0 for s in sum_abs_L2],
        "mean_dh_L1": [s / total_tokens if total_tokens else 0.0 for s in sum_diff_L1],
        "mean_dh_L2": [s / total_tokens if total_tokens else 0.0 for s in sum_diff_L2],
    }


def write_csv(path: str, rows: List[Dict], fieldnames: List[str]) -> None:
    dirname = os.path.dirname(path)
    if dirname:  # Only create directory if path has a directory component
        os.makedirs(dirname, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-a", required=True, help="Baseline model (before)")
    ap.add_argument("--model-b", required=True, help="Target model (after)")
    ap.add_argument("--forget-text", help="Path to forget prompts")
    ap.add_argument("--retain-text", help="Path to retain prompts")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--max-samples", type=int, default=128)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--outdir", default="outputs/activation_stats")
    ap.add_argument("--cache-fp16", action="store_true",
                    help="Use half-precision caching to reduce disk I/O (default: False)")
    args = ap.parse_args()

    if not args.forget_text or not os.path.exists(args.forget_text):
        print("Skipping: forget-text missing/not found")
        return
    if not args.retain_text or not os.path.exists(args.retain_text):
        print("Skipping: retain-text missing/not found")
        return

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    forget = read_lines(args.forget_text, args.max_samples)
    retain = read_lines(args.retain_text, args.max_samples)

    rows = []

    # Warn about caching mode for reproducibility
    if args.cache_fp16:
        print("[WARNING] Using FP16 caching - this reduces precision for faster I/O")
        print("          For exact reproducibility, omit --cache-fp16")
    else:
        print("[INFO] Using full precision caching (default)")

    for split_name, texts in [("forget", forget), ("retain", retain)]:
        cache_dir = tempfile.mkdtemp(prefix="activation_cache_")

        try:
            print(f"\n=== {split_name.upper()} split ===")

            # Step 1: Cache model_a hidden states + get absolute norms
            result_a = cache_hidden_states(
                args.model_a, texts, cache_dir, device, dtype, args.max_length, args.batch_size, args.cache_fp16
            )
            num_layers = result_a["num_layers"]

            # Step 2: Run model_b, compute absolute norms + diffs
            result_b = compute_activation_diffs(
                args.model_b, texts, cache_dir, device, dtype, args.max_length, args.batch_size, num_layers
            )

            # Collect results
            for layer_idx in range(num_layers):
                rows.append({
                    "layer": layer_idx,
                    "split": split_name,
                    "model_a_norm_L1": result_a["mean_norm_L1"][layer_idx],
                    "model_a_norm_L2": result_a["mean_norm_L2"][layer_idx],
                    "model_b_norm_L1": result_b["mean_norm_L1"][layer_idx],
                    "model_b_norm_L2": result_b["mean_norm_L2"][layer_idx],
                    "mean_dh_L1": result_b["mean_dh_L1"][layer_idx],
                    "mean_dh_L2": result_b["mean_dh_L2"][layer_idx],
                })

            # Summary row
            rows.append({
                "layer": "ALL_MEAN",
                "split": split_name,
                "model_a_norm_L1": float(np.mean(result_a["mean_norm_L1"])),
                "model_a_norm_L2": float(np.mean(result_a["mean_norm_L2"])),
                "model_b_norm_L1": float(np.mean(result_b["mean_norm_L1"])),
                "model_b_norm_L2": float(np.mean(result_b["mean_norm_L2"])),
                "mean_dh_L1": float(np.mean(result_b["mean_dh_L1"])),
                "mean_dh_L2": float(np.mean(result_b["mean_dh_L2"])),
            })

        finally:
            shutil.rmtree(cache_dir, ignore_errors=True)

    # Write output
    os.makedirs(args.outdir, exist_ok=True)

    outpath = os.path.join(args.outdir, "activation_stats.csv")
    fieldnames = ["layer", "split", "model_a_norm_L1", "model_a_norm_L2", "model_b_norm_L1", "model_b_norm_L2", "mean_dh_L1", "mean_dh_L2"]
    write_csv(outpath, rows, fieldnames)
    print(f"\nWrote: {outpath}")


if __name__ == "__main__":
    main()
