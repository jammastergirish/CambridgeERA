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
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


def resolve_dtype(dtype: str, device: str) -> torch.dtype:
    if dtype == "auto":
        if device == "cuda":
            return torch.bfloat16
        if device == "mps":
            return torch.float16
        return torch.float32
    mapping = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    return mapping[dtype]


def read_lines(path: str, max_samples: int) -> List[str]:
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return lines[:max_samples]


@torch.inference_mode()
def mean_hidden_norms(model_id: str, texts: List[str], device: str, dtype: torch.dtype, max_length: int, batch_size: int) -> List[float]:
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, low_cpu_mem_usage=True)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, low_cpu_mem_usage=True)

    model.eval()
    model.to(device)

    sums = None
    counts = None

    for i in tqdm(range(0, len(texts), batch_size), desc=f"{model_id} forward"):
        batch = texts[i : i + batch_size]
        inp = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inp = {k: v.to(device) for k, v in inp.items()}
        out = model(**inp, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states
        if hs is None:
            raise RuntimeError("No hidden_states returned; cannot compute activation norms.")

        if sums is None:
            sums = [0.0 for _ in hs]
            counts = [0.0 for _ in hs]

        mask = inp["attention_mask"].float()
        token_count = float(mask.sum().item())

        for li, h in enumerate(hs):
            # h: [B, T, D]
            norms = torch.linalg.vector_norm(h.float(), ord=2, dim=-1)  # [B, T]
            sums[li] += float((norms * mask).sum().item())
            counts[li] += token_count

    return [s / c if c else 0.0 for s, c in zip(sums, counts)]


def write_csv(path: str, rows: List[Dict], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--forget-text")
    ap.add_argument("--retain-text")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--max-samples", type=int, default=128)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--outdir", default="outputs/activation_norms")
    args = ap.parse_args()

    # If you don't want activations, just don't run this script.
    # Also: fail gracefully if files missing.
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
    for model_id in args.models:
        for split_name, texts in [("forget", forget), ("retain", retain)]:
            norms = mean_hidden_norms(model_id, texts, device=device, dtype=dtype, max_length=args.max_length, batch_size=args.batch_size)
            for layer_idx, v in enumerate(norms):
                rows.append({"model": model_id, "split": split_name, "layer": layer_idx, "mean_norm": v})
            rows.append({"model": model_id, "split": split_name, "layer": "ALL_MEAN", "mean_norm": float(np.mean(norms))})

    outpath = os.path.join(args.outdir, "activation_norms.csv")
    write_csv(outpath, rows, ["model", "split", "layer", "mean_norm"])
    print(f"Wrote: {outpath}")


if __name__ == "__main__":
    main()
