#!/usr/bin/env python
# /// script
# dependencies = [
#   "torch",
#   "transformers",
#   "numpy",
#   "tqdm",
#   "safetensors",
#   "huggingface_hub",
# ]
# ///

"""
Per-component, per-layer weight comparison (cosine similarity + relative Frobenius)
between Deep Ignorance model variants.

Reuses SmartLoader from collect_param_stats.py for shard-streaming model loading.
"""

import argparse
import csv
import gc
import json
import os
import re
from typing import Dict, List, Optional, Set

import numpy as np
import torch
from tqdm import tqdm
from safetensors.torch import load_file as load_safetensors_file
from safetensors import safe_open
from huggingface_hub import snapshot_download


# --- layer parsing ---
LAYER_PATTERNS = [
    re.compile(r"\.layers\.(\d+)\."),
    re.compile(r"\.h\.(\d+)\."),
    re.compile(r"\.blocks\.(\d+)\."),
]


class SmartLoader:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            print(f"'{model_path}' not found locally. Attempting HF Hub download...")
            try:
                model_path = snapshot_download(
                    repo_id=model_path,
                    allow_patterns=["*.safetensors", "*.bin", "*.json"],
                    ignore_patterns=["*.msgpack", "*.h5"],
                )
                print(f"Downloaded/Found at: {model_path}")
            except Exception:
                pass

        self.model_path = model_path
        self.index = {}
        self.is_safetensors = False
        self.single_file = None
        self._scan_structure()

        self.current_shard_path = None
        self.current_shard_data = {}

    def _scan_structure(self):
        sf_index = os.path.join(self.model_path, "model.safetensors.index.json")
        sf_single = os.path.join(self.model_path, "model.safetensors")
        pt_index = os.path.join(self.model_path, "pytorch_model.bin.index.json")
        pt_single = os.path.join(self.model_path, "pytorch_model.bin")

        if os.path.exists(sf_index):
            self.is_safetensors = True
            with open(sf_index, "r") as f:
                data = json.load(f)
            self.index = data["weight_map"]
        elif os.path.exists(sf_single):
            self.is_safetensors = True
            self.single_file = sf_single
        elif os.path.exists(pt_index):
            self.is_safetensors = False
            with open(pt_index, "r") as f:
                data = json.load(f)
            self.index = data["weight_map"]
        elif os.path.exists(pt_single):
            self.is_safetensors = False
            self.single_file = pt_single
        else:
            if os.path.isfile(self.model_path):
                self.is_safetensors = self.model_path.endswith(".safetensors")
                self.single_file = self.model_path
            else:
                raise FileNotFoundError(f"Could not find model weights in {self.model_path}")

    def get_all_param_names(self) -> Set[str]:
        if self.index:
            return set(self.index.keys())
        if self.is_safetensors:
            with safe_open(self.single_file, framework="pt", device="cpu") as f:
                return set(f.keys())
        else:
            print(f"Warning: Loading full checkpoint {self.single_file} to list keys.")
            self.current_shard_data = torch.load(self.single_file, map_location="cpu", weights_only=True)
            self.current_shard_path = self.single_file
            return set(self.current_shard_data.keys())

    def get_param(self, name: str, device: str, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if self.single_file:
            path = self.single_file
        else:
            if name not in self.index:
                return None
            filename = self.index[name]
            path = os.path.join(self.model_path, filename)

        if self.current_shard_path != path:
            del self.current_shard_data
            gc.collect()
            self.current_shard_path = path
            if self.is_safetensors:
                self.current_shard_data = load_safetensors_file(path, device="cpu")
            else:
                self.current_shard_data = torch.load(path, map_location="cpu", weights_only=True)

        if name not in self.current_shard_data:
            return None
        tensor = self.current_shard_data[name]
        return tensor.to(dtype=dtype, device=device)


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
    if dtype not in mapping:
        raise ValueError(f"Unknown dtype '{dtype}'. Use auto|fp32|fp16|bf16")
    return mapping[dtype]


def extract_layer(param_name: str) -> Optional[int]:
    for pat in LAYER_PATTERNS:
        m = pat.search(param_name)
        if m:
            return int(m.group(1))
    return None


def classify_component(param_name: str) -> str:
    if "query_key_value" in param_name:
        return "qkv"
    elif "attention.dense" in param_name:
        return "proj"
    elif "dense_h_to_4h" in param_name:
        return "mlp_expand"
    elif "dense_4h_to_h" in param_name:
        return "mlp_contract"
    return "other"


def write_csv(path: str, rows: List[Dict], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def fast_spectral_norm(W, num_it=10):
    """Find sigma_1(W), the largest singular value of W, via power iteration."""
    v = torch.randn(W.shape[1], 1, device=W.device, dtype=W.dtype)
    for _ in range(num_it):
        u = torch.matmul(W, v)
        u_norm = torch.norm(u)
        if u_norm == 0:
            return 0.0
        u = u / u_norm
        v = torch.matmul(W.t(), u)
        v_norm = torch.norm(v)
        if v_norm == 0:
            return 0.0
        v = v / v_norm
    sigma_1 = torch.norm(torch.matmul(W, v))
    return sigma_1.item()


def _compute_metrics(Wa: torch.Tensor, Wb: torch.Tensor):
    """Compute all per-layer metrics for a weight pair and their diff."""
    Wa_f = Wa.float()
    Wb_f = Wb.float()
    Wa_flat = Wa_f.flatten()
    Wb_flat = Wb_f.flatten()

    # Cosine similarity between the two weight tensors
    cos_sim = torch.nn.functional.cosine_similarity(
        Wa_flat.unsqueeze(0), Wb_flat.unsqueeze(0)
    ).item()

    # Element-wise difference
    dW = Wa_f - Wb_f
    dW_flat = dW.flatten()
    n_elem = dW_flat.numel()

    # Frobenius norms
    fro_norm = dW_flat.norm().item()
    Wb_norm = Wb_flat.norm().item()
    rel_fro = fro_norm / Wb_norm if Wb_norm > 0 else float("inf")
    fro_norm_normalized = fro_norm / (n_elem ** 0.5)

    # Element-wise stats of the diff
    diff_mean = dW_flat.mean().item()
    diff_std = dW_flat.std().item()
    diff_abs_mean = dW_flat.abs().mean().item()

    # Spectral norm of the diff (largest singular value)
    diff_spectral_norm = fast_spectral_norm(dW)

    return {
        "cosine_sim": cos_sim,
        "rel_frobenius": rel_fro,
        "frobenius_norm": fro_norm,
        "fro_norm_normalized": fro_norm_normalized,
        "diff_mean": diff_mean,
        "diff_std": diff_std,
        "diff_abs_mean": diff_abs_mean,
        "diff_spectral_norm": diff_spectral_norm,
        "elements": n_elem,
    }


def _pick_sanity_params(all_linear_names: List[str], n: int = 3) -> List[str]:
    """Pick up to n weight names that belong to known components for sanity checks."""
    picked = []
    for name in all_linear_names:
        if classify_component(name) != "other" and extract_layer(name) is not None:
            picked.append(name)
            if len(picked) >= n:
                break
    return picked


def run_sanity_checks(
    loader_a: SmartLoader,
    loader_b: SmartLoader,
    linear_names: List[str],
    device: str,
    dtype: torch.dtype,
) -> bool:
    """Run sanity checks. Returns True if all pass, False otherwise."""
    sample = _pick_sanity_params(linear_names)
    if not sample:
        print("FAIL: No suitable weight matrices found for sanity checks")
        return False

    all_passed = True

    # --- Check 1: Self-comparison (A vs A) ---
    print("\n--- Sanity Check 1: Self-comparison (A vs A) ---")
    for name in sample:
        Wa = loader_a.get_param(name, device, dtype)
        m = _compute_metrics(Wa, Wa)

        cos_ok = m["cosine_sim"] == 1.0
        fro_ok = m["rel_frobenius"] == 0.0
        passed = cos_ok and fro_ok

        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")
        if not cos_ok:
            print(f"         cosine_sim = {m['cosine_sim']} (expected 1.0)")
        if not fro_ok:
            print(f"         rel_frobenius = {m['rel_frobenius']} (expected 0.0)")

        if not passed:
            all_passed = False
        del Wa

    # --- Check 2: Symmetry cos(A,B) == cos(B,A) ---
    print("\n--- Sanity Check 2: Cosine similarity symmetry ---")
    for name in sample:
        Wa = loader_a.get_param(name, device, dtype)
        Wb = loader_b.get_param(name, device, dtype)
        if Wb is None or Wa.shape != Wb.shape:
            print(f"  SKIP: {name} (shape mismatch or missing)")
            continue

        m_ab = _compute_metrics(Wa, Wb)
        m_ba = _compute_metrics(Wb, Wa)

        passed = abs(m_ab["cosine_sim"] - m_ba["cosine_sim"]) < 1e-6
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")
        print(f"         cos(A,B) = {m_ab['cosine_sim']:.10f}")
        print(f"         cos(B,A) = {m_ba['cosine_sim']:.10f}")
        if not passed:
            print(f"         delta = {abs(m_ab['cosine_sim'] - m_ba['cosine_sim']):.2e}")
            all_passed = False

        del Wa, Wb

    # --- Check 3: Valid ranges ---
    print("\n--- Sanity Check 3: Value ranges ---")
    range_violations = 0
    for name in sample:
        Wa = loader_a.get_param(name, device, dtype)
        Wb = loader_b.get_param(name, device, dtype)
        if Wb is None or Wa.shape != Wb.shape:
            continue

        m = _compute_metrics(Wa, Wb)

        cos_ok = -1.0 <= m["cosine_sim"] <= 1.0
        fro_ok = m["rel_frobenius"] >= 0.0
        norm_ok = m["frobenius_norm"] >= 0.0

        passed = cos_ok and fro_ok and norm_ok
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")
        if not cos_ok:
            print(f"         cosine_sim = {m['cosine_sim']} (out of [-1, 1])")
            range_violations += 1
        if not fro_ok:
            print(f"         rel_frobenius = {m['rel_frobenius']} (negative)")
            range_violations += 1
        if not norm_ok:
            print(f"         frobenius_norm = {m['frobenius_norm']} (negative)")
            range_violations += 1

        if not passed:
            all_passed = False
        del Wa, Wb

    # --- Summary ---
    print()
    if all_passed:
        print("All sanity checks PASSED")
    else:
        print("Some sanity checks FAILED — aborting")
    print()
    return all_passed


def main():
    ap = argparse.ArgumentParser(
        description="Compute per-component, per-layer cosine similarity and relative "
                    "Frobenius norm between two model checkpoints."
    )
    ap.add_argument("--model-a", required=True, help="First model (numerator in cosine sim)")
    ap.add_argument("--model-b", required=True, help="Second model (denominator / normalization target)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--outdir", default="outputs/weight_comparison")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sanity-check", action="store_true",
                     help="Run sanity checks before main comparison and exit early on failure")
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    print(f"Device: {device}, Dtype: {dtype}")
    print(f"Model A: {args.model_a}")
    print(f"Model B: {args.model_b}")

    try:
        loader_a = SmartLoader(args.model_a)
        loader_b = SmartLoader(args.model_b)
    except FileNotFoundError as e:
        print(f"Error loading models: {e}")
        return

    names_a = loader_a.get_all_param_names()
    names_b = loader_b.get_all_param_names()
    all_names = sorted(names_a.intersection(names_b))

    # Filter to 2D weight tensors only
    linear_names = [n for n in all_names if n.endswith(".weight")]

    if args.sanity_check:
        if not run_sanity_checks(loader_a, loader_b, linear_names, device, dtype):
            return

    rows = []
    print(f"Scanning {len(linear_names)} weight matrices...")

    for name in tqdm(linear_names, desc="Processing"):
        Wa = loader_a.get_param(name, device, dtype)
        if Wa is None or Wa.ndim != 2:
            continue

        Wb = loader_b.get_param(name, device, dtype)
        if Wb is None:
            continue

        if Wa.shape != Wb.shape:
            print(f"Skipping {name}: shape mismatch {Wa.shape} vs {Wb.shape}")
            continue

        layer = extract_layer(name)
        component = classify_component(name)

        # Skip non-layer or non-component weights (embeddings, final LN, etc.)
        if layer is None or component == "other":
            del Wa, Wb
            continue

        metrics = _compute_metrics(Wa, Wb)

        row = {"layer": layer, "component": component}
        row.update(metrics)
        rows.append(row)

        del Wa, Wb

    # --- Output ---
    out_a = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.model_a)
    out_b = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.model_b)
    outdir = os.path.join(args.outdir, f"{out_a}__vs__{out_b}")
    os.makedirs(outdir, exist_ok=True)

    # per_component.csv
    per_component_fields = [
        "layer", "component", "elements",
        "cosine_sim", "rel_frobenius", "frobenius_norm", "fro_norm_normalized",
        "diff_mean", "diff_std", "diff_abs_mean", "diff_spectral_norm",
    ]
    rows.sort(key=lambda r: (r["layer"], r["component"]))
    write_csv(os.path.join(outdir, "per_component.csv"), rows, per_component_fields)

    # summary.csv — aggregate stats per component across layers
    from collections import defaultdict
    metric_keys = [
        "cosine_sim", "rel_frobenius", "frobenius_norm", "fro_norm_normalized",
        "diff_mean", "diff_std", "diff_abs_mean", "diff_spectral_norm",
    ]
    comp_stats = defaultdict(lambda: {k: [] for k in metric_keys})
    for r in rows:
        c = r["component"]
        for k in metric_keys:
            comp_stats[c][k].append(r[k])

    summary_rows = []
    for comp in ["qkv", "proj", "mlp_expand", "mlp_contract"]:
        if comp not in comp_stats:
            continue
        s = comp_stats[comp]
        sr = {
            "component": comp,
            "n_layers": len(s["cosine_sim"]),
            "elements": rows[0]["elements"] if rows else 0,  # same within component
        }
        for k in metric_keys:
            vals = s[k]
            sr[f"{k}_mean"] = float(np.mean(vals))
            sr[f"{k}_min"] = float(np.min(vals))
            sr[f"{k}_max"] = float(np.max(vals))
            sr[f"{k}_std"] = float(np.std(vals))
        summary_rows.append(sr)

    # Fix elements per component (each component has different size)
    for sr in summary_rows:
        comp_rows = [r for r in rows if r["component"] == sr["component"]]
        if comp_rows:
            sr["elements"] = comp_rows[0]["elements"]

    summary_fields = ["component", "n_layers", "elements"]
    for k in metric_keys:
        summary_fields.extend([f"{k}_mean", f"{k}_min", f"{k}_max", f"{k}_std"])
    write_csv(os.path.join(outdir, "summary.csv"), summary_rows, summary_fields)

    # Print summary to stdout
    print(f"\n{'='*70}")
    print(f"Results: {outdir}")
    print(f"{'='*70}")
    print(f"{'Component':<15} {'Cos Sim (mean)':>15} {'Rel Fro (mean)':>15} {'Layers':>7}")
    print(f"{'-'*15} {'-'*15} {'-'*15} {'-'*7}")
    for sr in summary_rows:
        print(f"{sr['component']:<15} {sr['cosine_sim_mean']:>15.6f} {sr['rel_frobenius_mean']:>15.6f} {sr['n_layers']:>7}")
    print()


if __name__ == "__main__":
    main()
