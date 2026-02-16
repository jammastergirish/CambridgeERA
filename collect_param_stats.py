#!/usr/bin/env python
# /// script
# dependencies = [
#   "torch",
#   "transformers",
#   "numpy",
#   "tqdm",
#   "safetensors",
#   "huggingface_hub",
#   "wandb",
#   "pandas",
# ]
# ///

import argparse
import gc
import json
import os
from typing import Optional, Set

import numpy as np
import torch
from tqdm import tqdm
from safetensors.torch import load_file as load_safetensors_file
from safetensors import safe_open
from huggingface_hub import snapshot_download

from utils import (
    resolve_device,
    resolve_dtype,
    extract_layer,
    classify_coarse,
    stable_rank,
    empirical_rank,
    write_csv,
    init_wandb,
    log_csv_as_table,
    finish_wandb,
)


class SmartLoader:
    def __init__(self, model_path: str):
        # Handle HF Hub IDs: If path doesn't exist locally, try downloading
        if not os.path.exists(model_path):
            print(f"'{model_path}' not found locally. Attempting HF Hub download...")
            try:
                # Download only essential files for stats
                model_path = snapshot_download(
                    repo_id=model_path, 
                    allow_patterns=["*.safetensors", "*.bin", "*.json"],
                    ignore_patterns=["*.msgpack", "*.h5"]
                )
                print(f"Downloaded/Found at: {model_path}")
            except Exception as e:
                # If it looks like a path but failed, or network failed, we'll crash later or raise here
                pass

        self.model_path = model_path
        self.index = {}
        self.is_safetensors = False
        self.single_file = None
        self._scan_structure()
        
        # Cache one shard at a time
        self.current_shard_path = None
        self.current_shard_data = {}

    def _scan_structure(self):
        # 1. Check for safetensors index
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
            # Fallback: check if user passed a direct file path instead of dir
            if os.path.isfile(self.model_path):
                if self.model_path.endswith(".safetensors"):
                    self.is_safetensors = True
                    self.single_file = self.model_path
                else:
                    self.is_safetensors = False
                    self.single_file = self.model_path
            else:
                raise FileNotFoundError(f"Could not find model weights in {self.model_path}")

    def get_all_param_names(self) -> Set[str]:
        if self.index:
            return set(self.index.keys())
        
        # Single file case: we must peek
        if self.is_safetensors:
            with safe_open(self.single_file, framework="pt", device="cpu") as f:
                return set(f.keys())
        else:
            # Torch bin: heavy, but we have to load to list keys if we essentially want to know them
            # Optimization: Just load 'map_location="meta"'? No, torch.load doesn't support that well for full files.
            # We'll just load it once to get keys and cache it if it fits, or trust usage.
            # BUT for high-mem safety, let's just claim strictly needed params.
            # Actually, `torch.load` of a 10GB file might kill us merely to list keys.
            # Let's hope single-file models aren't huge (usually <10GB).
            print(f"Warning: Loading full checkpoint {self.single_file} to list keys (Legacy PT format).")
            self.current_shard_data = torch.load(self.single_file, map_location="cpu", weights_only=True)
            self.current_shard_path = self.single_file
            return set(self.current_shard_data.keys())

    def get_param(self, name: str, device: str, dtype: torch.dtype) -> Optional[torch.Tensor]:
        # Determine which file contains this param
        if self.single_file:
            path = self.single_file
        else:
            if name not in self.index:
                return None
            filename = self.index[name]
            path = os.path.join(self.model_path, filename)
        
        # Check if loaded
        if self.current_shard_path != path:
            # Unload old
            del self.current_shard_data
            gc.collect()
            
            # Load new
            self.current_shard_path = path
            if self.is_safetensors:
                self.current_shard_data = load_safetensors_file(path, device="cpu")
            else:
                self.current_shard_data = torch.load(path, map_location="cpu", weights_only=True)
                
        # Get tensor
        if name not in self.current_shard_data:
            return None
            
        tensor = self.current_shard_data[name]
        
        # Convert / Move
        # We assume tensor is on CPU initially to save GPU mem, then we move just this param
        tensor = tensor.to(dtype=dtype, device=device)
        return tensor





def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-a", required=True, help="Baseline / before model path")
    ap.add_argument("--model-b", required=True, help="After model path")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--sr-iters", type=int, default=5)
    ap.add_argument("--empirical-rank", action="store_true", default=False,
                     help="Compute empirical rank via full SVD (slow, off by default)")
    ap.add_argument("--empirical-threshold", type=float, default=0.99,
                     help="Threshold for empirical rank (fraction of variance to capture, default: 0.99)")
    ap.add_argument("--outdir", default="outputs/param_stats")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    args = ap.parse_args()
    init_wandb("collect_param_stats", args)

    # Set seed for reproducibility (vital for Power Iteration stability)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    print(f"[collect_param_stats] Initializing SmartLoaders (streaming mode)")
    print(f"  Model A (baseline) : {args.model_a}")
    print(f"  Model B (target)   : {args.model_b}")
    print(f"  Device: {device}  |  Dtype: {dtype}")

    try:
        loader_a = SmartLoader(args.model_a)
        loader_b = SmartLoader(args.model_b)
    except FileNotFoundError as e:
        print(f"Error loading models: {e}")
        return

    # Get intersect of parameters
    names_a = loader_a.get_all_param_names()
    names_b = loader_b.get_all_param_names()
    
    # Filter for Weights only (skip biases, layernorms mostly by name convention for stats)
    # We stick to the logic: Must end in .weight
    all_names = sorted(list(names_a.intersection(names_b)))
    
    linear_names = []
    # Pre-scan to filter potential linear names
    for n in all_names:
        if n.endswith(".weight"):
            linear_names.append(n)

    rows = []
    per_layer = {}

    print(f"[collect_param_stats] Scanning {len(linear_names)} weight matrices for Frobenius norm, stable rank & empirical rank...")
    
    for name in tqdm(linear_names, desc="Comparing weight matrices", unit="matrix"):
        # Load A
        Wa = loader_a.get_param(name, device, dtype)
        if Wa is None or Wa.ndim != 2:
            continue
            
        # Load B
        Wb = loader_b.get_param(name, device, dtype)
        if Wb is None:
            continue
            
        if Wa.shape != Wb.shape:
            print(f"Skipping {name}: shape mismatch {Wa.shape} vs {Wb.shape}")
            continue

        # Calc Stats
        dW = (Wb - Wa)
        
        layer = extract_layer(name)
        group = classify_coarse(name)

        dW_fro = float(dW.float().norm().item())
        dW_sr = stable_rank(dW, iters=args.sr_iters)
        W_sr = stable_rank(Wa, iters=args.sr_iters)

        row = {
            "name": name,
            "layer": layer if layer is not None else -1,
            "group": group,
            "shape0": Wa.shape[0],
            "shape1": Wa.shape[1],
            "dW_fro": dW_fro,
            "dW_stable_rank": dW_sr,
            "W_stable_rank": W_sr,
        }

        if args.empirical_rank:
            dW_er = empirical_rank(dW, threshold=args.empirical_threshold)
            W_er = empirical_rank(Wa, threshold=args.empirical_threshold)
            row["dW_empirical_rank"] = dW_er
            row["W_empirical_rank"] = W_er

        rows.append(row)

        if layer is not None:
            key = (layer, group)
            defaults = {"sum_dW_fro_sq": 0.0, "sum_dW_sr": 0.0, "count": 0}
            if args.empirical_rank:
                defaults["sum_dW_er"] = 0.0
            st = per_layer.setdefault(key, defaults)
            st["sum_dW_fro_sq"] += dW_fro * dW_fro
            st["sum_dW_sr"] += dW_sr
            if args.empirical_rank:
                st["sum_dW_er"] += dW_er
            st["count"] += 1
            
        # Explicit delete to aid GC in loop
        del Wa
        del Wb
        del dW

    # Write Output
    os.makedirs(args.outdir, exist_ok=True)

    per_matrix_fields = ["name", "layer", "group", "shape0", "shape1", "dW_fro", "dW_stable_rank", "W_stable_rank"]
    if args.empirical_rank:
        per_matrix_fields += ["dW_empirical_rank", "W_empirical_rank"]
    write_csv(
        os.path.join(args.outdir, "per_matrix.csv"),
        rows,
        per_matrix_fields,
    )

    layer_rows = []
    for (layer, group), st in sorted(per_layer.items(), key=lambda x: (x[0][0], x[0][1])):
        row = {
            "layer": layer,
            "group": group,
            "dW_fro_layer": float(np.sqrt(st["sum_dW_fro_sq"])),
            "mean_dW_stable_rank": st["sum_dW_sr"] / max(st["count"], 1),
            "count_mats": st["count"],
        }
        if args.empirical_rank:
            row["mean_dW_empirical_rank"] = st["sum_dW_er"] / max(st["count"], 1)
        layer_rows.append(row)

    per_layer_fields = ["layer", "group", "dW_fro_layer", "mean_dW_stable_rank", "count_mats"]
    if args.empirical_rank:
        per_layer_fields.insert(-1, "mean_dW_empirical_rank")
    write_csv(
        os.path.join(args.outdir, "per_layer.csv"),
        layer_rows,
        per_layer_fields,
    )

    print(f"[collect_param_stats] ✓ Wrote {len(rows)} per-matrix rows and {len(layer_rows)} per-layer rows to {args.outdir}")
    log_csv_as_table(os.path.join(args.outdir, "per_matrix.csv"), "per_matrix")
    log_csv_as_table(os.path.join(args.outdir, "per_layer.csv"), "per_layer")
    finish_wandb()

if __name__ == "__main__":
    main()
