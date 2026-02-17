"""
Shared utilities for model_diffs analysis scripts.
"""
import csv
import os
import re
from typing import Dict, List, Optional

import torch


def load_dotenv(path: str = None):
    """Load .env file into environment. No external dependencies needed."""
    if path is None:
        # Look for .env in the same directory as this file
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if value and not os.environ.get(key):  # Don't override existing env vars
                os.environ[key] = value


# Auto-load .env on import so standalone scripts get HF_TOKEN etc.
load_dotenv()


def compute_spectral_norm(A: torch.Tensor) -> float:
    """
    Compute spectral norm (largest singular value) using SVD.
    More stable than power iteration but potentially slower.
    """
    if A.numel() == 0 or min(A.shape) == 0:
        return 0.0
    try:
        s = torch.linalg.svdvals(A.float())
        return float(s[0].item()) if len(s) > 0 else 0.0
    except:
        return spectral_norm_power(A)  # Fallback to power iteration


# --- Device / dtype resolution ---
def resolve_device(device: str) -> str:
    """Resolve 'auto' device to the best available (cuda > mps > cpu)."""
    if device != "auto":
        resolved = device
    elif torch.cuda.is_available():
        resolved = "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        resolved = "mps"
    else:
        resolved = "cpu"
    print(f"[device] Using device: {resolved}" + (f" (resolved from '{device}')" if device == "auto" else ""))
    return resolved


def resolve_dtype(dtype: str, device: str) -> torch.dtype:
    """Resolve 'auto' dtype based on device, or parse explicit dtype string."""
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


# --- Parameter name parsing ---
LAYER_PATTERNS = [
    re.compile(r"\.layers\.(\d+)\."),
    re.compile(r"\.h\.(\d+)\."),
    re.compile(r"\.blocks\.(\d+)\."),
]
COARSE_ATTN_KEYS = ("attn", "attention", "self_attn")
COARSE_MLP_KEYS = ("mlp", "ffn", "feed_forward", "intermediate")


def extract_layer(param_name: str) -> Optional[int]:
    """Extract layer number from parameter name (e.g., 'layers.15.mlp' -> 15)."""
    for pat in LAYER_PATTERNS:
        m = pat.search(param_name)
        if m:
            return int(m.group(1))
    return None


def classify_coarse(param_name: str) -> str:
    """Classify parameter into coarse groups: 'attn', 'mlp', or 'other'."""
    s = param_name.lower()
    if any(k in s for k in COARSE_ATTN_KEYS):
        return "attn"
    if any(k in s for k in COARSE_MLP_KEYS):
        return "mlp"
    return "other"


# Component-level classification (granular sub-component within attn / mlp)
# Maps param name fragments → component label
_COMP_RULES = [
    # Attention QKV (fused or separate)
    ("attention.query_key_value", "qkv"),
    ("self_attn.qkv", "qkv"),
    ("self_attn.q_proj", "qkv"),
    ("self_attn.k_proj", "qkv"),
    ("self_attn.v_proj", "qkv"),
    ("attn.q_proj", "qkv"),
    ("attn.k_proj", "qkv"),
    ("attn.v_proj", "qkv"),
    ("query_key_value", "qkv"),
    # Attention output projection
    ("attention.dense", "proj"),
    ("self_attn.o_proj", "proj"),
    ("attn.o_proj", "proj"),
    ("attn.out_proj", "proj"),
    # MLP expand (hidden → 4h)
    ("mlp.dense_h_to_4h", "mlp_expand"),
    ("mlp.gate_proj", "mlp_expand"),
    ("mlp.up_proj", "mlp_expand"),
    ("mlp.fc1", "mlp_expand"),
    ("mlp.c_fc", "mlp_expand"),
    ("mlp.w1", "mlp_expand"),
    ("mlp.w3", "mlp_expand"),
    # MLP contract (4h → hidden)
    ("mlp.dense_4h_to_h", "mlp_contract"),
    ("mlp.down_proj", "mlp_contract"),
    ("mlp.fc2", "mlp_contract"),
    ("mlp.c_proj", "mlp_contract"),
    ("mlp.w2", "mlp_contract"),
]


def classify_granular(param_name: str) -> str:
    """Classify parameter into granular component: 'qkv', 'proj', 'mlp_expand', 'mlp_contract', or 'other'."""
    s = param_name.lower()
    for fragment, label in _COMP_RULES:
        if fragment in s:
            return label
    return "other"


# --- Math utilities ---
def frobenius_norm(A: torch.Tensor) -> float:
    """Compute Frobenius norm of a tensor."""
    return float(torch.norm(A.float(), p='fro').item())


def nuclear_norm(A: torch.Tensor) -> float:
    """Compute nuclear norm (sum of singular values)."""
    if A.numel() == 0 or A.ndim != 2:
        return 0.0
    try:
        s = torch.linalg.svdvals(A.float())
        return float(s.sum().item())
    except:
        return 0.0


def spectral_norm_power(A: torch.Tensor, iters: int = 5, eps: float = 1e-12) -> float:
    """Estimate spectral norm using power iteration."""
    m, n = A.shape
    if m == 0 or n == 0:
        return 0.0
    v = torch.randn(n, 1, device=A.device, dtype=A.dtype)
    v = v / (v.norm() + eps)
    for _ in range(iters):
        u = A @ v
        u = u / (u.norm() + eps)
        v = A.T @ u
        v = v / (v.norm() + eps)
    u = A @ v
    return float(u.norm().item())


def stable_rank_and_spectral(A: torch.Tensor, iters: int = 5, use_svd: bool = False) -> tuple[float, float]:
    """Compute stable rank AND spectral norm in one pass.

    Returns:
        (stable_rank, spectral_norm)
    """
    if A.numel() == 0:
        return 0.0, 0.0
    Af = A.float()
    fro_sq = float((Af * Af).sum(dtype=torch.float64).item())
    if fro_sq == 0.0:
        return 0.0, 0.0
    spec = compute_spectral_norm(Af) if use_svd else spectral_norm_power(Af, iters=iters)
    if spec <= 0:
        return 0.0, 0.0
    sr = fro_sq / (spec * spec)
    return sr, float(spec)


def stable_rank(A: torch.Tensor, iters: int = 5, use_svd: bool = False) -> float:
    """Compute stable rank = ||A||_F^2 / ||A||_2^2 (soft measure of matrix rank)."""
    sr, _ = stable_rank_and_spectral(A, iters=iters, use_svd=use_svd)
    return sr


def empirical_rank(A: torch.Tensor, threshold: float = 0.99) -> int:
    """
    Compute empirical rank as the number of singular values needed to
    capture 'threshold' fraction of total variance (sum of squared singular values).

    Args:
        A: Input matrix
        threshold: Fraction of variance to capture (default 0.99)

    Returns:
        Number of singular values needed to capture threshold of variance
    """
    if A.numel() == 0:
        return 0

    # Compute SVD (we only need singular values)
    # Use float32 for memory efficiency
    Af = A.float()
    try:
        # torch.linalg.svdvals is more efficient when we only need singular values
        s = torch.linalg.svdvals(Af)
    except:
        # Fallback to standard SVD if svdvals not available
        _, s, _ = torch.linalg.svd(Af, full_matrices=False)

    # Compute squared singular values (these represent variance)
    s_squared = s * s
    total_variance = s_squared.sum().item()

    if total_variance == 0.0:
        return 0

    # Find how many singular values we need to capture threshold of variance
    cumsum = torch.cumsum(s_squared, dim=0)
    threshold_variance = threshold * total_variance

    # Find first index where cumsum exceeds threshold
    rank = torch.searchsorted(cumsum, threshold_variance).item() + 1

    # Ensure rank doesn't exceed matrix dimensions
    return min(rank, min(A.shape))


def condition_number(A: torch.Tensor, eps: float = 1e-10) -> float:
    """
    Compute condition number (ratio of largest to smallest singular value).
    Large condition numbers indicate numerical instability.
    """
    if A.numel() == 0 or A.ndim != 2:
        return 1.0
    try:
        s = torch.linalg.svdvals(A.float())
        if len(s) == 0:
            return 1.0
        s_max = s[0].item()
        s_min = s[-1].item()
        return s_max / max(s_min, eps)
    except:
        return float('inf')


def compute_rank_deficiency(A: torch.Tensor, threshold: float = 1e-6) -> int:
    """
    Compute rank deficiency (how many dimensions are effectively zero).
    Returns min(m, n) - numerical_rank.
    """
    if A.numel() == 0:
        return min(A.shape) if A.ndim == 2 else 0
    try:
        s = torch.linalg.svdvals(A.float())
        numerical_rank = (s > threshold).sum().item()
        return min(A.shape) - numerical_rank
    except:
        return 0


# --- I/O utilities ---
def write_csv(path: str, rows: List[Dict], fieldnames: List[str]) -> None:
    """Write list of dicts to CSV, creating directories as needed."""
    dirname = os.path.dirname(path)
    if dirname:  # Only create directory if path has a directory component
        os.makedirs(dirname, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# --- Weights & Biases helpers ---
def _derive_run_name(script_name: str, args) -> str:
    """Derive a descriptive W&B run name from args.outdir (last 2 path segments)."""
    outdir = getattr(args, "outdir", None)
    if not outdir:
        return script_name
    # Normalise and grab last 2 non-empty components
    parts = [p for p in outdir.replace("\\", "/").split("/") if p]
    # Strip common root prefixes like "outputs" or "unlearned_models"
    while parts and parts[0] in ("outputs", "unlearned_models", "plots"):
        parts = parts[1:]
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    if parts:
        return parts[-1]
    return script_name


def init_wandb(script_name: str, args, project: str = "cambridge_era", **kw):
    """Initialise a W&B run.  No-ops gracefully if wandb is not installed or WANDB_MODE=disabled."""
    try:
        import wandb
    except ImportError:
        print(f"[wandb] wandb not installed — skipping logging for {script_name}")
        return None
    run_name = _derive_run_name(script_name, args)
    group = os.environ.get("WANDB_RUN_GROUP", None)
    run = wandb.init(
        project=project,
        name=run_name,
        config=vars(args) if hasattr(args, "__dict__") else {},
        group=group,
        tags=[script_name],
        reinit=True,
        **kw,
    )
    return run


def log_csv_as_table(csv_path: str, key: str = "results"):
    """Upload a CSV file as a wandb.Table artefact."""
    try:
        import wandb
        if wandb.run is None:
            return
        import pandas as pd
        df = pd.read_csv(csv_path)
        wandb.log({key: wandb.Table(dataframe=df)})
    except Exception:
        pass


def log_plots(outdir: str, key_prefix: str = "plots"):
    """Glob all PNGs in *outdir* and log them as wandb.Images."""
    try:
        import wandb
        import glob as _glob
        if wandb.run is None:
            return
        for png in sorted(_glob.glob(os.path.join(outdir, "*.png"))):
            name = os.path.splitext(os.path.basename(png))[0]
            wandb.log({f"{key_prefix}/{name}": wandb.Image(png)})
    except Exception:
        pass


def finish_wandb():
    """Finish the current W&B run if active."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass
