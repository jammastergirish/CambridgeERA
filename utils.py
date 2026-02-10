"""
Shared utilities for model_diffs analysis scripts.
"""
import csv
import os
import re
from typing import Dict, List, Optional

import torch


def sanitize_filename(name: str) -> str:
    """
    Sanitize a string for safe use in filenames.
    Replaces any character that isn't alphanumeric, underscore, dot, or hyphen with underscore.
    """
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


# --- Device / dtype resolution ---
def resolve_device(device: str) -> str:
    """Resolve 'auto' device to the best available (cuda > mps > cpu)."""
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


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


# --- Math utilities ---
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


def stable_rank(A: torch.Tensor, iters: int = 5) -> float:
    """Compute stable rank = ||A||_F^2 / ||A||_2^2 (soft measure of matrix rank)."""
    if A.numel() == 0:
        return 0.0
    Af = A.float()
    fro_sq = float((Af * Af).sum(dtype=torch.float64).item())
    if fro_sq == 0.0:
        return 0.0
    spec = spectral_norm_power(Af, iters=iters)
    if spec <= 0:
        return 0.0
    return fro_sq / (spec * spec)


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
