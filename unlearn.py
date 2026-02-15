# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch",
#     "transformers",
#     "accelerate",
#     "tqdm",
# ]
# ///
"""
Multi-method LLM unlearning pipeline.

Supported methods:
  ga_simple — Pure Gradient Ascent on forget set only (no retain loss)
  ga       — Gradient Ascent on forget set + Gradient Descent on retain set
  grad_diff — Gradient Difference (weighted forget ascent + retain descent)
  dpo      — Direct Preference Optimization (forget=rejected, retain=chosen)
  npo      — Negative Preference Optimization (DPO-inspired, with reference model)
  simnpo   — Simple NPO (reference-free variant of NPO + retain NLL)
  rmu      — Representation Misdirection for Unlearning
  cb       — Circuit Breakers (representation rerouting via cosine similarity)
  lat      — Latent Adversarial Training (adversarial perturbations in hidden states)

Usage:
  uv run --script unlearn.py --model <HF_ID> --method ga --outdir outputs/test
"""

import argparse
import math
import random
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Reuse device / dtype helpers from utils.py if available, else inline
# ---------------------------------------------------------------------------
try:
    from utils import resolve_device, resolve_dtype
except ImportError:

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
        mapping = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        if dtype not in mapping:
            raise ValueError(f"Unknown dtype '{dtype}'. Use auto|fp32|fp16|bf16")
        return mapping[dtype]


# ===================================================================
# Data loading
# ===================================================================

def load_lines(path: str, max_lines: int | None = None) -> list[str]:
    """Load non-empty lines from a text file."""
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    if max_lines:
        lines = lines[:max_lines]
    return lines


def tokenize_texts(
    texts: list[str], tokenizer, max_length: int, device: str
) -> list[dict]:
    """Tokenize a list of strings into batches of input_ids + attention_mask."""
    batches = []
    for text in texts:
        enc = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        batches.append({k: v.to(device) for k, v in enc.items()})
    return batches


def make_batches(items: list[dict], batch_size: int) -> list[list[dict]]:
    """Group single-sample dicts into mini-batches."""
    batches = []
    for i in range(0, len(items), batch_size):
        chunk = items[i : i + batch_size]
        batch = {
            k: torch.cat([c[k] for c in chunk], dim=0) for k in chunk[0]
        }
        batches.append(batch)
    return batches


# ===================================================================
# Loss functions for each method
# ===================================================================

def nll_loss(model, batch: dict) -> torch.Tensor:
    """Standard next-token prediction loss (cross-entropy)."""
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :].contiguous()
    labels = input_ids[:, 1:].contiguous()
    mask = attention_mask[:, 1:].contiguous()
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)), labels.view(-1), reduction="none"
    )
    loss = (loss * mask.view(-1)).sum() / mask.sum().clamp(min=1)
    return loss


def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute per-token log-probs: shape (B, T)."""
    log_p = F.log_softmax(logits, dim=-1)
    return torch.gather(log_p, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def avg_log_prob(
    model, batch: dict, return_per_token: bool = False
) -> torch.Tensor:
    """Average per-token log-prob under `model` for the batch."""
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    with torch.no_grad() if not model.training else torch.enable_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    per_token = log_probs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
    mask = attention_mask[:, 1:].float()
    avg = (per_token * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
    if return_per_token:
        return avg, per_token, mask
    return avg


# ---- GA Simple ---------------------------------------------------------

def ga_simple_loss(model, forget_batch: dict) -> torch.Tensor:
    """Pure Gradient Ascent: negate NLL on forget set only."""
    return -nll_loss(model, forget_batch)


# ---- GA ----------------------------------------------------------------

def ga_loss(model, forget_batch: dict, retain_batch: dict) -> torch.Tensor:
    """Gradient Ascent: negate NLL on forget + standard NLL on retain."""
    l_forget = -nll_loss(model, forget_batch)  # ascent (maximise NLL)
    l_retain = nll_loss(model, retain_batch)
    return l_forget + l_retain


# ---- GradDiff ----------------------------------------------------------

def grad_diff_loss(
    model, forget_batch: dict, retain_batch: dict, forget_weight: float = 1.0
) -> torch.Tensor:
    """Gradient Difference: L = retain_NLL - weight * forget_NLL."""
    l_retain = nll_loss(model, retain_batch)
    l_forget = nll_loss(model, forget_batch)
    return l_retain - forget_weight * l_forget


# ---- DPO ---------------------------------------------------------------

def dpo_loss(
    model,
    ref_model,
    forget_batch: dict,
    retain_batch: dict,
    beta: float,
) -> torch.Tensor:
    """
    DPO loss: chosen=retain, rejected=forget.
    L = -log σ(β * (log π_θ(y_w)/π_ref(y_w) - log π_θ(y_l)/π_ref(y_l)))
    """
    # Policy log-probs
    lp_chosen = avg_log_prob(model, retain_batch)
    lp_rejected = avg_log_prob(model, forget_batch)

    # Reference log-probs (frozen)
    with torch.no_grad():
        ref_lp_chosen = avg_log_prob(ref_model, retain_batch)
        ref_lp_rejected = avg_log_prob(ref_model, forget_batch)

    logits_diff = beta * (
        (lp_chosen - ref_lp_chosen) - (lp_rejected - ref_lp_rejected)
    )
    return -F.logsigmoid(logits_diff).mean()


# ---- NPO ---------------------------------------------------------------

def npo_loss(
    model,
    ref_model,
    forget_batch: dict,
    retain_batch: dict,
    beta: float,
) -> torch.Tensor:
    """
    NPO: L = -(2/β) * E[log σ(-β * log(π_θ / π_ref))]  on forget
         + NLL on retain
    """
    lp_forget = avg_log_prob(model, forget_batch)
    with torch.no_grad():
        ref_lp_forget = avg_log_prob(ref_model, forget_batch)

    npo_term = -(2.0 / beta) * F.logsigmoid(
        -beta * (lp_forget - ref_lp_forget)
    ).mean()

    retain_nll = nll_loss(model, retain_batch)
    return npo_term + retain_nll


# ---- SimNPO ------------------------------------------------------------

def simnpo_loss(
    model,
    forget_batch: dict,
    retain_batch: dict,
    beta: float,
) -> torch.Tensor:
    """
    SimNPO (reference-free): L = -(2/β) * E[log σ(-β * avg_log_prob_θ)]
                                + NLL on retain
    """
    lp_forget = avg_log_prob(model, forget_batch)

    simnpo_term = -(2.0 / beta) * F.logsigmoid(-beta * lp_forget).mean()
    retain_nll = nll_loss(model, retain_batch)
    return simnpo_term + retain_nll


# ---- RMU ---------------------------------------------------------------

def get_layer_activations(model, batch: dict, layer_ids: list[int]):
    """
    Run a forward pass and capture hidden states at specified layer indices.
    Returns dict {layer_id: tensor of shape (B, T, D)}.
    """
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        output_hidden_states=True,
    )
    # hidden_states is a tuple of (n_layers + 1) tensors (embedding + each layer)
    hidden = outputs.hidden_states
    return {lid: hidden[lid + 1] for lid in layer_ids}  # +1 to skip embedding


def rmu_loss(
    model,
    forget_batch: dict,
    retain_batch: dict,
    layer_ids: list[int],
    random_targets: dict,  # {layer_id: (D,) tensor — fixed random direction}
    retain_targets: dict,  # {layer_id: (B, T, D) tensor — cached clean activations}
    steering_coeff: float,
    alpha: float,
) -> torch.Tensor:
    """
    RMU: push forget-set activations toward random target,
         pull retain-set activations toward original (cached) activations.
    """
    forget_acts = get_layer_activations(model, forget_batch, layer_ids)
    retain_acts = get_layer_activations(model, retain_batch, layer_ids)

    loss = torch.tensor(0.0, device=next(model.parameters()).device)

    for lid in layer_ids:
        # Forget: MSE toward (steering_coeff * random_direction)
        target_f = random_targets[lid].unsqueeze(0).unsqueeze(0) * steering_coeff
        loss = loss + F.mse_loss(forget_acts[lid], target_f.expand_as(forget_acts[lid]))

        # Retain: MSE toward cached clean activations
        target_r = retain_targets[lid]
        # Handle batch-size mismatch by truncating
        bsz = min(retain_acts[lid].size(0), target_r.size(0))
        loss = loss + alpha * F.mse_loss(
            retain_acts[lid][:bsz], target_r[:bsz].detach()
        )

    return loss


# ---- Circuit Breakers --------------------------------------------------

def cb_loss(
    model,
    forget_batch: dict,
    retain_batch: dict,
    layer_ids: list[int],
    random_targets: dict,
    retain_targets: dict,
    steering_coeff: float,
    alpha: float,
) -> torch.Tensor:
    """
    Circuit Breakers (Representation Rerouting):
    Forget: maximize cosine similarity toward random direction.
    Retain: minimize cosine distance from original activations.
    """
    forget_acts = get_layer_activations(model, forget_batch, layer_ids)
    retain_acts = get_layer_activations(model, retain_batch, layer_ids)

    loss = torch.tensor(0.0, device=next(model.parameters()).device)

    for lid in layer_ids:
        # Forget: cosine similarity toward (steering_coeff * random target)
        fa = forget_acts[lid].flatten(0, 1)  # (B*T, D)
        rt = random_targets[lid].unsqueeze(0).expand_as(fa) * steering_coeff
        # We want fa to ALIGN with rt, so minimize negative cosine sim
        cos_sim = F.cosine_similarity(fa, rt, dim=-1)
        loss = loss - cos_sim.mean()  # negate: minimize → maximize alignment

        # Retain: cosine similarity toward cached activations
        ra = retain_acts[lid]
        tr = retain_targets[lid]
        bsz = min(ra.size(0), tr.size(0))
        ra_flat = ra[:bsz].flatten(0, 1)
        tr_flat = tr[:bsz].detach().flatten(0, 1)
        retain_cos = F.cosine_similarity(ra_flat, tr_flat, dim=-1)
        loss = loss + alpha * (1.0 - retain_cos.mean())  # keep close

    return loss


# ---- LAT ---------------------------------------------------------------

def lat_loss(
    model,
    forget_batch: dict,
    retain_batch: dict,
    layer_ids: list[int],
    lat_eps: float,
    lat_steps: int,
) -> torch.Tensor:
    """
    Latent Adversarial Training:
    1. Inner loop: find adversarial perturbation δ at target layers that
       MAXIMIZES the model's ability to produce forget-set outputs.
    2. Outer loop: train model to produce HIGH loss on forget even WITH δ,
       plus standard retain NLL.
    """
    device = next(model.parameters()).device

    # --- Forward pass to get shapes ---
    with torch.no_grad():
        out = model(
            input_ids=forget_batch["input_ids"],
            attention_mask=forget_batch["attention_mask"],
            output_hidden_states=True,
        )
        # Pick a single target layer for perturbation
        target_lid = layer_ids[len(layer_ids) // 2]  # middle layer
        hidden_shape = out.hidden_states[target_lid + 1].shape

    # --- Inner loop: find adversarial perturbation ---
    delta = torch.zeros(hidden_shape, device=device, requires_grad=True)

    for _adv_step in range(lat_steps):
        # Hook to add perturbation at the target layer
        handle = None
        hook_layer_idx = [0]

        def make_hook(d):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    return (output[0] + d,) + output[1:]
                return output + d
            return hook_fn

        # Find the target layer module
        layers = None
        for attr in ["model.layers", "gpt_neox.layers", "transformer.h"]:
            parts = attr.split(".")
            obj = model
            try:
                for p in parts:
                    obj = getattr(obj, p)
                layers = obj
                break
            except AttributeError:
                continue

        if layers is None:
            # Fallback: just use GA loss
            print(f"[WARNING] LAT: Could not find model layers structure. Falling back to gradient ascent loss.")
            print(f"[WARNING] LAT: This may happen with non-standard model architectures.")
            return -nll_loss(model, forget_batch) + nll_loss(model, retain_batch)

        handle = layers[target_lid].register_forward_hook(make_hook(delta))

        out = model(
            input_ids=forget_batch["input_ids"],
            attention_mask=forget_batch["attention_mask"],
        )
        logits = out.logits[:, :-1, :].contiguous()
        labels = forget_batch["input_ids"][:, 1:].contiguous()
        mask = forget_batch["attention_mask"][:, 1:].contiguous()
        adv_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1), reduction="none"
        )
        # Adversary wants to MINIMIZE loss (make model good at forget)
        adv_loss = -(adv_loss * mask.view(-1)).sum() / mask.sum().clamp(min=1)

        handle.remove()

        adv_loss.backward(inputs=[delta])
        with torch.no_grad():
            delta.data = delta.data - lat_eps * delta.grad.sign()
            delta.data = delta.data.clamp(-lat_eps, lat_eps)
            delta.grad.zero_()

    # --- Outer loop: train with optimal perturbation ---
    delta = delta.detach()  # freeze perturbation
    handle = layers[target_lid].register_forward_hook(make_hook(delta))

    # Forget loss WITH perturbation (gradient ascent)
    forget_loss = -nll_loss(model, forget_batch)
    handle.remove()

    # Retain loss (standard)
    retain_loss = nll_loss(model, retain_batch)

    return forget_loss + retain_loss


# ---- CB-LAT (combined) -------------------------------------------------

def cb_lat_loss(
    model,
    forget_batch: dict,
    retain_batch: dict,
    layer_ids: list[int],
    random_targets: dict,
    retain_targets: dict,
    steering_coeff: float,
    alpha: float,
    lat_eps: float,
    lat_steps: int,
) -> torch.Tensor:
    """
    CB-LAT: Circuit Breakers + Latent Adversarial Training.
    Inner loop: find adversarial perturbation that helps model recall forget data.
    Outer loop: apply CB representation rerouting with perturbation active,
                so model unlearns even under adversarial pressure.
    """
    device = next(model.parameters()).device

    # Find target layer module
    target_lid = layer_ids[len(layer_ids) // 2]
    layers = None
    for attr in ["model.layers", "gpt_neox.layers", "transformer.h"]:
        parts = attr.split(".")
        obj = model
        try:
            for p in parts:
                obj = getattr(obj, p)
            layers = obj
            break
        except AttributeError:
            continue

    if layers is None:
        # Fallback to plain CB
        print(f"[WARNING] CB-LAT: Could not find model layers structure. Falling back to plain Circuit Breakers.")
        print(f"[WARNING] CB-LAT: This may happen with non-standard model architectures.")
        return cb_loss(model, forget_batch, retain_batch, layer_ids,
                       random_targets, retain_targets, steering_coeff, alpha)

    # Get hidden shape
    with torch.no_grad():
        out = model(input_ids=forget_batch["input_ids"],
                    attention_mask=forget_batch["attention_mask"],
                    output_hidden_states=True)
        hidden_shape = out.hidden_states[target_lid + 1].shape

    # Inner loop: adversarial perturbation
    delta = torch.zeros(hidden_shape, device=device, requires_grad=True)

    def make_hook(d):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                return (output[0] + d,) + output[1:]
            return output + d
        return hook_fn

    for _ in range(lat_steps):
        handle = layers[target_lid].register_forward_hook(make_hook(delta))
        out = model(input_ids=forget_batch["input_ids"],
                    attention_mask=forget_batch["attention_mask"])
        logits = out.logits[:, :-1, :].contiguous()
        labels = forget_batch["input_ids"][:, 1:].contiguous()
        mask = forget_batch["attention_mask"][:, 1:].contiguous()
        adv_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1), reduction="none")
        adv_loss = -(adv_loss * mask.view(-1)).sum() / mask.sum().clamp(min=1)
        handle.remove()
        adv_loss.backward(inputs=[delta])
        with torch.no_grad():
            delta.data = delta.data - lat_eps * delta.grad.sign()
            delta.data = delta.data.clamp(-lat_eps, lat_eps)
            delta.grad.zero_()

    # Outer loop: CB rerouting with perturbation active
    delta = delta.detach()
    handle = layers[target_lid].register_forward_hook(make_hook(delta))

    # Get perturbed forget activations
    forget_acts = get_layer_activations(model, forget_batch, layer_ids)
    handle.remove()

    # Retain activations (no perturbation)
    retain_acts = get_layer_activations(model, retain_batch, layer_ids)

    loss = torch.tensor(0.0, device=device)
    for lid in layer_ids:
        fa = forget_acts[lid].flatten(0, 1)
        rt = random_targets[lid].unsqueeze(0).expand_as(fa) * steering_coeff
        cos_sim = F.cosine_similarity(fa, rt, dim=-1)
        loss = loss - cos_sim.mean()

        ra = retain_acts[lid]
        tr = retain_targets[lid]
        bsz = min(ra.size(0), tr.size(0))
        retain_cos = F.cosine_similarity(
            ra[:bsz].flatten(0, 1), tr[:bsz].detach().flatten(0, 1), dim=-1)
        loss = loss + alpha * (1.0 - retain_cos.mean())

    return loss


# ===================================================================
# Validation function
# ===================================================================

def run_validation(model, eval_forget_batches, eval_retain_batches, epoch, step, verbose=True):
    """Run validation on held-out data and return metrics."""
    if not eval_forget_batches or not eval_retain_batches:
        return None

    model.eval()
    with torch.no_grad():
        forget_nll = sum(nll_loss(model, b).item() for b in eval_forget_batches) / len(eval_forget_batches)
        retain_nll = sum(nll_loss(model, b).item() for b in eval_retain_batches) / len(eval_retain_batches)
    model.train()

    gap = forget_nll - retain_nll

    if verbose:
        print(f"\n[VAL @ epoch {epoch+1}, step {step}]  forget_NLL={forget_nll:.4f}  retain_NLL={retain_nll:.4f}  gap={gap:.4f}")

    return {"forget_nll": forget_nll, "retain_nll": retain_nll, "gap": gap}


# ===================================================================
# Main training loop
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-method LLM Unlearning")
    parser.add_argument("--model", required=True, help="Base model (HF ID or local path)")
    parser.add_argument(
        "--method",
        required=True,
        choices=["ga_simple", "ga", "grad_diff", "dpo", "npo", "simnpo", "rmu", "cb", "lat", "cb_lat"],
        help="Unlearning method",
    )
    parser.add_argument("--forget-data", default="data/forget.txt")
    parser.add_argument("--retain-data", default="data/retain.txt")
    parser.add_argument("--outdir", required=True, help="Output directory for unlearned model")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--beta", type=float, default=0.1, help="Inverse temp for NPO/SimNPO/DPO")
    parser.add_argument("--alpha", type=float, default=100.0, help="Retain weight for RMU")
    parser.add_argument("--steering-coeff", type=float, default=20.0, help="Steering coeff for RMU")
    parser.add_argument(
        "--layer-id",
        default="5,6,7",
        help="Comma-separated target layer indices for RMU/CB/LAT",
    )
    parser.add_argument("--forget-weight", type=float, default=1.0,
                        help="Weight for forget loss in GradDiff")
    parser.add_argument("--lat-eps", type=float, default=0.1,
                        help="Perturbation budget for LAT")
    parser.add_argument("--lat-steps", type=int, default=5,
                        help="Number of adversarial inner steps for LAT")
    parser.add_argument("--eval-split", type=float, default=0.1,
                        help="Fraction of data to hold out for evaluation (0 to disable)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping max norm (0 to disable)")
    parser.add_argument("--grad-accum-steps", type=int, default=1,
                        help="Gradient accumulation steps for effective larger batch size")
    parser.add_argument("--eval-interval", type=int, default=0,
                        help="Evaluate every N steps during training (0 to disable)")
    args = parser.parse_args()

    # ---- Setup ----
    device = resolve_device(args.device)
    pt_dtype = resolve_dtype(args.dtype, device)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    print(f"[unlearn] method={args.method}  device={device}  dtype={pt_dtype}")
    print(f"[unlearn] model={args.model}")
    print(f"[unlearn] forget={args.forget_data}  retain={args.retain_data}")

    # Log optimization settings for reproducibility
    if args.grad_accum_steps > 1:
        print(f"[unlearn] WARNING: Gradient accumulation enabled (steps={args.grad_accum_steps})")
        print(f"[unlearn]          Effective batch size = {args.batch_size * args.grad_accum_steps}")
        print(f"[unlearn]          This may affect convergence compared to true larger batch training")
    if args.grad_clip == 0:
        print(f"[unlearn] WARNING: Gradient clipping disabled - training may be less stable")

    # ---- Load tokenizer ----
    print("[unlearn] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Load model ----
    print("[unlearn] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=pt_dtype, trust_remote_code=True
    )
    model.to(device)
    model.train()

    # Enable gradient checkpointing to save memory
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # ---- Load reference model (for DPO / NPO) ----
    ref_model = None
    if args.method in ("dpo", "npo"):
        print("[unlearn] Loading reference model (frozen copy)...")
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=pt_dtype, trust_remote_code=True
        )
        ref_model.to(device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

    # ---- Load and tokenize data ----
    print("[unlearn] Tokenizing data...")
    forget_texts = load_lines(args.forget_data)
    retain_texts = load_lines(args.retain_data)
    print(f"[unlearn]   forget samples: {len(forget_texts)}")
    print(f"[unlearn]   retain samples: {len(retain_texts)}")

    # ---- Train / eval split ----
    eval_forget_batches = []
    eval_retain_batches = []
    if args.eval_split > 0:
        random.seed(args.seed)
        random.shuffle(forget_texts)
        random.shuffle(retain_texts)
        n_f_eval = max(1, int(len(forget_texts) * args.eval_split))
        n_r_eval = max(1, int(len(retain_texts) * args.eval_split))
        eval_forget_texts = forget_texts[:n_f_eval]
        eval_retain_texts = retain_texts[:n_r_eval]
        forget_texts = forget_texts[n_f_eval:]
        retain_texts = retain_texts[n_r_eval:]
        eval_forget_items = tokenize_texts(eval_forget_texts, tokenizer, args.max_length, device)
        eval_retain_items = tokenize_texts(eval_retain_texts, tokenizer, args.max_length, device)
        eval_forget_batches = make_batches(eval_forget_items, args.batch_size)
        eval_retain_batches = make_batches(eval_retain_items, args.batch_size)
        print(f"[unlearn]   eval split: {n_f_eval} forget, {n_r_eval} retain")
        print(f"[unlearn]   train: {len(forget_texts)} forget, {len(retain_texts)} retain")

    forget_items = tokenize_texts(forget_texts, tokenizer, args.max_length, device)
    retain_items = tokenize_texts(retain_texts, tokenizer, args.max_length, device)

    forget_batches = make_batches(forget_items, args.batch_size)
    retain_batches = make_batches(retain_items, args.batch_size)

    n_steps = min(len(forget_batches), len(retain_batches))
    print(f"[unlearn]   steps/epoch: {n_steps}")

    # ---- RMU / CB / LAT setup: cache retain activations + random targets ----
    layer_ids = [int(x) for x in args.layer_id.split(",")]
    random_targets = {}
    retain_act_cache: list[dict] = []

    if args.method in ("rmu", "cb", "lat", "cb_lat"):
        n_layers = model.config.num_hidden_layers
        bad = [lid for lid in layer_ids if lid < 0 or lid >= n_layers]
        if bad:
            sys.exit(
                f"[unlearn] ERROR: --layer-id values {bad} out of range for "
                f"this model ({n_layers} layers, valid 0..{n_layers - 1})"
            )
        print(f"[unlearn] {args.method.upper()}: target layers={layer_ids}  (model has {n_layers})")

    if args.method in ("rmu", "cb", "cb_lat"):
        if n_steps == 0:
            sys.exit("[unlearn] ERROR: No training steps available (n_steps=0)")

        hidden_dim = model.config.hidden_size

        # Fixed random target vectors per layer
        for lid in layer_ids:
            random_targets[lid] = torch.randn(hidden_dim, device=device, dtype=torch.float32)
            random_targets[lid] = random_targets[lid] / random_targets[lid].norm()

        # Cache clean retain activations
        print(f"[unlearn] {args.method.upper()}: caching retain activations...")
        model.eval()
        with torch.no_grad():
            for rb in retain_batches[:n_steps]:
                acts = get_layer_activations(model, rb, layer_ids)
                retain_act_cache.append({lid: a.detach().float() for lid, a in acts.items()})
        model.train()

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    total_steps = n_steps * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # ---- Training ----
    print(f"\n[unlearn] Starting training: {args.epochs} epoch(s), {n_steps} steps/epoch\n")
    global_step = 0

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        pbar = tqdm(range(n_steps), desc=f"Epoch {epoch+1}/{args.epochs}", unit="step")

        for step_idx in pbar:
            fb = forget_batches[step_idx]
            rb = retain_batches[step_idx]

            # Zero grad at accumulation boundaries (or every step if no accumulation)
            if args.grad_accum_steps > 1:
                if step_idx % args.grad_accum_steps == 0:
                    optimizer.zero_grad()
            else:
                optimizer.zero_grad()

            if args.method == "ga_simple":
                loss = ga_simple_loss(model, fb)
            elif args.method == "ga":
                loss = ga_loss(model, fb, rb)
            elif args.method == "grad_diff":
                loss = grad_diff_loss(model, fb, rb, args.forget_weight)
            elif args.method == "dpo":
                loss = dpo_loss(model, ref_model, fb, rb, args.beta)
            elif args.method == "npo":
                loss = npo_loss(model, ref_model, fb, rb, args.beta)
            elif args.method == "simnpo":
                loss = simnpo_loss(model, fb, rb, args.beta)
            elif args.method == "rmu":
                loss = rmu_loss(
                    model, fb, rb, layer_ids,
                    random_targets, retain_act_cache[step_idx % len(retain_act_cache)],
                    args.steering_coeff, args.alpha,
                )
            elif args.method == "cb":
                loss = cb_loss(
                    model, fb, rb, layer_ids,
                    random_targets, retain_act_cache[step_idx % len(retain_act_cache)],
                    args.steering_coeff, args.alpha,
                )
            elif args.method == "lat":
                loss = lat_loss(
                    model, fb, rb, layer_ids,
                    args.lat_eps, args.lat_steps,
                )
            elif args.method == "cb_lat":
                loss = cb_lat_loss(
                    model, fb, rb, layer_ids,
                    random_targets, retain_act_cache[step_idx % len(retain_act_cache)],
                    args.steering_coeff, args.alpha,
                    args.lat_eps, args.lat_steps,
                )
            else:
                raise ValueError(f"Unknown method: {args.method}")

            # Handle gradient accumulation if enabled
            if args.grad_accum_steps > 1:
                # Scale loss for gradient accumulation
                loss = loss / args.grad_accum_steps
                loss.backward()

                # Step optimizer only at accumulation boundaries or last step
                if (step_idx + 1) % args.grad_accum_steps == 0 or step_idx == n_steps - 1:
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                    optimizer.step()
                    scheduler.step()

                epoch_loss += loss.item() * args.grad_accum_steps  # Unscale for logging
            else:
                # Standard training without accumulation
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
            global_step += 1
            avg = epoch_loss / (step_idx + 1)
            # Display unscaled loss for consistency
            display_loss = loss.item() * args.grad_accum_steps if args.grad_accum_steps > 1 else loss.item()
            pbar.set_postfix(loss=f"{display_loss:.4f}", avg=f"{avg:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

            # Periodic validation
            if args.eval_interval > 0 and global_step % args.eval_interval == 0:
                run_validation(model, eval_forget_batches, eval_retain_batches, epoch, global_step)

        avg_epoch = epoch_loss / max(n_steps, 1)
        print(f"  → epoch {epoch+1} done.  avg_loss={avg_epoch:.4f}\n")

    # ---- Evaluation ----
    if eval_forget_batches and eval_retain_batches:
        print("[unlearn] Running evaluation on held-out split...")
        model.eval()
        with torch.no_grad():
            forget_nll = sum(nll_loss(model, b).item() for b in eval_forget_batches) / len(eval_forget_batches)
            retain_nll = sum(nll_loss(model, b).item() for b in eval_retain_batches) / len(eval_retain_batches)
        print(f"[unlearn] Eval  forget_NLL={forget_nll:.4f}  retain_NLL={retain_nll:.4f}")
        print(f"[unlearn] Eval  gap (forget - retain) = {forget_nll - retain_nll:.4f}")
        print(f"[unlearn]   → Good unlearning: high forget_NLL + low retain_NLL\n")

    # ---- Save ----
    print(f"[unlearn] Saving model to {args.outdir} ...")
    os.makedirs(args.outdir, exist_ok=True)

    # Disable gradient checkpointing before saving (not serializable)
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    model.save_pretrained(args.outdir)
    tokenizer.save_pretrained(args.outdir)
    print("[unlearn] Done ✓")
    print("[unlearn] ===================================================================")
    print("[unlearn] ===================================================================")


if __name__ == "__main__":
    main()
