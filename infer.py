# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch",
#     "transformers",
#     "accelerate",
#     "pandas",
#     "wandb",
#     "dotenv",
# ]
# ///
"""
Simple inference script for comparing model outputs.

Works with both HuggingFace model IDs and local unlearned models.

Usage:
  # HuggingFace model
  uv run infer.py --model EleutherAI/deep-ignorance-unfiltered --prompt "What is biotin?"

  # Local unlearned model
  uv run infer.py --model unlearned_models/EleutherAI_deep-ignorance-unfiltered__ga --prompt "What is biotin?"

  # Interactive mode (keeps prompting)
  uv run infer.py --model EleutherAI/deep-ignorance-unfiltered --interactive

  # Compare two models side by side
  uv run infer.py --model EleutherAI/deep-ignorance-unfiltered --model-b EleutherAI/deep-ignorance-unfiltered-cb-lat --prompt "What is biotin?"

  # Sweep all unlearned models + base models
  uv run infer.py --sweep --prompt "What is biotin?"
  uv run infer.py --sweep --include-base --prompt "What is biotin?"
  uv run infer.py --sweep --models-dir path/to/models --prompt "What is biotin?"
"""

import argparse
import hashlib
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from utils import resolve_device, resolve_dtype, init_wandb, finish_wandb
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
        mapping = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
        if dtype not in mapping:
            raise ValueError(f"Unknown dtype '{dtype}'. Use auto|fp32|fp16|bf16")
        return mapping[dtype]

    def init_wandb(*a, **kw):
        return None

    def finish_wandb():
        pass


def load_model(model_path: str, device: str, pt_dtype: torch.dtype):
    """Load a model and tokenizer from HF ID or local path."""
    print(f"[infer] Loading: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=pt_dtype, trust_remote_code=True
    )
    model.to(device)
    model.eval()
    print(f"[infer] Loaded on {device} ({pt_dtype})")
    return model, tokenizer


@torch.no_grad()
def generate(model, tokenizer, prompt: str, device: str,
             max_new_tokens: int = 200, temperature: float = 0.7,
             top_p: float = 0.9, do_sample: bool = True) -> str:
    """Generate text from a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature if do_sample else 1.0,
        top_p=top_p if do_sample else 1.0,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
    )
    # Decode only the new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def run_prompt(prompt, model_a, tok_a, label_a, device, args,
               model_b=None, tok_b=None, label_b=None):
    """Run a prompt through one or two models and print results."""
    print(f"\n{'─' * 60}")
    print(f"Prompt: {prompt}")
    print(f"{'─' * 60}")

    output_a = generate(model_a, tok_a, prompt, device,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        do_sample=not args.greedy)
    print(f"\n[{label_a}]")
    print(output_a)

    if model_b is not None:
        output_b = generate(model_b, tok_b, prompt, device,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            do_sample=not args.greedy)
        print(f"\n[{label_b}]")
        print(output_b)

    print()


# ---- Sweep helpers -----------------------------------------------------------

def discover_models(models_dir: str) -> list[str]:
    """Return sorted list of local model directories (those containing config.json)."""
    models = []
    if not os.path.isdir(models_dir):
        return models
    for entry in sorted(os.listdir(models_dir)):
        candidate = os.path.join(models_dir, entry)
        if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "config.json")):
            models.append(candidate)
    return models


def prompt_hash(prompt: str) -> str:
    """Short deterministic hash of the prompt for the output filename."""
    return hashlib.sha256(prompt.encode()).hexdigest()[:12]


def run_sweep(args):
    """Run the same prompt through every model in models_dir and save a CSV."""
    import pandas as pd

    device = resolve_device(args.device)
    pt_dtype = resolve_dtype(args.dtype, device)

    # Collect model paths
    model_paths: list[str] = []

    # Optionally include well-known base / filtered / unlearned HF models
    if args.include_base:
        from dotenv import load_dotenv
        load_dotenv()
        base_models = [
            "EleutherAI/deep-ignorance-unfiltered",
            "EleutherAI/deep-ignorance-e2e-strong-filter",
            "EleutherAI/deep-ignorance-unfiltered-cb-lat",
        ]
        model_paths.extend(base_models)

    # Auto-discover local unlearned models
    local_models = discover_models(args.models_dir)
    if not local_models and not model_paths:
        print(f"[infer] No models found in {args.models_dir} and --include-base not set.", file=sys.stderr)
        sys.exit(1)
    model_paths.extend(local_models)

    print(f"[infer] Sweep: {len(model_paths)} model(s)")
    for p in model_paths:
        print(f"  • {p}")

    # Generate outputs
    rows = []
    for model_path in model_paths:
        label = model_path.rstrip("/").split("/")[-1]
        try:
            model, tokenizer = load_model(model_path, device, pt_dtype)
            output = generate(model, tokenizer, args.prompt, device,
                              max_new_tokens=args.max_new_tokens,
                              temperature=args.temperature,
                              top_p=args.top_p,
                              do_sample=not args.greedy)
            rows.append({"model": label, "model_path": model_path, "output": output})
            print(f"\n[{label}]")
            print(output)
        except Exception as e:
            print(f"\n[{label}] ERROR: {e}")
            rows.append({"model": label, "model_path": model_path, "output": f"ERROR: {e}"})
        finally:
            # Free GPU memory before loading next model
            del model, tokenizer
            if device == "cuda":
                torch.cuda.empty_cache()

    # Save CSV
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    filename = f"{prompt_hash(args.prompt)}.csv"
    csv_path = os.path.join(outdir, filename)

    df = pd.DataFrame(rows)
    df.insert(0, "prompt", args.prompt)
    df.to_csv(csv_path, index=False)
    print(f"\n[infer] ✓ Saved {len(rows)} outputs to {csv_path}")

    # W&B logging
    try:
        import wandb
        if wandb.run is not None:
            wandb.log({"inference_sweep": wandb.Table(dataframe=df)})
    except Exception:
        pass

    return csv_path


def main():
    parser = argparse.ArgumentParser(description="Run inference on a model")
    parser.add_argument("--model", default=None, help="Model A: HF ID or local path")
    parser.add_argument("--model-b", default=None, help="Optional Model B for side-by-side comparison")
    parser.add_argument("--prompt", default=None, help="Prompt to run (omit for --interactive)")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode: keep prompting")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--greedy", action="store_true", help="Greedy decoding (deterministic)")

    # Sweep mode
    parser.add_argument("--sweep", action="store_true",
                        help="Run prompt through all models in --models-dir and save CSV")
    parser.add_argument("--models-dir", default="unlearned_models",
                        help="Directory containing unlearned model folders (default: unlearned_models)")
    parser.add_argument("--include-base", action="store_true",
                        help="Also include the base HF models (unfiltered, filtered, cb-lat)")
    parser.add_argument("--outdir", default="outputs/inference",
                        help="Output directory for sweep CSVs (default: outputs/inference)")

    args = parser.parse_args()
    init_wandb("infer", args)

    # Sweep mode
    if args.sweep:
        if not args.prompt:
            print("Error: --sweep requires --prompt", file=sys.stderr)
            sys.exit(1)
        run_sweep(args)
        finish_wandb()
        return

    # Original single/dual model modes
    if not args.model:
        print("Error: provide --model (or use --sweep)", file=sys.stderr)
        sys.exit(1)

    if not args.prompt and not args.interactive:
        print("Error: provide --prompt or --interactive", file=sys.stderr)
        sys.exit(1)

    device = resolve_device(args.device)
    pt_dtype = resolve_dtype(args.dtype, device)

    # Load model(s)
    model_a, tok_a = load_model(args.model, device, pt_dtype)
    label_a = args.model.rstrip("/").split("/")[-1]

    model_b, tok_b, label_b = None, None, None
    if args.model_b:
        model_b, tok_b = load_model(args.model_b, device, pt_dtype)
        label_b = args.model_b.rstrip("/").split("/")[-1]

    # Single prompt mode
    if args.prompt:
        run_prompt(args.prompt, model_a, tok_a, label_a, device, args,
                   model_b, tok_b, label_b)
        finish_wandb()
        return

    # Interactive mode
    print("\n[infer] Interactive mode — type 'quit' to exit\n")
    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not prompt or prompt.lower() in ("quit", "exit", "q"):
            break
        run_prompt(prompt, model_a, tok_a, label_a, device, args,
                   model_b, tok_b, label_b)
    finish_wandb()


if __name__ == "__main__":
    main()

