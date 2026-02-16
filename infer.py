# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch",
#     "transformers",
#     "accelerate",
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
"""

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
        mapping = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
        if dtype not in mapping:
            raise ValueError(f"Unknown dtype '{dtype}'. Use auto|fp32|fp16|bf16")
        return mapping[dtype]


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


def main():
    parser = argparse.ArgumentParser(description="Run inference on a model")
    parser.add_argument("--model", required=True, help="Model A: HF ID or local path")
    parser.add_argument("--model-b", default=None, help="Optional Model B for side-by-side comparison")
    parser.add_argument("--prompt", default=None, help="Prompt to run (omit for --interactive)")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode: keep prompting")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--greedy", action="store_true", help="Greedy decoding (deterministic)")
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
