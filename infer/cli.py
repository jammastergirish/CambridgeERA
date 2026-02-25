# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch",
#     "transformers",
#     "accelerate",
# ]
# ///
"""
Quick inference on any HuggingFace model.

Usage:
  uv run infer/cli.py --model EleutherAI/deep-ignorance-unfiltered --prompt "What is the capital of France?"
  uv run infer/cli.py --model EleutherAI/deep-ignorance-unfiltered --prompt "..." --device cuda:1
"""

import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]


def load_model(model_id: str, device: str = "auto", dtype: str = "auto"):
    """Load model and tokenizer, picking the optimal GPU automatically."""
    device = resolve_device(device)
    pt_dtype = resolve_dtype(dtype, device)

    print(f"[infer] device={device}  dtype={pt_dtype}")
    print(f"[infer] Loading model {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # On CUDA, use device_map="auto" so HuggingFace/accelerate spreads the model
    # across available GPUs, automatically placing layers on the GPU with the most
    # free memory — equivalent to what unlearn.py does with --device auto.
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=pt_dtype,
            trust_remote_code=True,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=pt_dtype,
            trust_remote_code=True,
        )
        model.to(device)

    model.eval()
    return model, tokenizer, device


@torch.no_grad()
def generate(model, tokenizer, prompt: str, device: str, max_tokens: int = 200) -> str:
    """Generate a response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Run inference on a HuggingFace model")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--prompt", required=True, help="Prompt to run")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument(
        "--device", default="auto",
        help="Device to use: auto (default), cuda, cuda:N, mps, cpu"
    )
    parser.add_argument(
        "--dtype", default="auto",
        help="dtype: auto (default), fp32, fp16, bf16"
    )
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model, args.device, args.dtype)

    print(f"\n> {args.prompt}\n")
    print(generate(model, tokenizer, args.prompt, device, args.max_tokens))


if __name__ == "__main__":
    main()
