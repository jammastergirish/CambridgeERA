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
"""

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_id: str):
    """Load model and tokenizer, auto-detecting device and dtype."""
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    print(f"Loading {model_id} on {device} ({dtype})")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=True
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
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model)

    print(f"\n> {args.prompt}\n")
    print(generate(model, tokenizer, args.prompt, device, args.max_tokens))


if __name__ == "__main__":
    main()
