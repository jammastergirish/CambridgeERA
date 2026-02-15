#!/usr/bin/env python
# /// script
# dependencies = [
#   "torch",
#   "transformers",
# ]
# ///

"""Quick test to see if activation extraction works."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/deep-ignorance-unfiltered")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/deep-ignorance-unfiltered")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Test on CPU first
device = "cpu"
model = model.to(device)
model.eval()

print(f"Model on {device}")
print(f"Model has gpt_neox.layers: {hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers')}")
if hasattr(model, 'gpt_neox'):
    print(f"Number of layers: {len(model.gpt_neox.layers)}")

# Test with single text
text = "This is a test sentence."
print(f"\nTesting with text: {text}")

inputs = tokenizer(text, return_tensors="pt").to(device)
print(f"Tokenized input shape: {inputs['input_ids'].shape}")

with torch.no_grad():
    print("Running forward pass...")
    outputs = model(**inputs, output_hidden_states=True)
    print(f"Number of hidden states: {len(outputs.hidden_states)}")

    # Try to access layer 0
    layer_0 = outputs.hidden_states[0]
    print(f"Layer 0 shape: {layer_0.shape}")

    # Try mean pooling
    pooled = layer_0.mean(dim=1)
    print(f"Pooled shape: {pooled.shape}")

print("\n✓ Basic test passed on CPU!")

# Now test on MPS if available
if torch.backends.mps.is_available():
    print("\nTesting on MPS...")
    device = "mps"
    model = model.to(device)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        layer_0 = outputs.hidden_states[0]
        pooled = layer_0.mean(dim=1).cpu()
        print(f"MPS pooled shape: {pooled.shape}")

    print("✓ MPS test passed!")
else:
    print("\nMPS not available, skipping MPS test")