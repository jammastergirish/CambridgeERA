#!/usr/bin/env python
# /// script
# dependencies = [
#   "torch",
# ]
# ///

import torch
from utils import resolve_device, resolve_dtype

device = resolve_device('auto')
dtype = resolve_dtype('auto', device)

print(f"Device selected: {device}")
print(f"Dtype selected: {dtype}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
print(f"CUDA available: {torch.cuda.is_available()}")