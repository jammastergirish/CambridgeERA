#!/usr/bin/env python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
#   "numpy",
#   "safetensors",
#   "pytest",
#   "matplotlib",
#   "tqdm",
#   "transformers",
#   "datasets",
#   "huggingface_hub",
#   "wandb",
#   "pandas",
#   "scikit-learn",
#   "scipy",
# ]
# ///

"""
Test suite runner — executes all tests via pytest.

Usage:
    uv run tests/run_tests.py          # all tests
    uv run tests/run_tests.py -k utils # just utils tests
    uv run tests/run_tests.py -v       # verbose
"""

import sys
import pytest

if __name__ == "__main__":
    sys.exit(pytest.main([
        "tests/",
        "-v",
        "--tb=short",
    ] + sys.argv[1:]))
