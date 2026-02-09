#!/usr/bin/env python
"""
Experimental sanity checks for unlearning experiments.
Run this before starting experiments to catch common issues.
"""

import os
import sys
from pathlib import Path

def check_data_overlap():
    """Check if forget and retain sets have overlapping samples."""
    forget_path = "data/forget.txt"
    retain_path = "data/retain.txt"

    if not os.path.exists(forget_path) or not os.path.exists(retain_path):
        print("⚠️  Data files not found. Run create_datasets.py first.")
        return False

    with open(forget_path) as f:
        forget_lines = set(line.strip() for line in f if line.strip())

    with open(retain_path) as f:
        retain_lines = set(line.strip() for line in f if line.strip())

    overlap = forget_lines & retain_lines

    if overlap:
        print(f"❌ Found {len(overlap)} overlapping samples between forget and retain!")
        print(f"   First overlap: {list(overlap)[0][:100]}...")
        return False
    else:
        print(f"✅ No overlap: {len(forget_lines)} forget, {len(retain_lines)} retain samples")
        return True

def check_data_balance():
    """Check if datasets are roughly balanced."""
    forget_path = "data/forget.txt"
    retain_path = "data/retain.txt"

    if not os.path.exists(forget_path) or not os.path.exists(retain_path):
        return False

    with open(forget_path) as f:
        n_forget = sum(1 for line in f if line.strip())

    with open(retain_path) as f:
        n_retain = sum(1 for line in f if line.strip())

    ratio = max(n_forget, n_retain) / max(min(n_forget, n_retain), 1)

    if ratio > 2:
        print(f"⚠️  Unbalanced data: forget={n_forget}, retain={n_retain} (ratio={ratio:.1f})")
        print(f"   Consider balancing or using weighted sampling")
        return False
    else:
        print(f"✅ Balanced data: forget={n_forget}, retain={n_retain} (ratio={ratio:.1f})")
        return True

def check_gpu_determinism():
    """Check if GPU determinism is enabled."""
    import torch

    if torch.cuda.is_available():
        # Check if deterministic algorithms are enabled
        if not torch.backends.cudnn.deterministic:
            print("⚠️  CUDNN deterministic mode disabled. Results may vary slightly.")
            print("   To enable: torch.backends.cudnn.deterministic = True")

        # Check benchmark mode
        if torch.backends.cudnn.benchmark:
            print("⚠️  CUDNN benchmark mode enabled. This can reduce reproducibility.")
            print("   To disable: torch.backends.cudnn.benchmark = False")

    print(f"✅ PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"✅ CUDA: {torch.version.cuda}")
        print(f"✅ GPU: {torch.cuda.get_device_name()}")
    else:
        print("ℹ️  No CUDA available")

    return True

def check_output_structure():
    """Check if output directory structure is clean."""
    if os.path.exists("outputs") and len(os.listdir("outputs")) > 10:
        print("⚠️  Output directory has many experiments. Consider archiving old runs.")
    else:
        print("✅ Output directory is clean")
    return True

def check_dependencies():
    """Check if all required packages are installed."""
    required = ["torch", "transformers", "numpy", "tqdm"]
    missing = []

    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        return False
    else:
        print(f"✅ All core packages installed")
        return True

def suggest_test_run():
    """Suggest a minimal test run command."""
    print("\n📝 Suggested test run to verify setup:")
    print("   uv run --script unlearn.py \\")
    print("     --model EleutherAI/pythia-70m \\  # Small model for testing")
    print("     --method ga \\")
    print("     --epochs 1 \\")
    print("     --eval-split 0.1 \\")
    print("     --batch-size 2 \\")
    print("     --outdir outputs/test_run")
    print("\n   This should complete in <5 minutes on most systems")

def main():
    print("🔍 Running experimental sanity checks...\n")

    checks = [
        ("Data overlap", check_data_overlap),
        ("Data balance", check_data_balance),
        ("Dependencies", check_dependencies),
        ("GPU determinism", check_gpu_determinism),
        ("Output structure", check_output_structure),
    ]

    all_passed = True
    for name, check_func in checks:
        print(f"\n--- {name} ---")
        passed = check_func()
        if not passed:
            all_passed = False

    print("\n" + "="*50)
    if all_passed:
        print("✅ All checks passed! Ready for experiments.")
        suggest_test_run()
    else:
        print("❌ Some issues found. Please address them before running experiments.")
        sys.exit(1)

if __name__ == "__main__":
    main()