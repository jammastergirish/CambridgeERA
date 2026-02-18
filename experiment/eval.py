#!/usr/bin/env python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
#   "transformers",
#   "lm-eval",
# ]
# ///

"""
Benchmark evaluation using EleutherAI's lm-evaluation-harness.

Runs multiple benchmarks in a single pass (shared model load):
  - MMLU          general knowledge / capabilities
  - WMDP          hazardous-knowledge proxy (bio, cyber, chem)
  - HellaSwag     commonsense reasoning
  - TruthfulQA    truthfulness (mc2)

Output structure (e.g. --outdir outputs/model_name/evals):
  outputs/model_name/evals/
    summary.json          combined results for all tasks
    mmlu.json             per-task result files
    wmdp.json
    hellaswag.json
    truthfulqa_mc2.json

Usage:
  uv run experiment/eval.py --model EleutherAI/deep-ignorance-unfiltered \
      --outdir outputs/base/evals

  # Subset of tasks
  uv run experiment/eval.py --model EleutherAI/deep-ignorance-unfiltered \
      --outdir /tmp/test --tasks mmlu wmdp

  # Limit samples for a quick sanity check
  uv run experiment/eval.py --model EleutherAI/deep-ignorance-unfiltered \
      --outdir /tmp/test --limit 20
"""

import argparse
import json
import os

import lm_eval


DEFAULT_TASKS = ["mmlu", "wmdp", "hellaswag", "truthfulqa_mc2"]


def main():
    parser = argparse.ArgumentParser(description="Run lm-evaluation-harness benchmarks.")
    parser.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--device", default="auto", help="Device (auto/cuda/mps/cpu)")
    parser.add_argument("--dtype", default="auto", help="Dtype (auto/float16/bfloat16/float32)")
    parser.add_argument("--outdir", required=True, help="Directory to save results")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS,
                        help=f"Benchmarks to run (default: {' '.join(DEFAULT_TASKS)})")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max samples per task (default: full benchmark)")
    parser.add_argument("--batch-size", default="auto", help="Batch size (default: auto)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Build model_args string for lm-eval
    model_args = f"pretrained={args.model}"
    if args.dtype != "auto":
        model_args += f",dtype={args.dtype}"

    # Resolve device
    device = args.device
    if device == "auto":
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"[eval] Model:   {args.model}")
    print(f"[eval] Device:  {device}")
    print(f"[eval] Tasks:   {', '.join(args.tasks)}")
    if args.limit:
        print(f"[eval] Limit:   {args.limit} samples per task")
    print()

    # Run all tasks in one call (model loaded once, tasks evaluated sequentially)
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=args.tasks,
        batch_size=args.batch_size,
        device=device,
        limit=args.limit,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
    )

    # Print summary table
    print("\n" + "=" * 60)
    print(f"{'Task':<30} {'Metric':<20} {'Value':>8}")
    print("=" * 60)
    for task_name, task_results in sorted(results["results"].items()):
        for metric, value in sorted(task_results.items()):
            if metric.endswith(",none") and not metric.startswith("alias"):
                clean_metric = metric.replace(",none", "")
                if isinstance(value, float):
                    print(f"{task_name:<30} {clean_metric:<20} {value:>8.4f}")
    print("=" * 60)

    # Save results
    os.makedirs(args.outdir, exist_ok=True)

    # Per-task JSON files  (e.g. mmlu.json, wmdp.json)
    for task_name, task_results in results["results"].items():
        task_path = os.path.join(args.outdir, f"{task_name}.json")
        with open(task_path, "w") as f:
            json.dump({"model": args.model, "task": task_name, **task_results},
                      f, indent=2, default=str)

    # Combined summary
    save_data = {
        "model": args.model,
        "tasks": args.tasks,
        "results": results["results"],
        "configs": {k: str(v) for k, v in results.get("configs", {}).items()},
    }
    summary_path = os.path.join(args.outdir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\n[eval] ✓ Results saved to {args.outdir}/")
    for task_name in sorted(results["results"]):
        print(f"         {task_name}.json")
    print(f"         summary.json")


if __name__ == "__main__":
    main()
