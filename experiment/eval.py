#!/usr/bin/env python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
#   "transformers",
#   "accelerate",
#   "lm-eval",
#   "wandb",
# ]
# ///

"""
Benchmark evaluation using EleutherAI's lm-evaluation-harness.

Automatically uses device_map='auto' with CPU offloading to handle large models
and prevent OOM errors. Selects the GPU with most free memory when available.

Runs multiple benchmarks in a single pass (shared model load):
  Standard (built-in to lm-eval):
    - MMLU          general knowledge / capabilities

  Custom (vendored from EleutherAI/deep-ignorance):
    - wmdp_bio_robust_rewritten   robust MCQA with rewritten questions
    - wmdp_bio_cloze_verified     perplexity-based (no other choices visible)
    - wmdp_bio_categorized_mcqa   MCQA broken down by threat category

Output structure (e.g. --outdir outputs/model_name/evals):
  outputs/model_name/evals/
    summary.json                    combined results for all tasks
    mmlu.json                       per-task result files
    wmdp_bio_robust_rewritten.json
    wmdp_bio_cloze_verified.json
    ...

Usage:
  # Standard usage (automatically handles memory management)
  uv run experiment/eval.py --model EleutherAI/deep-ignorance-unfiltered \
      --outdir outputs/base/evals

  # Subset of tasks
  uv run experiment/eval.py --model EleutherAI/deep-ignorance-unfiltered \
      --outdir /tmp/test --tasks mmlu wmdp_bio_cloze_verified

  # Limit samples for a quick sanity check
  uv run experiment/eval.py --model EleutherAI/deep-ignorance-unfiltered \
      --outdir /tmp/test --limit 20
"""

import argparse
import json
import os
import sys
import gc

import lm_eval
from lm_eval.tasks import TaskManager
import torch

# Add project root to path so we can import utils
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)
from utils import model_outdir, pick_best_gpu


DEFAULT_TASKS = [
    "mmlu",                          # general capabilities (built-in)
    "wikitext",                      # perplexity (built-in)
    "wmdp_bio_robust_rewritten",     # robust MCQA (custom)
    "wmdp_bio_cloze_verified",       # cloze / perplexity-based (custom)
    "wmdp_bio_categorized_mcqa",     # MCQA by threat category (custom)
]

# Resolve path to vendored custom task YAMLs (lm_eval_tasks/ in project root)
_CUSTOM_TASKS_DIR = os.path.join(_PROJECT_ROOT, "lm_eval_tasks")


def main():
    parser = argparse.ArgumentParser(description="Run lm-evaluation-harness benchmarks.")
    parser.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--device", default="auto", help="Device (auto/cuda/mps/cpu)")
    parser.add_argument("--dtype", default="auto", help="Dtype (auto/float16/bfloat16/float32)")
    parser.add_argument("--outdir", default=None,
                        help="Directory to save results (default: outputs/<model>/evals)")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS,
                        help=f"Benchmarks to run (default: {' '.join(DEFAULT_TASKS)})")
    parser.add_argument("--include-path", default=_CUSTOM_TASKS_DIR,
                        help="Path to custom lm-eval task YAMLs (default: lm_eval_tasks/)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max samples per task (default: full benchmark)")
    parser.add_argument("--batch-size", default="auto", help="Batch size (default: auto)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", type=str, default=None, help="W&B project to log results to")
    parser.add_argument("--wandb-name", type=str, default=None, help="W&B run name")
    args = parser.parse_args()

    # Auto-derive outdir from model name if not specified
    if args.outdir is None:
        args.outdir = model_outdir(args.model, suffix="evals")

    # Build model_args string for lm-eval with memory-efficient defaults
    model_args = f"pretrained={args.model}"

    # Always use device_map='auto' for automatic memory management
    model_args += ",device_map=auto"

    # Always set up CPU offloading to handle large models
    offload_folder = "/tmp/offload"
    model_args += f",offload_folder={offload_folder}"
    os.makedirs(offload_folder, exist_ok=True)

    # Always use low CPU memory usage for efficiency
    model_args += ",low_cpu_mem_usage=True"

    # Add dtype if specified
    if args.dtype != "auto":
        model_args += f",dtype={args.dtype}"

    # Clear memory before starting evaluation
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Use pick_best_gpu() to select the GPU with most free memory,
    # then let device_map='auto' use it with CPU offloading as needed
    if torch.cuda.is_available():
        best_gpu = pick_best_gpu()
        # Restrict to the best GPU for device_map to use primarily
        os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
        print(f"[eval] Selected GPU {best_gpu} (most free memory)")

    # Since we're using device_map='auto', we don't specify a device to lm-eval
    device = None  # Let device_map handle everything

    # Build task manager with custom task path so vendored YAMLs are discovered
    task_manager = TaskManager(include_path=args.include_path)

    print(f"[eval] Model:   {args.model}")
    print(f"[eval] Device:  auto (device_map with CPU offloading)")
    print(f"[eval] Tasks:   {', '.join(args.tasks)}")
    print(f"[eval] Custom:  {args.include_path}")
    if args.limit:
        print(f"[eval] Limit:   {args.limit} samples per task")
    print()

    # Run all tasks in one call (model loaded once, tasks evaluated sequentially)
    try:
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
            task_manager=task_manager,
        )
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n[eval] ERROR: CUDA OOM despite device_map='auto'. Try:")
        print(f"  1. Reduce batch size: --batch-size 1")
        print(f"  2. Kill other GPU processes and retry")
        print(f"  3. Increase system RAM for CPU offloading")
        raise e
    finally:
        # Clear memory after evaluation
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

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

    # Per-task JSON files
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

    # High-level markdown summary
    _write_high_level_summary(results["results"], args.model, args.outdir)

    print(f"\n[eval] ✓ Results saved to {args.outdir}/")
    for task_name in sorted(results["results"]):
        print(f"         {task_name}.json")
    print(f"         summary.json")
    print(f"         high_level_summary.md")

    # Log to W&B if requested
    if args.wandb_project:
        try:
            import wandb
            run_name = args.wandb_name or args.model
            wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
            flat = {}
            for task_name, task_results in results["results"].items():
                for metric_key, value in task_results.items():
                    if metric_key.endswith(",none") and not metric_key.startswith("alias"):
                        clean_metric = metric_key.replace(",none", "")
                        flat[f"eval_bench/{task_name}/{clean_metric}"] = value
            wandb.log(flat)
            wandb.summary.update(flat)
            wandb.finish()
            print(f"[eval] ✓ Metrics logged to Weights & Biases (project: {args.wandb_project}, run: {run_name})")
        except Exception as e:
            print(f"[eval] WARNING: Failed to log to W&B: {e}")


def _write_high_level_summary(results: dict, model: str, outdir: str) -> None:
    """Write a concise markdown table with headline metrics."""
    # Rows: (label, task_key, metric_key)
    rows = [
        ("MMLU",                       "mmlu",                       "acc,none"),
        ("WikiText (word perplexity)",  "wikitext",                   "word_perplexity,none"),
        ("WMDP Bio (categorized MCQ)",  "wmdp_bio_categorized_mcqa",  "acc,none"),
        ("↳ Robust subset",            "wmdp_bio_robust",            "acc,none"),
        ("↳ Shortcut subset",          "wmdp_bio_shortcut",          "acc,none"),
        ("WMDP Bio (cloze verified)",   "wmdp_bio_cloze_verified",    "acc_norm,none"),
        ("WMDP Bio (robust rewritten)", "wmdp_bio_robust_rewritten",  "acc,none"),
    ]

    lines = [
        f"# Eval Summary: `{model}`\n",
        "| Benchmark | Score |",
        "|-----------|-------|",
    ]
    for label, task_key, metric_key in rows:
        task_data = results.get(task_key)
        if task_data is None:
            continue
        value = task_data.get(metric_key)
        if value is None:
            continue
        if metric_key.startswith("word_perplexity"):
            lines.append(f"| {label} | {value:.2f} |")
        else:
            lines.append(f"| {label} | {value:.1%} |")

    md_path = os.path.join(outdir, "high_level_summary.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
