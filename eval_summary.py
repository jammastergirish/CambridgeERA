#!/usr/bin/env python
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

"""
Compare eval results across all unlearned models.

Scans outputs/*/evals/summary.json, groups models by unlearning method,
and prints comparison tables with the base model as reference.

Usage:
    uv run eval_summary.py
    uv run eval_summary.py --sort mmlu
    uv run eval_summary.py --csv
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict


# Metrics matching high_level_summary.md
METRICS = [
    ("MMLU",          "mmlu",                      "acc,none",             "pct"),
    ("WikiPPL",       "wikitext",                   "word_perplexity,none", "float"),
    ("WMDP MCQ",      "wmdp_bio_categorized_mcqa",  "acc,none",             "pct"),
    ("┗ Robust",      "wmdp_bio_robust",            "acc,none",             "pct"),
    ("┗ Shortcut",    "wmdp_bio_shortcut",          "acc,none",             "pct"),
    ("WMDP Cloze",    "wmdp_bio_cloze_verified",    "acc_norm,none",        "pct"),
    ("WMDP Rewrite",  "wmdp_bio_robust_rewritten",  "acc,none",             "pct"),
]

ABBREV_TO_NAME = {
    "ep": "epochs", "lr": "lr", "bs": "batch", "rw": "ret_wt",
    "fw": "fgt_wt", "b": "beta", "a": "alpha", "sc": "steer",
    "ly": "layers", "le": "lat_eps", "ls": "lat_st", "wn": "noise", "wr": "lambda",
}

PARAM_RE = re.compile(
    r"(ep|lr|bs|rw|fw|sc|le|ls|wn|wr|ly|a|b)(.+?)(?=_(?:ep|lr|bs|rw|fw|sc|le|ls|wn|wr|ly|a|b)|$)"
)

BASE_MODEL = "EleutherAI_deep-ignorance-unfiltered"


def parse_model_name(model_dir: str) -> dict | None:
    """Parse model directory name into method + params."""
    basename = os.path.basename(model_dir)
    parts = basename.split("__")
    if len(parts) < 2:
        return None

    method_parts = []
    param_suffix = ""
    for i, part in enumerate(parts[1:], start=1):
        if part.startswith("ep"):
            param_suffix = "__".join(parts[i:])
            break
        method_parts.append(part)

    method = "_".join(method_parts) if method_parts else parts[1]
    params = {}
    if param_suffix:
        for match in PARAM_RE.finditer(param_suffix):
            abbrev, value = match.group(1), match.group(2)
            params[ABBREV_TO_NAME.get(abbrev, abbrev)] = value

    return {"base_model": parts[0], "method": method, "params": params, "dir": basename}


def load_metrics(evals_dir: str) -> dict:
    """Load metrics from summary.json."""
    summary_path = os.path.join(evals_dir, "summary.json")
    if not os.path.exists(summary_path):
        return {}

    with open(summary_path) as f:
        data = json.load(f)

    results = data.get("results", {})
    metrics = {}
    for label, task_key, metric_key, fmt in METRICS:
        task_data = results.get(task_key)
        if task_data and metric_key in task_data:
            metrics[label] = (task_data[metric_key], fmt)
    return metrics


def fmt_metric(value: float, fmt: str) -> str:
    return f"{value:.1%}" if fmt == "pct" else f"{value:.2f}"


def print_reference(name: str, metrics: dict):
    """Print the reference model prominently."""
    print(f"\n  ★ {name}")
    print(f"  {'─' * 50}")
    for label, _, _, _ in METRICS:
        if label in metrics:
            value, fmt = metrics[label]
            print(f"    {label:<16} {fmt_metric(value, fmt):>8}")
    print()


def print_method_table(method: str, models: list[dict], sort_key: str | None = None):
    """Print a comparison table for one method."""
    if not models:
        return

    # Collect param + metric column names
    param_names = list(dict.fromkeys(p for m in models for p in m["params"]))
    metric_names = [l for l, _, _, _ in METRICS if any(l in m["metrics"] for m in models)]

    if not metric_names:
        return

    # Sort
    if sort_key:
        for label, _, _, _ in METRICS:
            if sort_key.lower() in label.lower():
                models.sort(key=lambda m: m["metrics"].get(label, (0,))[0], reverse=True)
                break

    headers = param_names + metric_names
    widths = {h: max(len(h), 8) for h in headers}

    # Compute widths
    for m in models:
        for h in param_names:
            widths[h] = max(widths[h], len(str(m["params"].get(h, "—"))))
        for h in metric_names:
            if h in m["metrics"]:
                widths[h] = max(widths[h], len(fmt_metric(*m["metrics"][h])))

    # Print
    print(f"\n  {method.upper()}")
    print(f"  {'─' * 50}")

    header = "  "
    sep = "  "
    for h in headers:
        header += f" {h:>{widths[h]}} │"
        sep += f" {'─' * widths[h]}─┼"
    print(header.rstrip("│"))
    print(sep.rstrip("┼"))

    for m in models:
        row = "  "
        for h in param_names:
            row += f" {m['params'].get(h, '—'):>{widths[h]}} │"
        for h in metric_names:
            if h in m["metrics"]:
                row += f" {fmt_metric(*m['metrics'][h]):>{widths[h]}} │"
            else:
                row += f" {'—':>{widths[h]}} │"
        print(row.rstrip("│"))


def print_csv(all_models: dict[str, list[dict]], baselines: list[dict]):
    """Print all results as CSV."""
    all_params = sorted(set(p for ms in all_models.values() for m in ms for p in m["params"]))
    metric_labels = [l for l, _, _, _ in METRICS]

    print(",".join(["model", "method"] + all_params + metric_labels))

    for b in baselines:
        row = [b["name"], "baseline"] + [""] * len(all_params)
        for l in metric_labels:
            row.append(f"{b['metrics'][l][0]:.4f}" if l in b["metrics"] else "")
        print(",".join(row))

    for method, models in sorted(all_models.items()):
        for m in models:
            row = [m["dir"], method]
            for p in all_params:
                row.append(m["params"].get(p, ""))
            for l in metric_labels:
                row.append(f"{m['metrics'][l][0]:.4f}" if l in m["metrics"] else "")
            print(",".join(row))


def main():
    parser = argparse.ArgumentParser(description="Compare eval results across unlearned models.")
    parser.add_argument("--outputs-dir", default="outputs", help="Root outputs directory")
    parser.add_argument("--sort", default=None, help="Sort by metric (e.g. mmlu, wmdp)")
    parser.add_argument("--csv", action="store_true", help="Output as CSV")
    args = parser.parse_args()

    pattern = os.path.join(args.outputs_dir, "*", "evals", "summary.json")
    summary_files = sorted(glob.glob(pattern))

    if not summary_files:
        print(f"No eval results found in {args.outputs_dir}/*/evals/summary.json")
        sys.exit(1)

    by_method: dict[str, list[dict]] = defaultdict(list)
    baselines = []
    base_model_metrics = None

    for sf in summary_files:
        evals_dir = os.path.dirname(sf)
        model_dir = os.path.dirname(evals_dir)
        metrics = load_metrics(evals_dir)
        if not metrics:
            continue

        basename = os.path.basename(model_dir)

        # Check if this is the base model
        if basename == BASE_MODEL:
            base_model_metrics = metrics

        parsed = parse_model_name(model_dir)
        if parsed and parsed["params"]:
            parsed["metrics"] = metrics
            by_method[parsed["method"]].append(parsed)
        else:
            baselines.append({"name": basename, "metrics": metrics})

    if args.csv:
        print_csv(by_method, baselines)
        return

    # Print base model front and center
    if base_model_metrics:
        print_reference("EleutherAI/deep-ignorance-unfiltered (BASE)", base_model_metrics)

    # Print other baselines (filtered, etc.)
    for b in baselines:
        if b["name"] != BASE_MODEL:
            print_reference(b["name"], b["metrics"])

    # Print each method
    for method in sorted(by_method.keys()):
        print_method_table(method, by_method[method], sort_key=args.sort)

    total = sum(len(v) for v in by_method.values()) + len(baselines)
    print(f"\n  {total} models · {len(by_method)} methods · {len(baselines)} baselines\n")


if __name__ == "__main__":
    main()
