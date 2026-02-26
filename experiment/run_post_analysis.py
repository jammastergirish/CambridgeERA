#!/usr/bin/env python
# /// script
# dependencies = [
#   "pandas",
#   "numpy",
#   "matplotlib",
#   "wandb",
# ]
# ///

"""
Post-hoc analysis runner for factor decomposition and Goldilocks curves.

This script demonstrates how to use the new analysis tools on existing
unlearning experimental results. It can be run after completing hyperparameter
sweeps to understand what factors drive unlearning effectiveness.

Usage:
    # Run factor decomposition on sweep results
    uv run experiment/run_post_analysis.py --mode factor --results-csv sweep_results.csv

    # Run Goldilocks curve analysis
    uv run experiment/run_post_analysis.py --mode goldilocks --results-csv sweep_results.csv

    # Run both analyses
    uv run experiment/run_post_analysis.py --mode both --results-csv sweep_results.csv
"""

import argparse
import json
import os
import sys
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

# Import the new analysis modules
from experiment.factor_decomposition_analysis import run_factor_analysis
from experiment.goldilocks_curve_analysis import run_goldilocks_analysis


def detect_experimental_factors(df: pd.DataFrame) -> List[str]:
    """Auto-detect likely experimental factors from DataFrame columns."""
    hyperparameter_patterns = [
        'lr', 'learning_rate', 'epochs', 'batch_size', 'method',
        'alpha', 'beta', 'forget_weight', 'retain_weight', 'steering_coeff',
        'lat_eps', 'lat_steps', 'wt_noise_std', 'wt_reg_lambda'
    ]

    detected = []
    for col in df.columns:
        col_lower = col.lower()
        for pattern in hyperparameter_patterns:
            if pattern in col_lower:
                detected.append(col)
                break

    return detected


def detect_target_metrics(df: pd.DataFrame) -> List[str]:
    """Auto-detect likely target metrics from DataFrame columns."""
    target_patterns = [
        'mmlu', 'wmdp', 'accuracy', 'loss', 'perplexity', 'score',
        'hellaswag', 'truthfulqa', 'eval'
    ]

    detected = []
    for col in df.columns:
        col_lower = col.lower()
        for pattern in target_patterns:
            if pattern in col_lower and 'acc' in col_lower:
                detected.append(col)
                break

    return detected


def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc analysis for unlearning experimental results"
    )
    parser.add_argument("--mode", choices=["factor", "goldilocks", "both"],
                       default="both", help="Analysis mode to run")
    parser.add_argument("--results-csv", required=True,
                       help="Path to experimental results CSV")
    parser.add_argument("--outdir", default="post_analysis",
                       help="Output directory")
    parser.add_argument("--hyperparameter-cols", nargs="*",
                       help="Manual specification of hyperparameter columns")
    parser.add_argument("--target-cols", nargs="*",
                       help="Manual specification of target columns")
    parser.add_argument("--wandb-project", help="W&B project name")

    args = parser.parse_args()

    # Load and inspect data
    print(f"Loading experimental results from {args.results_csv}")
    df = pd.read_csv(args.results_csv)
    print(f"Loaded {len(df)} experimental runs with {len(df.columns)} columns")

    # Auto-detect or use manual specification
    if args.hyperparameter_cols:
        hyperparameter_cols = args.hyperparameter_cols
    else:
        hyperparameter_cols = detect_experimental_factors(df)
        print(f"Auto-detected hyperparameter columns: {hyperparameter_cols}")

    if args.target_cols:
        target_cols = args.target_cols
    else:
        target_cols = detect_target_metrics(df)
        print(f"Auto-detected target columns: {target_cols}")

    if not hyperparameter_cols:
        print("Warning: No hyperparameter columns found. Please specify --hyperparameter-cols")
        return

    if not target_cols:
        print("Warning: No target columns found. Please specify --target-cols")
        return

    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)

    # Run analyses
    if args.mode in ["factor", "both"]:
        print("\n" + "="*50)
        print("RUNNING FACTOR DECOMPOSITION ANALYSIS")
        print("="*50)

        try:
            factor_results = run_factor_analysis(
                results_csv=args.results_csv,
                target_cols=target_cols,
                outdir=os.path.join(args.outdir, "factor_analysis"),
                factor_specification=None
            )

            print("Factor analysis completed successfully!")

        except Exception as e:
            print(f"Factor analysis failed: {str(e)}")

    if args.mode in ["goldilocks", "both"]:
        print("\n" + "="*50)
        print("RUNNING GOLDILOCKS CURVE ANALYSIS")
        print("="*50)

        try:
            goldilocks_results = run_goldilocks_analysis(
                results_csv=args.results_csv,
                hyperparameter_cols=hyperparameter_cols,
                target_cols=target_cols,
                outdir=os.path.join(args.outdir, "goldilocks_analysis"),
                title="Unlearning Hyperparameter Optimization"
            )

            print("Goldilocks analysis completed successfully!")

        except Exception as e:
            print(f"Goldilocks analysis failed: {str(e)}")

    print(f"\nPost-hoc analysis complete. Results saved to {args.outdir}")

    # Save a summary of what was analyzed
    summary = {
        "input_file": args.results_csv,
        "n_experiments": len(df),
        "hyperparameter_cols": hyperparameter_cols,
        "target_cols": target_cols,
        "analyses_run": args.mode,
        "output_directory": args.outdir
    }

    with open(os.path.join(args.outdir, "analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nNext steps:")
    print(f"1. Review factor analysis results in {args.outdir}/factor_analysis/")
    print(f"2. Check Goldilocks curves in {args.outdir}/goldilocks_analysis/plots/")
    print(f"3. Use optimal hyperparameters for your next experiments")


if __name__ == "__main__":
    main()