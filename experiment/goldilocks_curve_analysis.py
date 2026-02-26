#!/usr/bin/env python
# /// script
# dependencies = [
#   "torch",
#   "numpy",
#   "pandas",
#   "matplotlib",
#   "seaborn",
#   "scipy",
#   "wandb",
#   "scikit-learn",
# ]
# ///

"""
Goldilocks curve visualization for hyperparameter optimization.

Inspired by subliminal learning Experiment 3b, this creates direct visualizations
of the relationship between hyperparameters and performance outcomes, identifying
optimal "Goldilocks zones" for unlearning effectiveness.

Key insights from subliminal learning:
- Peak accuracy occurred at specific distance from initialization (≈4-5 units)
- The scatter plot directly revealed the inverted-U relationship
- Both under-movement and over-movement were harmful
- The visualization guided hyperparameter selection

For unlearning applications:
- Plot WMDP vs MMLU across hyperparameter sweeps (Pareto frontiers)
- Identify optimal learning rates, epochs, method parameters
- Visualize trade-offs between forgetting and capability retention
- Replace trial-and-error with systematic optimization guidance

This addresses a critical gap: current unlearning work relies on ad-hoc
hyperparameter selection without systematic visualization of the trade-off space.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple, Union

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import griddata, UnivariateSpline
from scipy.optimize import minimize_scalar
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler

from utils import (
    init_wandb,
    finish_wandb,
    log_csv_as_table,
    log_plots,
    write_csv,
)


# ---------------------------------------------------------------------------
# Goldilocks curve fitting and analysis
# ---------------------------------------------------------------------------

def fit_goldilocks_curve(
    x: np.ndarray,
    y: np.ndarray,
    curve_type: str = "gaussian_process"
) -> Tuple[callable, Dict]:
    """
    Fit a smooth curve to identify the Goldilocks zone.

    Args:
        x: Hyperparameter values
        y: Performance values
        curve_type: Type of curve fitting ("spline", "gaussian_process", or "polynomial")

    Returns:
        Tuple of (prediction_function, fit_info)
    """
    # Remove NaN values and sort by x
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean, y_clean = x[mask], y[mask]

    if len(x_clean) < 3:
        raise ValueError("Need at least 3 data points to fit curve")

    sort_idx = np.argsort(x_clean)
    x_clean, y_clean = x_clean[sort_idx], y_clean[sort_idx]

    fit_info = {"curve_type": curve_type, "n_points": len(x_clean)}

    if curve_type == "spline":
        # Univariate spline fitting
        try:
            spline = UnivariateSpline(x_clean, y_clean, s=len(x_clean)*0.1)

            def predict_fn(x_new):
                return spline(x_new)

            fit_info.update({
                "spline_smoothing": spline.get_smoothing_factor(),
                "spline_residual": spline.get_residual()
            })

        except Exception as e:
            # Fallback to simple polynomial
            coeffs = np.polyfit(x_clean, y_clean, min(3, len(x_clean)-1))
            poly = np.poly1d(coeffs)
            predict_fn = poly
            fit_info["fallback"] = f"spline_failed_{str(e)[:50]}"

    elif curve_type == "gaussian_process":
        # Gaussian Process Regression for smooth interpolation
        try:
            # Standardize inputs for better GP performance
            scaler_x = StandardScaler()
            scaler_y = StandardScaler()

            x_scaled = scaler_x.fit_transform(x_clean.reshape(-1, 1)).flatten()
            y_scaled = scaler_y.fit_transform(y_clean.reshape(-1, 1)).flatten()

            # Define kernel with appropriate length scale
            kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
            gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.01, random_state=42)
            gpr.fit(x_scaled.reshape(-1, 1), y_scaled)

            def predict_fn(x_new):
                x_new_scaled = scaler_x.transform(np.array(x_new).reshape(-1, 1)).flatten()
                y_pred_scaled, y_std_scaled = gpr.predict(x_new_scaled.reshape(-1, 1), return_std=True)
                y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                y_std = y_std_scaled * scaler_y.scale_[0]
                return y_pred if len(y_pred) == 1 else (y_pred, y_std)

            fit_info.update({
                "gpr_kernel": str(gpr.kernel_),
                "gpr_log_likelihood": gpr.log_marginal_likelihood_value_
            })

        except Exception as e:
            # Fallback to polynomial
            coeffs = np.polyfit(x_clean, y_clean, min(3, len(x_clean)-1))
            poly = np.poly1d(coeffs)
            predict_fn = poly
            fit_info["fallback"] = f"gp_failed_{str(e)[:50]}"

    elif curve_type == "polynomial":
        # Polynomial fitting (degree 2 or 3 for inverted-U)
        degree = min(3, len(x_clean) - 1)
        coeffs = np.polyfit(x_clean, y_clean, degree)
        poly = np.poly1d(coeffs)

        def predict_fn(x_new):
            return poly(x_new)

        fit_info.update({
            "polynomial_degree": degree,
            "polynomial_coeffs": coeffs.tolist()
        })

    else:
        raise ValueError(f"Unknown curve type: {curve_type}")

    return predict_fn, fit_info


def find_goldilocks_optimum(
    predict_fn: callable,
    x_range: Tuple[float, float],
    maximize: bool = True
) -> Dict:
    """
    Find the optimum point on the Goldilocks curve.

    Args:
        predict_fn: Fitted curve prediction function
        x_range: (min_x, max_x) range to search
        maximize: Whether to find maximum (True) or minimum (False)

    Returns:
        Dictionary with optimum information
    """
    def objective(x):
        try:
            y = predict_fn(x)
            # Handle case where predict_fn returns (mean, std) tuple
            if isinstance(y, tuple):
                y = y[0]
            return -y if maximize else y
        except:
            return np.inf if maximize else -np.inf

    try:
        result = minimize_scalar(objective, bounds=x_range, method='bounded')

        optimum_x = result.x
        optimum_y = predict_fn(optimum_x)

        # Handle GP case with uncertainty
        if isinstance(optimum_y, tuple):
            optimum_y, optimum_std = optimum_y
        else:
            optimum_std = None

        return {
            "optimum_x": optimum_x,
            "optimum_y": optimum_y,
            "optimum_std": optimum_std,
            "optimization_success": result.success,
            "optimization_message": result.message if hasattr(result, 'message') else None
        }

    except Exception as e:
        return {
            "error": f"Optimization failed: {str(e)}",
            "optimum_x": None,
            "optimum_y": None
        }


# ---------------------------------------------------------------------------
# Multi-dimensional Goldilocks analysis
# ---------------------------------------------------------------------------

def create_2d_goldilocks_surface(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    grid_resolution: int = 50
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a 2D surface for visualizing Goldilocks zones across two hyperparameters.

    Returns:
        Tuple of (X_grid, Y_grid, Z_grid) for surface plotting
    """
    # Clean data
    clean_df = df[[x_col, y_col, z_col]].dropna()

    if len(clean_df) < 4:
        raise ValueError("Need at least 4 data points for 2D surface")

    x = clean_df[x_col].values
    y = clean_df[y_col].values
    z = clean_df[z_col].values

    # Create regular grid
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    xi = np.linspace(x_min, x_max, grid_resolution)
    yi = np.linspace(y_min, y_max, grid_resolution)
    X_grid, Y_grid = np.meshgrid(xi, yi)

    # Interpolate to grid
    try:
        Z_grid = griddata((x, y), z, (X_grid, Y_grid), method='cubic', fill_value=np.nan)

        # Fill remaining NaN values with linear interpolation
        mask = np.isnan(Z_grid)
        if np.any(mask):
            Z_grid_linear = griddata((x, y), z, (X_grid, Y_grid), method='linear', fill_value=np.nan)
            Z_grid[mask] = Z_grid_linear[mask]

        # Final fallback to nearest neighbor
        mask = np.isnan(Z_grid)
        if np.any(mask):
            Z_grid_nearest = griddata((x, y), z, (X_grid, Y_grid), method='nearest')
            Z_grid[mask] = Z_grid_nearest[mask]

    except Exception as e:
        # Fallback to simple griddata
        Z_grid = griddata((x, y), z, (X_grid, Y_grid), method='nearest')

    return X_grid, Y_grid, Z_grid


# ---------------------------------------------------------------------------
# Visualization functions
# ---------------------------------------------------------------------------

def create_goldilocks_plots(
    df: pd.DataFrame,
    hyperparameter_cols: List[str],
    target_cols: List[str],
    outdir: str,
    title: str = "Goldilocks Curve Analysis"
) -> Dict:
    """
    Create comprehensive Goldilocks curve visualizations.

    Args:
        df: DataFrame with experimental results
        hyperparameter_cols: List of hyperparameter column names
        target_cols: List of target performance column names
        outdir: Output directory for plots
        title: Plot title prefix

    Returns:
        Dictionary with analysis results
    """
    os.makedirs(outdir, exist_ok=True)

    results = {
        "curve_fits": {},
        "optima": {},
        "pareto_frontiers": {}
    }

    # 1D Goldilocks curves for each hyperparameter vs each target
    for target_col in target_cols:
        for hyperparam_col in hyperparameter_cols:

            # Skip if either column is missing
            if hyperparam_col not in df.columns or target_col not in df.columns:
                continue

            # Clean data
            clean_df = df[[hyperparam_col, target_col]].dropna()
            if len(clean_df) < 3:
                continue

            x = clean_df[hyperparam_col].values
            y = clean_df[target_col].values

            # Fit curve
            try:
                predict_fn, fit_info = fit_goldilocks_curve(x, y, curve_type="gaussian_process")

                # Find optimum
                x_range = (x.min(), x.max())
                optimum_info = find_goldilocks_optimum(predict_fn, x_range, maximize=True)

                curve_key = f"{target_col}_vs_{hyperparam_col}"
                results["curve_fits"][curve_key] = fit_info
                results["optima"][curve_key] = optimum_info

                # Create individual plot
                fig, ax = plt.subplots(figsize=(10, 6))

                # Plot original data points
                ax.scatter(x, y, alpha=0.6, s=50, color='blue', label='Experimental data')

                # Plot fitted curve
                x_smooth = np.linspace(x.min(), x.max(), 200)
                try:
                    y_smooth = predict_fn(x_smooth)
                    if isinstance(y_smooth, tuple):
                        y_pred, y_std = y_smooth
                        ax.plot(x_smooth, y_pred, 'r-', linewidth=2, label='Fitted curve')
                        ax.fill_between(x_smooth, y_pred - 1.96*y_std, y_pred + 1.96*y_std,
                                       alpha=0.2, color='red', label='95% confidence')
                    else:
                        ax.plot(x_smooth, y_smooth, 'r-', linewidth=2, label='Fitted curve')
                except:
                    pass  # Skip curve plotting if prediction fails

                # Mark optimum
                if optimum_info.get("optimum_x") is not None:
                    ax.axvline(optimum_info["optimum_x"], color='green', linestyle='--',
                              alpha=0.8, label=f'Optimum: {optimum_info["optimum_x"]:.3f}')

                ax.set_xlabel(hyperparam_col)
                ax.set_ylabel(target_col)
                ax.set_title(f'{title}: {target_col} vs {hyperparam_col}')
                ax.legend()
                ax.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(os.path.join(outdir, f'goldilocks_{target_col}_vs_{hyperparam_col}.png'),
                           dpi=300, bbox_inches='tight')
                plt.close()

            except Exception as e:
                print(f"Warning: Failed to create curve for {target_col} vs {hyperparam_col}: {str(e)}")

    # 2D Pareto frontier plots (if we have multiple targets)
    if len(target_cols) >= 2:
        for i in range(len(target_cols)):
            for j in range(i+1, len(target_cols)):
                target_x, target_y = target_cols[i], target_cols[j]

                # Clean data for both targets
                clean_df = df[[target_x, target_y]].dropna()
                if len(clean_df) < 3:
                    continue

                # Create Pareto frontier plot
                fig, ax = plt.subplots(figsize=(10, 8))

                # Color points by a primary hyperparameter if available
                if hyperparameter_cols and hyperparameter_cols[0] in clean_df.columns:
                    color_col = hyperparameter_cols[0]
                    scatter = ax.scatter(clean_df[target_x], clean_df[target_y],
                                       c=clean_df[color_col], s=60, alpha=0.7,
                                       cmap='viridis')
                    plt.colorbar(scatter, label=color_col)
                else:
                    ax.scatter(clean_df[target_x], clean_df[target_y],
                              s=60, alpha=0.7, color='blue')

                # Identify approximate Pareto frontier
                try:
                    points = clean_df[[target_x, target_y]].values
                    pareto_mask = np.ones(len(points), dtype=bool)

                    for i, point in enumerate(points):
                        for j, other_point in enumerate(points):
                            if i != j:
                                # Check if other_point dominates point (assuming higher is better for both)
                                if np.all(other_point >= point) and np.any(other_point > point):
                                    pareto_mask[i] = False
                                    break

                    pareto_points = points[pareto_mask]
                    if len(pareto_points) > 1:
                        # Sort by first objective for connecting line
                        sorted_indices = np.argsort(pareto_points[:, 0])
                        pareto_sorted = pareto_points[sorted_indices]
                        ax.plot(pareto_sorted[:, 0], pareto_sorted[:, 1], 'r--',
                               linewidth=2, alpha=0.8, label='Approximate Pareto frontier')
                        ax.scatter(pareto_sorted[:, 0], pareto_sorted[:, 1],
                                  s=100, color='red', marker='*', label='Pareto optimal')

                        results["pareto_frontiers"][f"{target_x}_vs_{target_y}"] = {
                            "pareto_points": pareto_sorted.tolist(),
                            "n_pareto_points": len(pareto_sorted)
                        }

                except Exception as e:
                    print(f"Warning: Failed to compute Pareto frontier for {target_x} vs {target_y}: {str(e)}")

                ax.set_xlabel(target_x)
                ax.set_ylabel(target_y)
                ax.set_title(f'{title}: {target_x} vs {target_y} Trade-off')
                ax.legend()
                ax.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(os.path.join(outdir, f'pareto_{target_x}_vs_{target_y}.png'),
                           dpi=300, bbox_inches='tight')
                plt.close()

    # 2D surface plots (if we have 2+ hyperparameters and 1+ target)
    if len(hyperparameter_cols) >= 2 and len(target_cols) >= 1:
        for target_col in target_cols[:2]:  # Limit to first 2 targets to avoid too many plots
            for i in range(len(hyperparameter_cols)):
                for j in range(i+1, min(i+2, len(hyperparameter_cols))):  # Limit combinations
                    hyperparam_x, hyperparam_y = hyperparameter_cols[i], hyperparameter_cols[j]

                    try:
                        X_grid, Y_grid, Z_grid = create_2d_goldilocks_surface(
                            df, hyperparam_x, hyperparam_y, target_col, grid_resolution=30
                        )

                        fig = plt.figure(figsize=(12, 8))
                        ax = fig.add_subplot(111, projection='3d')

                        surface = ax.plot_surface(X_grid, Y_grid, Z_grid, cmap='viridis',
                                                alpha=0.8, edgecolor='none')

                        # Add original data points
                        clean_df = df[[hyperparam_x, hyperparam_y, target_col]].dropna()
                        ax.scatter(clean_df[hyperparam_x], clean_df[hyperparam_y],
                                 clean_df[target_col], color='red', s=30, alpha=0.6)

                        ax.set_xlabel(hyperparam_x)
                        ax.set_ylabel(hyperparam_y)
                        ax.set_zlabel(target_col)
                        ax.set_title(f'{title}: {target_col} Surface')

                        plt.colorbar(surface, shrink=0.6)
                        plt.tight_layout()
                        plt.savefig(os.path.join(outdir, f'surface_{target_col}_{hyperparam_x}_{hyperparam_y}.png'),
                                   dpi=300, bbox_inches='tight')
                        plt.close()

                    except Exception as e:
                        print(f"Warning: Failed to create surface plot for {target_col}: {str(e)}")

    print(f"Goldilocks curve analysis plots saved to {outdir}")
    return results


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def run_goldilocks_analysis(
    results_csv: str,
    hyperparameter_cols: List[str],
    target_cols: List[str],
    outdir: str,
    title: str = "Goldilocks Analysis"
) -> Dict:
    """
    Run comprehensive Goldilocks curve analysis.

    Args:
        results_csv: Path to experimental results CSV
        hyperparameter_cols: List of hyperparameter column names
        target_cols: List of target performance column names
        outdir: Output directory
        title: Analysis title

    Returns:
        Analysis results dictionary
    """
    os.makedirs(outdir, exist_ok=True)

    # Load data
    print(f"Loading experimental results from {results_csv}")
    df = pd.read_csv(results_csv)
    print(f"Loaded {len(df)} experimental runs")

    # Validate columns
    missing_hyperparams = [col for col in hyperparameter_cols if col not in df.columns]
    missing_targets = [col for col in target_cols if col not in df.columns]

    if missing_hyperparams:
        print(f"Warning: Missing hyperparameter columns: {missing_hyperparams}")
        hyperparameter_cols = [col for col in hyperparameter_cols if col in df.columns]

    if missing_targets:
        print(f"Warning: Missing target columns: {missing_targets}")
        target_cols = [col for col in target_cols if col in df.columns]

    if not hyperparameter_cols or not target_cols:
        raise ValueError("No valid hyperparameter or target columns found")

    # Create visualizations and analysis
    print("Creating Goldilocks curve visualizations...")
    plot_outdir = os.path.join(outdir, "plots")
    results = create_goldilocks_plots(df, hyperparameter_cols, target_cols, plot_outdir, title)

    # Save detailed results
    with open(os.path.join(outdir, "goldilocks_analysis.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Create summary table of optima
    optima_data = []
    for curve_key, optimum_info in results["optima"].items():
        if optimum_info.get("optimum_x") is not None:
            target_col, hyperparam_col = curve_key.split("_vs_")
            optima_data.append({
                "target": target_col,
                "hyperparameter": hyperparam_col,
                "optimal_value": optimum_info["optimum_x"],
                "optimal_performance": optimum_info["optimum_y"],
                "uncertainty": optimum_info.get("optimum_std"),
                "optimization_success": optimum_info.get("optimization_success", False)
            })

    if optima_data:
        optima_df = pd.DataFrame(optima_data)
        optima_df.to_csv(os.path.join(outdir, "goldilocks_optima.csv"), index=False)

        print(f"\nGoldilocks analysis complete. Results saved to {outdir}")
        print(f"\nOptimal hyperparameter values found:")
        for _, row in optima_df.iterrows():
            print(f"  {row['target']} maximized when {row['hyperparameter']} = {row['optimal_value']:.4f} "
                  f"(performance: {row['optimal_performance']:.4f})")

    return results


# ---------------------------------------------------------------------------
# CLI interface
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Goldilocks curve visualization for hyperparameter optimization"
    )
    parser.add_argument("--results-csv", required=True,
                       help="Path to experimental results CSV")
    parser.add_argument("--hyperparameter-cols", nargs="+", required=True,
                       help="Hyperparameter column names")
    parser.add_argument("--target-cols", nargs="+", required=True,
                       help="Target performance column names")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--title", default="Goldilocks Analysis", help="Analysis title")
    parser.add_argument("--wandb-project", help="W&B project name")
    parser.add_argument("--wandb-name", help="W&B run name")

    args = parser.parse_args()

    # Initialize W&B if specified
    wandb_run = None
    if args.wandb_project:
        wandb_run = init_wandb(args.wandb_project, args.wandb_name or "goldilocks_analysis")

    try:
        results = run_goldilocks_analysis(
            results_csv=args.results_csv,
            hyperparameter_cols=args.hyperparameter_cols,
            target_cols=args.target_cols,
            outdir=args.outdir,
            title=args.title
        )

        # Log results to W&B
        if wandb_run:
            import wandb

            # Log optimal hyperparameter values
            optima_summary = {}
            for curve_key, optimum_info in results["optima"].items():
                if optimum_info.get("optimum_x") is not None:
                    target_col, hyperparam_col = curve_key.split("_vs_")
                    optima_summary[f"optimal_{hyperparam_col}_for_{target_col}"] = optimum_info["optimum_x"]
                    optima_summary[f"max_{target_col}_via_{hyperparam_col}"] = optimum_info["optimum_y"]

            wandb.log(optima_summary)

            # Log tables and plots
            if os.path.exists(os.path.join(args.outdir, "goldilocks_optima.csv")):
                log_csv_as_table("goldilocks_optima", os.path.join(args.outdir, "goldilocks_optima.csv"))
            log_plots(os.path.join(args.outdir, "plots"))

    finally:
        if wandb_run:
            finish_wandb()


if __name__ == "__main__":
    main()