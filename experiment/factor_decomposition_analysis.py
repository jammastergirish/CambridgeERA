#!/usr/bin/env python
# /// script
# dependencies = [
#   "torch",
#   "numpy",
#   "pandas",
#   "scikit-learn",
#   "matplotlib",
#   "seaborn",
#   "wandb",
#   "statsmodels",
# ]
# ///

"""
Multi-factor variance decomposition for unlearning effectiveness.

Inspired by subliminal learning Experiment 5b, this decomposes the variance
in unlearning outcomes into contributing factors using rigorous statistical
methods. Key insights from subliminal learning:

- Animal identity explained 68.1% of variance in subliminal prompting
- Prompt template explained 14.6%
- Geometric factors (cosine similarity) explained ~0% of magnitude
- Total R² = 82.7%, with 17.3% unexplained variance

For unlearning applications:
- Decompose WMDP/MMLU performance across experimental factors
- Identify which factors matter most for unlearning success
- Quantify method-specific vs hyperparameter vs architectural contributions
- Replace qualitative analysis with quantified factor importance

This provides a systematic framework for understanding what drives
unlearning effectiveness and guides hyperparameter optimization.
"""

import argparse
import json
import os
import sys
from itertools import combinations
from typing import Dict, List, Optional, Tuple, Union

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm

from utils import (
    init_wandb,
    finish_wandb,
    log_csv_as_table,
    log_plots,
    write_csv,
)


# ---------------------------------------------------------------------------
# Factor definition and encoding
# ---------------------------------------------------------------------------

def identify_experimental_factors(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Automatically identify experimental factors from a results DataFrame.

    Returns factor metadata including type (categorical vs continuous),
    unique values, and encoding strategy.
    """
    factors = {}

    # Common experimental factors to look for
    factor_patterns = {
        'method': ['method', 'unlearning_method', 'approach'],
        'learning_rate': ['lr', 'learning_rate', 'learn_rate'],
        'batch_size': ['batch_size', 'bs'],
        'epochs': ['epochs', 'num_epochs', 'training_epochs'],
        'architecture': ['arch', 'architecture', 'model_size'],
        'layer_count': ['layers', 'num_layers', 'depth'],
        'hidden_dim': ['hidden_dim', 'hidden_size', 'd_model'],
        'forget_weight': ['forget_weight', 'alpha', 'forget_coeff'],
        'retain_weight': ['retain_weight', 'beta', 'retain_coeff'],
        'steering_coeff': ['steering_coeff', 'steering_coefficient'],
        'dataset_size': ['dataset_size', 'train_size', 'n_samples'],
    }

    # Identify factors present in the DataFrame
    for factor_name, column_patterns in factor_patterns.items():
        matched_column = None
        for pattern in column_patterns:
            for col in df.columns:
                if pattern.lower() in col.lower():
                    matched_column = col
                    break
            if matched_column:
                break

        if matched_column:
            values = df[matched_column].dropna()
            unique_values = values.unique()

            # Determine if categorical or continuous
            if len(unique_values) <= 10 or values.dtype == 'object':
                factor_type = 'categorical'
            else:
                factor_type = 'continuous'

            factors[factor_name] = {
                'column': matched_column,
                'type': factor_type,
                'unique_values': unique_values.tolist() if len(unique_values) <= 20 else f"{len(unique_values)} unique values",
                'n_unique': len(unique_values),
                'missing_count': df[matched_column].isna().sum()
            }

    return factors


def encode_factors(df: pd.DataFrame, factors: Dict[str, Dict]) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Encode factors for statistical analysis.

    - Categorical factors: Label encoding + one-hot for regression
    - Continuous factors: Standardization
    """
    encoded_df = df.copy()
    encoders = {}

    for factor_name, factor_info in factors.items():
        column = factor_info['column']
        factor_type = factor_info['type']

        if factor_type == 'categorical':
            # Label encode for tree methods
            encoder = LabelEncoder()
            encoded_df[f'{factor_name}_encoded'] = encoder.fit_transform(
                encoded_df[column].astype(str)
            )
            encoders[factor_name] = encoder

            # One-hot encode for linear methods (if not too many categories)
            if factor_info['n_unique'] <= 10:
                dummies = pd.get_dummies(encoded_df[column], prefix=factor_name)
                encoded_df = pd.concat([encoded_df, dummies], axis=1)

        elif factor_type == 'continuous':
            # Standardize continuous factors
            scaler = StandardScaler()
            encoded_df[f'{factor_name}_scaled'] = scaler.fit_transform(
                encoded_df[[column]]
            ).flatten()
            encoders[factor_name] = scaler

    return encoded_df, encoders


# ---------------------------------------------------------------------------
# Variance decomposition methods
# ---------------------------------------------------------------------------

def linear_variance_decomposition(
    df: pd.DataFrame,
    target_col: str,
    factor_cols: List[str],
    interaction_terms: bool = True
) -> Dict:
    """
    Linear regression-based variance decomposition.

    Returns R² for each factor and their interactions.
    """
    results = {
        'method': 'linear_regression',
        'target': target_col,
        'factors': factor_cols,
        'individual_r2': {},
        'cumulative_r2': {},
        'interaction_r2': {},
        'model_performance': {}
    }

    # Clean data
    clean_df = df.dropna(subset=[target_col] + factor_cols)
    if len(clean_df) == 0:
        return {'error': 'No complete cases after removing missing values'}

    y = clean_df[target_col].values

    # Individual factor contributions
    for factor in factor_cols:
        X = clean_df[[factor]].values
        model = LinearRegression().fit(X, y)
        r2 = r2_score(y, model.predict(X))
        results['individual_r2'][factor] = r2

    # Cumulative R² (sequential addition)
    cumulative_r2 = 0
    for i in range(1, len(factor_cols) + 1):
        X = clean_df[factor_cols[:i]].values
        model = LinearRegression().fit(X, y)
        r2 = r2_score(y, model.predict(X))
        results['cumulative_r2'][f'factors_1_to_{i}'] = r2

        # Marginal contribution of this factor
        marginal_r2 = r2 - cumulative_r2
        results['cumulative_r2'][f'marginal_factor_{i}'] = marginal_r2
        cumulative_r2 = r2

    # Interaction terms (if requested and computationally feasible)
    if interaction_terms and len(factor_cols) <= 5:
        # All factors main effects
        X_main = clean_df[factor_cols].values
        main_model = LinearRegression().fit(X_main, y)
        main_r2 = r2_score(y, main_model.predict(X_main))

        # Add interaction terms
        interaction_features = []
        interaction_names = []

        for i, j in combinations(range(len(factor_cols)), 2):
            interaction = clean_df.iloc[:, clean_df.columns.get_loc(factor_cols[i])] * \
                         clean_df.iloc[:, clean_df.columns.get_loc(factor_cols[j])]
            interaction_features.append(interaction.values)
            interaction_names.append(f'{factor_cols[i]}_x_{factor_cols[j]}')

        if interaction_features:
            X_with_interactions = np.column_stack([X_main] + interaction_features)
            interaction_model = LinearRegression().fit(X_with_interactions, y)
            interaction_r2 = r2_score(y, interaction_model.predict(X_with_interactions))

            results['interaction_r2']['main_effects'] = main_r2
            results['interaction_r2']['with_interactions'] = interaction_r2
            results['interaction_r2']['interaction_contribution'] = interaction_r2 - main_r2
            results['interaction_r2']['interaction_names'] = interaction_names

    # Full model performance
    X_all = clean_df[factor_cols].values
    full_model = LinearRegression().fit(X_all, y)
    full_r2 = r2_score(y, full_model.predict(X_all))

    results['model_performance'] = {
        'full_r2': full_r2,
        'adjusted_r2': 1 - (1 - full_r2) * (len(y) - 1) / (len(y) - len(factor_cols) - 1),
        'n_observations': len(y),
        'n_factors': len(factor_cols),
        'unexplained_variance': 1 - full_r2
    }

    return results


def random_forest_importance(
    df: pd.DataFrame,
    target_col: str,
    factor_cols: List[str],
    n_estimators: int = 100
) -> Dict:
    """
    Random Forest-based feature importance analysis.

    Captures non-linear relationships and interactions automatically.
    """
    results = {
        'method': 'random_forest',
        'target': target_col,
        'factors': factor_cols,
        'feature_importance': {},
        'model_performance': {}
    }

    # Clean data
    clean_df = df.dropna(subset=[target_col] + factor_cols)
    if len(clean_df) == 0:
        return {'error': 'No complete cases after removing missing values'}

    X = clean_df[factor_cols].values
    y = clean_df[target_col].values

    # Fit Random Forest
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    rf.fit(X, y)

    # Feature importance (normalized to sum to 1)
    importances = rf.feature_importances_
    for i, factor in enumerate(factor_cols):
        results['feature_importance'][factor] = importances[i]

    # Model performance
    y_pred = rf.predict(X)
    r2 = r2_score(y, y_pred)

    # Cross-validation score for more robust estimate
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring='r2')

    results['model_performance'] = {
        'training_r2': r2,
        'cv_mean_r2': cv_scores.mean(),
        'cv_std_r2': cv_scores.std(),
        'oob_score': rf.oob_score_ if rf.oob_score_ is not None else None,
        'n_observations': len(y),
        'n_factors': len(factor_cols)
    }

    return results


def anova_decomposition(
    df: pd.DataFrame,
    target_col: str,
    categorical_factors: List[str]
) -> Dict:
    """
    ANOVA-based variance decomposition for categorical factors.

    Provides F-statistics and p-values for factor significance.
    """
    if not categorical_factors:
        return {'error': 'No categorical factors provided for ANOVA'}

    results = {
        'method': 'anova',
        'target': target_col,
        'factors': categorical_factors,
        'anova_results': {},
        'factor_significance': {}
    }

    # Clean data
    clean_df = df.dropna(subset=[target_col] + categorical_factors)
    if len(clean_df) == 0:
        return {'error': 'No complete cases after removing missing values'}

    try:
        # Construct formula for statsmodels
        formula = f"{target_col} ~ " + " + ".join(categorical_factors)

        # Fit ANOVA model
        model = sm.formula.ols(formula, data=clean_df).fit()
        anova_table = anova_lm(model, typ=2)  # Type II ANOVA

        # Extract results
        for factor in categorical_factors:
            if factor in anova_table.index:
                results['factor_significance'][factor] = {
                    'f_statistic': anova_table.loc[factor, 'F'],
                    'p_value': anova_table.loc[factor, 'PR(>F)'],
                    'sum_squares': anova_table.loc[factor, 'sum_sq'],
                    'degrees_freedom': anova_table.loc[factor, 'df']
                }

        # Calculate eta-squared (effect size)
        total_ss = anova_table['sum_sq'].sum()
        for factor in categorical_factors:
            if factor in anova_table.index:
                eta_squared = anova_table.loc[factor, 'sum_sq'] / total_ss
                results['factor_significance'][factor]['eta_squared'] = eta_squared

        results['anova_results'] = {
            'model_r2': model.rsquared,
            'adjusted_r2': model.rsquared_adj,
            'f_statistic': model.fvalue,
            'model_p_value': model.f_pvalue,
            'aic': model.aic,
            'bic': model.bic,
            'n_observations': int(model.nobs)
        }

    except Exception as e:
        results['error'] = f'ANOVA failed: {str(e)}'

    return results


# ---------------------------------------------------------------------------
# Comprehensive factor analysis
# ---------------------------------------------------------------------------

def comprehensive_factor_analysis(
    df: pd.DataFrame,
    target_cols: List[str],
    factor_specification: Optional[Dict] = None
) -> Dict:
    """
    Run comprehensive multi-method factor analysis.

    Args:
        df: Results DataFrame
        target_cols: Outcome variables to analyze (e.g., ['wmdp_accuracy', 'mmlu_accuracy'])
        factor_specification: Optional manual factor specification

    Returns:
        Comprehensive analysis results
    """
    # Identify factors automatically if not specified
    if factor_specification is None:
        factors = identify_experimental_factors(df)
    else:
        factors = factor_specification

    if not factors:
        return {'error': 'No experimental factors identified in the data'}

    # Encode factors for analysis
    encoded_df, encoders = encode_factors(df, factors)

    # Prepare factor columns for different analysis types
    continuous_factors = [f"{name}_scaled" for name, info in factors.items()
                         if info['type'] == 'continuous' and f"{name}_scaled" in encoded_df.columns]
    categorical_factors = [info['column'] for name, info in factors.items()
                          if info['type'] == 'categorical']
    encoded_categorical = [f"{name}_encoded" for name, info in factors.items()
                          if info['type'] == 'categorical' and f"{name}_encoded" in encoded_df.columns]

    all_factors = continuous_factors + encoded_categorical

    results = {
        'factor_specification': factors,
        'encoding_info': {name: type(encoder).__name__ for name, encoder in encoders.items()},
        'analysis_methods': {},
        'target_analyses': {}
    }

    # Run analysis for each target variable
    for target_col in target_cols:
        if target_col not in encoded_df.columns:
            print(f"Warning: Target column '{target_col}' not found in data")
            continue

        target_results = {}

        # Linear regression decomposition
        if all_factors:
            linear_results = linear_variance_decomposition(
                encoded_df, target_col, all_factors, interaction_terms=True
            )
            target_results['linear'] = linear_results

        # Random Forest importance
        if all_factors:
            rf_results = random_forest_importance(
                encoded_df, target_col, all_factors
            )
            target_results['random_forest'] = rf_results

        # ANOVA (for categorical factors)
        if categorical_factors:
            anova_results = anova_decomposition(
                encoded_df, target_col, categorical_factors
            )
            target_results['anova'] = anova_results

        results['target_analyses'][target_col] = target_results

    return results


# ---------------------------------------------------------------------------
# Visualization functions
# ---------------------------------------------------------------------------

def create_factor_plots(analysis_results: Dict, outdir: str, title: str = "Factor Analysis"):
    """Create comprehensive factor analysis visualizations."""
    os.makedirs(outdir, exist_ok=True)

    # Set style
    plt.style.use('default')
    sns.set_palette("husl")

    target_analyses = analysis_results.get('target_analyses', {})

    for target_name, target_results in target_analyses.items():
        # Create multi-panel figure for this target
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"{title} - {target_name}", fontsize=16)

        # Plot 1: Linear regression R² values
        if 'linear' in target_results and 'individual_r2' in target_results['linear']:
            individual_r2 = target_results['linear']['individual_r2']
            factors = list(individual_r2.keys())
            r2_values = list(individual_r2.values())

            bars = axes[0, 0].bar(range(len(factors)), r2_values, color='skyblue')
            axes[0, 0].set_xticks(range(len(factors)))
            axes[0, 0].set_xticklabels(factors, rotation=45, ha='right')
            axes[0, 0].set_ylabel('Individual R²')
            axes[0, 0].set_title('Linear Regression: Individual Factor Contributions')
            axes[0, 0].grid(True, alpha=0.3)

            # Add value labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                               f'{height:.3f}', ha='center', va='bottom')

        # Plot 2: Random Forest feature importance
        if 'random_forest' in target_results and 'feature_importance' in target_results['random_forest']:
            importance = target_results['random_forest']['feature_importance']
            factors = list(importance.keys())
            importance_values = list(importance.values())

            bars = axes[0, 1].bar(range(len(factors)), importance_values, color='lightcoral')
            axes[0, 1].set_xticks(range(len(factors)))
            axes[0, 1].set_xticklabels(factors, rotation=45, ha='right')
            axes[0, 1].set_ylabel('Feature Importance')
            axes[0, 1].set_title('Random Forest: Feature Importance')
            axes[0, 1].grid(True, alpha=0.3)

            # Add value labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                               f'{height:.3f}', ha='center', va='bottom')

        # Plot 3: ANOVA effect sizes (eta-squared)
        if 'anova' in target_results and 'factor_significance' in target_results['anova']:
            significance = target_results['anova']['factor_significance']
            factors = list(significance.keys())
            eta_squared = [significance[f].get('eta_squared', 0) for f in factors]
            p_values = [significance[f].get('p_value', 1) for f in factors]

            # Color bars by significance
            colors = ['red' if p < 0.05 else 'orange' if p < 0.1 else 'lightgray' for p in p_values]

            bars = axes[1, 0].bar(range(len(factors)), eta_squared, color=colors)
            axes[1, 0].set_xticks(range(len(factors)))
            axes[1, 0].set_xticklabels(factors, rotation=45, ha='right')
            axes[1, 0].set_ylabel('Eta-squared (Effect Size)')
            axes[1, 0].set_title('ANOVA: Effect Sizes (Red=p<0.05, Orange=p<0.1)')
            axes[1, 0].grid(True, alpha=0.3)

            # Add value labels with p-values
            for i, bar in enumerate(bars):
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                               f'{height:.3f}\n(p={p_values[i]:.3f})',
                               ha='center', va='bottom', fontsize=8)

        # Plot 4: Model comparison (R² across methods)
        method_r2 = {}
        if 'linear' in target_results and 'model_performance' in target_results['linear']:
            method_r2['Linear'] = target_results['linear']['model_performance']['full_r2']
        if 'random_forest' in target_results and 'model_performance' in target_results['random_forest']:
            method_r2['Random Forest'] = target_results['random_forest']['model_performance']['cv_mean_r2']
        if 'anova' in target_results and 'anova_results' in target_results['anova']:
            method_r2['ANOVA'] = target_results['anova']['anova_results']['model_r2']

        if method_r2:
            methods = list(method_r2.keys())
            r2_vals = list(method_r2.values())

            bars = axes[1, 1].bar(range(len(methods)), r2_vals, color='lightgreen')
            axes[1, 1].set_xticks(range(len(methods)))
            axes[1, 1].set_xticklabels(methods)
            axes[1, 1].set_ylabel('R² / Explained Variance')
            axes[1, 1].set_title('Model Comparison')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim(0, 1)

            # Add value labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                               f'{height:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'factor_analysis_{target_name}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Factor analysis plots saved to {outdir}")


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def run_factor_analysis(
    results_csv: str,
    target_cols: List[str],
    outdir: str,
    factor_specification: Optional[Dict] = None
) -> Dict:
    """
    Run comprehensive factor analysis on experimental results.

    Args:
        results_csv: Path to CSV with experimental results
        target_cols: List of outcome variables to analyze
        outdir: Output directory
        factor_specification: Optional manual factor specification

    Returns:
        Analysis results dictionary
    """
    os.makedirs(outdir, exist_ok=True)

    # Load data
    print(f"Loading experimental results from {results_csv}")
    df = pd.read_csv(results_csv)
    print(f"Loaded {len(df)} experimental runs with {len(df.columns)} columns")

    # Validate target columns
    missing_targets = [col for col in target_cols if col not in df.columns]
    if missing_targets:
        print(f"Warning: Target columns not found: {missing_targets}")
        target_cols = [col for col in target_cols if col in df.columns]

    if not target_cols:
        raise ValueError("No valid target columns found in the data")

    # Run comprehensive analysis
    print("Running comprehensive factor analysis...")
    results = comprehensive_factor_analysis(df, target_cols, factor_specification)

    if 'error' in results:
        raise ValueError(f"Factor analysis failed: {results['error']}")

    # Save detailed results
    with open(os.path.join(outdir, "factor_analysis_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Create summary table
    summary_data = []
    for target_name, target_results in results['target_analyses'].items():
        # Linear regression summary
        if 'linear' in target_results and 'individual_r2' in target_results['linear']:
            for factor, r2 in target_results['linear']['individual_r2'].items():
                summary_data.append({
                    'target': target_name,
                    'method': 'linear_regression',
                    'factor': factor,
                    'contribution': r2,
                    'metric': 'individual_r2'
                })

        # Random Forest summary
        if 'random_forest' in target_results and 'feature_importance' in target_results['random_forest']:
            for factor, importance in target_results['random_forest']['feature_importance'].items():
                summary_data.append({
                    'target': target_name,
                    'method': 'random_forest',
                    'factor': factor,
                    'contribution': importance,
                    'metric': 'feature_importance'
                })

        # ANOVA summary
        if 'anova' in target_results and 'factor_significance' in target_results['anova']:
            for factor, stats in target_results['anova']['factor_significance'].items():
                summary_data.append({
                    'target': target_name,
                    'method': 'anova',
                    'factor': factor,
                    'contribution': stats.get('eta_squared', 0),
                    'metric': 'eta_squared',
                    'p_value': stats.get('p_value', None)
                })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(outdir, "factor_summary.csv"), index=False)

    # Create visualizations
    plot_outdir = os.path.join(outdir, "plots")
    create_factor_plots(results, plot_outdir, title="Unlearning Factor Analysis")

    # Print key findings
    print(f"\nFactor analysis complete. Results saved to {outdir}")
    print(f"\nKey findings:")

    for target_name, target_results in results['target_analyses'].items():
        print(f"\n{target_name.upper()}:")

        if 'linear' in target_results and 'model_performance' in target_results['linear']:
            total_r2 = target_results['linear']['model_performance']['full_r2']
            print(f"  Total explained variance (linear): {total_r2:.1%}")

            # Top factors
            if 'individual_r2' in target_results['linear']:
                top_factors = sorted(
                    target_results['linear']['individual_r2'].items(),
                    key=lambda x: x[1], reverse=True
                )[:3]
                print(f"  Top factors (linear):")
                for factor, r2 in top_factors:
                    print(f"    {factor}: {r2:.1%}")

        if 'random_forest' in target_results and 'model_performance' in target_results['random_forest']:
            rf_r2 = target_results['random_forest']['model_performance']['cv_mean_r2']
            print(f"  Total explained variance (RF): {rf_r2:.1%}")

    return results


# ---------------------------------------------------------------------------
# CLI interface
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-factor variance decomposition for unlearning effectiveness"
    )
    parser.add_argument("--results-csv", required=True,
                       help="Path to CSV with experimental results")
    parser.add_argument("--target-cols", nargs="+", required=True,
                       help="Target columns to analyze (e.g., wmdp_accuracy mmlu_accuracy)")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--factor-spec", help="JSON file with manual factor specification")
    parser.add_argument("--wandb-project", help="W&B project name")
    parser.add_argument("--wandb-name", help="W&B run name")

    args = parser.parse_args()

    # Load manual factor specification if provided
    factor_specification = None
    if args.factor_spec:
        with open(args.factor_spec, 'r') as f:
            factor_specification = json.load(f)

    # Initialize W&B if specified
    wandb_run = None
    if args.wandb_project:
        wandb_run = init_wandb(args.wandb_project, args.wandb_name or "factor_analysis")

    try:
        results = run_factor_analysis(
            results_csv=args.results_csv,
            target_cols=args.target_cols,
            outdir=args.outdir,
            factor_specification=factor_specification
        )

        # Log results to W&B
        if wandb_run:
            import wandb

            # Log summary metrics
            for target_name, target_results in results['target_analyses'].items():
                if 'linear' in target_results and 'model_performance' in target_results['linear']:
                    wandb.log({
                        f"{target_name}_total_r2": target_results['linear']['model_performance']['full_r2'],
                        f"{target_name}_adjusted_r2": target_results['linear']['model_performance']['adjusted_r2']
                    })

            log_csv_as_table("factor_summary", os.path.join(args.outdir, "factor_summary.csv"))
            log_plots(os.path.join(args.outdir, "plots"))

    finally:
        if wandb_run:
            finish_wandb()


if __name__ == "__main__":
    main()