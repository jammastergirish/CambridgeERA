#!/usr/bin/env python
# /// script
# dependencies = [
#   "torch",
#   "pandas",
#   "matplotlib",
#   "numpy",
# ]
# ///

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-layer-csv", required=True, help="Path to per_layer.csv from collect_param_stats.py")
    ap.add_argument("--per-matrix-csv", required=True, help="Path to per_matrix.csv from collect_param_stats.py")
    ap.add_argument("--outdir", default="outputs/mlp_attn_analysis")
    ap.add_argument("--title", default=None, help="Title for plots")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load data
    df_layer = pd.read_csv(args.per_layer_csv)
    df_matrix = pd.read_csv(args.per_matrix_csv)

    # 1. Compare total magnitude of changes in MLP vs Attention
    mlp_data = df_layer[df_layer['group'] == 'mlp']
    attn_data = df_layer[df_layer['group'] == 'attn']

    if not mlp_data.empty and not attn_data.empty:
        # Plot relative change magnitude
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Frobenius norm comparison
        ax1.plot(mlp_data['layer'], mlp_data['dW_fro_layer'], 'o-', label='MLP', color='blue')
        ax1.plot(attn_data['layer'], attn_data['dW_fro_layer'], 's-', label='Attention', color='orange')
        ax1.set_xlabel('Layer')
        ax1.set_ylabel(r'$\|\Delta W\|_F$')
        ax1.set_title('Weight Change Magnitude: MLP vs Attention')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Ratio plot
        merged = pd.merge(mlp_data[['layer', 'dW_fro_layer']],
                         attn_data[['layer', 'dW_fro_layer']],
                         on='layer', suffixes=('_mlp', '_attn'))
        merged['ratio'] = merged['dW_fro_layer_mlp'] / (merged['dW_fro_layer_attn'] + 1e-10)

        ax2.plot(merged['layer'], merged['ratio'], 'o-', color='green')
        ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Layer')
        ax2.set_ylabel(r'$\|\Delta W_{MLP}\|_F / \|\Delta W_{Attn}\|_F$')
        ax2.set_title('MLP/Attention Change Ratio')
        ax2.grid(alpha=0.3)

        plt.suptitle(args.title or 'MLP vs Attention Weight Changes')
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, 'mlp_vs_attn_magnitude.png'))
        plt.close()

    # 2. Analyze rank structure differences
    if 'mean_dW_stable_rank' in df_layer.columns:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Stable rank comparison
        if not mlp_data.empty and not attn_data.empty:
            ax = axes[0, 0]
            ax.plot(mlp_data['layer'], mlp_data['mean_dW_stable_rank'], 'o-', label='MLP', color='blue')
            ax.plot(attn_data['layer'], attn_data['mean_dW_stable_rank'], 's-', label='Attention', color='orange')
            ax.set_xlabel('Layer')
            ax.set_ylabel('Mean Stable Rank')
            ax.set_title('Stable Rank of Weight Changes')
            ax.legend()
            ax.grid(alpha=0.3)

            # Empirical rank comparison if available
            if 'mean_dW_empirical_rank' in df_layer.columns:
                ax = axes[0, 1]
                ax.plot(mlp_data['layer'], mlp_data['mean_dW_empirical_rank'], 'o-', label='MLP', color='blue')
                ax.plot(attn_data['layer'], attn_data['mean_dW_empirical_rank'], 's-', label='Attention', color='orange')
                ax.set_xlabel('Layer')
                ax.set_ylabel('Mean Empirical Rank')
                ax.set_title('Empirical Rank of Weight Changes')
                ax.legend()
                ax.grid(alpha=0.3)

                # Rank efficiency (empirical/stable ratio)
                ax = axes[1, 0]
                mlp_efficiency = mlp_data['mean_dW_empirical_rank'] / (mlp_data['mean_dW_stable_rank'] + 1e-10)
                attn_efficiency = attn_data['mean_dW_empirical_rank'] / (attn_data['mean_dW_stable_rank'] + 1e-10)
                ax.plot(mlp_data['layer'], mlp_efficiency, 'o-', label='MLP', color='blue')
                ax.plot(attn_data['layer'], attn_efficiency, 's-', label='Attention', color='orange')
                ax.set_xlabel('Layer')
                ax.set_ylabel('Empirical/Stable Rank Ratio')
                ax.set_title('Rank Efficiency (Higher = More Concentrated)')
                ax.legend()
                ax.grid(alpha=0.3)

        # Distribution of changes across layers
        ax = axes[1, 1]
        mlp_total = mlp_data['dW_fro_layer'].sum() if not mlp_data.empty else 0
        attn_total = attn_data['dW_fro_layer'].sum() if not attn_data.empty else 0

        if mlp_total > 0 or attn_total > 0:
            ax.bar(['MLP', 'Attention'], [mlp_total, attn_total], color=['blue', 'orange'])
            ax.set_ylabel(r'Total $\|\Delta W\|_F$ across layers')
            ax.set_title('Total Weight Change by Component')
            ax.grid(alpha=0.3, axis='y')

        plt.suptitle(args.title or 'Detailed MLP vs Attention Analysis')
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, 'mlp_vs_attn_detailed.png'))
        plt.close()

    # 3. Per-layer statistics summary
    summary = []
    for layer in sorted(df_layer['layer'].unique()):
        mlp_row = df_layer[(df_layer['layer'] == layer) & (df_layer['group'] == 'mlp')]
        attn_row = df_layer[(df_layer['layer'] == layer) & (df_layer['group'] == 'attn')]

        if not mlp_row.empty and not attn_row.empty:
            summary.append({
                'layer': layer,
                'mlp_fro': mlp_row['dW_fro_layer'].values[0],
                'attn_fro': attn_row['dW_fro_layer'].values[0],
                'ratio_mlp_attn': mlp_row['dW_fro_layer'].values[0] / (attn_row['dW_fro_layer'].values[0] + 1e-10),
                'mlp_stable_rank': mlp_row['mean_dW_stable_rank'].values[0] if 'mean_dW_stable_rank' in mlp_row else None,
                'attn_stable_rank': attn_row['mean_dW_stable_rank'].values[0] if 'mean_dW_stable_rank' in attn_row else None,
            })

    if summary:
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(os.path.join(args.outdir, 'mlp_attn_summary.csv'), index=False)

        # Print statistics
        print(f"\nMLP vs Attention Analysis Summary:")
        print(f"Average MLP/Attention change ratio: {summary_df['ratio_mlp_attn'].mean():.3f}")
        print(f"Layers where MLP changes more: {sum(summary_df['ratio_mlp_attn'] > 1)}/{len(summary_df)}")
        print(f"Max MLP dominance (layer {summary_df.loc[summary_df['ratio_mlp_attn'].idxmax(), 'layer']}): {summary_df['ratio_mlp_attn'].max():.3f}x")
        print(f"Max Attention dominance (layer {summary_df.loc[summary_df['ratio_mlp_attn'].idxmin(), 'layer']}): {1/summary_df['ratio_mlp_attn'].min():.3f}x")

    print(f"\nPlots saved to {args.outdir}")

if __name__ == "__main__":
    main()