#!/usr/bin/env bash
set -euo pipefail

clear && printf '\e[3J'
rm -rf outputs plots

# Configuration
OUTROOT="${OUTROOT:-outputs}"
PLOTROOT="${PLOTROOT:-plots}"

# Models
BASE="EleutherAI/deep-ignorance-unfiltered"
FILTERED="EleutherAI/deep-ignorance-e2e-strong-filter"
UNLEARNED="EleutherAI/deep-ignorance-unfiltered-cb-lat"

# Comparison names
COMP1="EleutherAI_deep-ignorance-unfiltered__to__EleutherAI_deep-ignorance-e2e-strong-filter"
COMP2="EleutherAI_deep-ignorance-unfiltered__to__EleutherAI_deep-ignorance-unfiltered-cb-lat"

# Device and dtype settings
PARAM_DEVICE="${PARAM_DEVICE:-cpu}"  # CPU + fp16 safer for Mac on param stats
PARAM_DTYPE="${PARAM_DTYPE:-fp16}"
ACTIVATION_DEVICE="${ACTIVATION_DEVICE:-auto}"
ACTIVATION_DTYPE="${ACTIVATION_DTYPE:-auto}"

# Data paths
FORGET="${FORGET_TEXT:-data/forget.txt}"
RETAIN="${RETAIN_TEXT:-data/retain.txt}"

echo "=========================================="
echo "      MODEL DIFFS ANALYSIS PIPELINE"
echo "=========================================="
echo ""
echo "Base model: $BASE"
echo "Comparison 1: -> $FILTERED"
echo "Comparison 2: -> $UNLEARNED"
echo ""

# ============================================
# STEP 1: Parameter Statistics
# ============================================
echo "=========================================="
echo "STEP 1: Collecting Parameter Statistics"
echo "=========================================="

echo ""
echo "Comparison 1: Base → Filtered"
echo "----------------------------------------"
uv run collect_param_stats.py \
  --model-a "$BASE" \
  --model-b "$FILTERED" \
  --device "$PARAM_DEVICE" \
  --dtype "$PARAM_DTYPE" \
  --outdir "${OUTROOT}/${COMP1}/param_stats"

echo ""
echo "Comparison 2: Base → Unlearned"
echo "----------------------------------------"
uv run collect_param_stats.py \
  --model-a "$BASE" \
  --model-b "$UNLEARNED" \
  --device "$PARAM_DEVICE" \
  --dtype "$PARAM_DTYPE" \
  --outdir "${OUTROOT}/${COMP2}/param_stats"

# ============================================
# STEP 2: Plot Parameter Statistics
# ============================================
echo ""
echo "=========================================="
echo "STEP 2: Plotting Parameter Statistics"
echo "=========================================="

echo ""
echo "Plotting Comparison 1..."
uv run plot_param_stats.py \
  --per-layer-csv "${OUTROOT}/${COMP1}/param_stats/per_layer.csv" \
  --outdir "${PLOTROOT}/${COMP1}/param_plots" \
  --title "EleutherAI/deep-ignorance-unfiltered → EleutherAI/deep-ignorance-e2e-strong-filter"

echo ""
echo "Plotting Comparison 2..."
uv run plot_param_stats.py \
  --per-layer-csv "${OUTROOT}/${COMP2}/param_stats/per_layer.csv" \
  --outdir "${PLOTROOT}/${COMP2}/param_plots" \
  --title "EleutherAI/deep-ignorance-unfiltered → EleutherAI/deep-ignorance-unfiltered-cb-lat"

# ============================================
# STEP 3: Generate Test Datasets
# ============================================
echo ""
echo "=========================================="
echo "STEP 3: Generating Test Datasets"
echo "=========================================="
uv run create_datasets.py

# ============================================
# STEP 4: Activation Analysis
# ============================================
echo ""
echo "=========================================="
echo "STEP 4: Collecting Activation Norms"
echo "=========================================="

if [[ ! -f "$FORGET" || ! -f "$RETAIN" ]]; then
  echo "Warning: Activation files missing; skipping activation analysis."
else
  echo ""
  echo "Comparison 1: Base → Filtered"
  echo "----------------------------------------"
  uv run collect_activation_norms.py \
    --model-a "$BASE" \
    --model-b "$FILTERED" \
    --forget-text "$FORGET" \
    --retain-text "$RETAIN" \
    --device "$ACTIVATION_DEVICE" \
    --dtype "$ACTIVATION_DTYPE" \
    --outdir "${OUTROOT}/${COMP1}/activation_stats"

  echo ""
  echo "Comparison 2: Base → Unlearned"
  echo "----------------------------------------"
  uv run collect_activation_norms.py \
    --model-a "$BASE" \
    --model-b "$UNLEARNED" \
    --forget-text "$FORGET" \
    --retain-text "$RETAIN" \
    --device "$ACTIVATION_DEVICE" \
    --dtype "$ACTIVATION_DTYPE" \
    --outdir "${OUTROOT}/${COMP2}/activation_stats"
fi

# ============================================
# STEP 5: Plot Activation Norms
# ============================================
echo ""
echo "=========================================="
echo "STEP 5: Plotting Activation Norms"
echo "=========================================="
uv run plot_activation_norms.py \
  --indir "$OUTROOT" \
  --outdir "$PLOTROOT"

# ============================================
# STEP 6: MLP vs Attention Analysis
# ============================================
echo ""
echo "=========================================="
echo "STEP 6: MLP vs Attention Analysis"
echo "=========================================="

echo ""
echo "Analyzing Comparison 1..."
uv run analyze_mlp_vs_attn.py \
  --per-layer-csv "${OUTROOT}/${COMP1}/param_stats/per_layer.csv" \
  --per-matrix-csv "${OUTROOT}/${COMP1}/param_stats/per_matrix.csv" \
  --outdir "${OUTROOT}/${COMP1}/mlp_attn_analysis" \
  --title "E2E Strong Filter: MLP vs Attention"

echo ""
echo "Analyzing Comparison 2..."
uv run analyze_mlp_vs_attn.py \
  --per-layer-csv "${OUTROOT}/${COMP2}/param_stats/per_layer.csv" \
  --per-matrix-csv "${OUTROOT}/${COMP2}/param_stats/per_matrix.csv" \
  --outdir "${OUTROOT}/${COMP2}/mlp_attn_analysis" \
  --title "CB-LAT: MLP vs Attention"

# ============================================
# STEP 7: Null Space & Subspace Analysis
# ============================================
echo ""
echo "=========================================="
echo "STEP 7: Null Space & Subspace Analysis"
echo "=========================================="
echo "Note: This is computationally intensive (SVD on 50 weight matrices)"

echo ""
echo "Analyzing Comparison 1..."
uv run null_space_analysis.py \
  --model-a "$BASE" \
  --model-b "$FILTERED" \
  --outdir "${OUTROOT}/${COMP1}/null_space_analysis" \
  --num-samples 50

echo ""
echo "Analyzing Comparison 2..."
uv run null_space_analysis.py \
  --model-a "$BASE" \
  --model-b "$UNLEARNED" \
  --outdir "${OUTROOT}/${COMP2}/null_space_analysis" \
  --num-samples 50

# ============================================
# COMPLETION
# ============================================
echo ""
echo "=========================================="
echo "        PIPELINE COMPLETE!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - Parameter stats: ${OUTROOT}/*/param_stats/"
echo "  - Activation stats: ${OUTROOT}/*/activation_stats/"
echo "  - MLP/Attn analysis: ${OUTROOT}/*/mlp_attn_analysis/"
echo "  - Null space analysis: ${OUTROOT}/*/null_space_analysis/"
echo ""
echo "Plots saved to:"
echo "  - Parameter plots: ${PLOTROOT}/*/param_plots/"
echo "  - Activation plots: ${PLOTROOT}/*/activation_plots/"
echo ""