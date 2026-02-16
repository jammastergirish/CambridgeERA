#!/usr/bin/env bash
set -euo pipefail

clear && printf '\e[3J'

# ---- Force flag: pass --force to rerun completed steps ----
FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
  echo "[pipeline] --force: will rerun all steps regardless of existing results"
fi

# Load .env (HF_TOKEN, etc.) if present
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Configuration — single output root for all results
OUTROOT="${OUTROOT:-outputs}"

# Models
BASE="EleutherAI/deep-ignorance-unfiltered"
FILTERED="EleutherAI/deep-ignorance-e2e-strong-filter"
UNLEARNED="EleutherAI/deep-ignorance-unfiltered-cb-lat"

# Comparison names
COMP1="EleutherAI_deep-ignorance-unfiltered__to__EleutherAI_deep-ignorance-e2e-strong-filter"
COMP2="EleutherAI_deep-ignorance-unfiltered__to__EleutherAI_deep-ignorance-unfiltered-cb-lat"

# Device and dtype settings
PARAM_DEVICE="${PARAM_DEVICE:-auto}"  # auto = cuda > mps > cpu
PARAM_DTYPE="${PARAM_DTYPE:-fp16}"
ACTIVATION_DEVICE="${ACTIVATION_DEVICE:-auto}"
ACTIVATION_DTYPE="${ACTIVATION_DTYPE:-auto}"

# Data paths
FORGET="${FORGET_TEXT:-data/forget.txt}"
RETAIN="${RETAIN_TEXT:-data/retain.txt}"

# ---- Skip-if-complete helper ----
# Usage: step_complete <dir> <sentinel_file>
# Returns 0 (true) if the sentinel exists and --force was not passed.
step_complete() {
  local dir="$1" sentinel="$2"
  if [[ "$FORCE" == "1" ]]; then return 1; fi
  [[ -f "${dir}/${sentinel}" ]]
}

echo "=========================================="
echo "      MODEL DIFFS ANALYSIS PIPELINE"
echo "=========================================="
echo ""
echo "Base model: $BASE"
echo "Comparison 1: -> $FILTERED"
echo "Comparison 2: -> $UNLEARNED"
echo "Output root: $OUTROOT"
echo ""

# # ============================================
# # STEP 1: Parameter Statistics
# # ============================================
# echo "=========================================="
# echo "STEP 1: Collecting Parameter Statistics"
# echo "=========================================="

# echo ""
# echo "Comparison 1: Base → Filtered"
# echo "----------------------------------------"
# if step_complete "${OUTROOT}/${COMP1}/param_stats" "per_layer.csv"; then
#   echo "  ✓ Already complete — skipping"
# else
#   uv run collect_param_stats.py \
#     --model-a "$BASE" \
#     --model-b "$FILTERED" \
#     --device "$PARAM_DEVICE" \
#     --dtype "$PARAM_DTYPE" \
#     --outdir "${OUTROOT}/${COMP1}/param_stats"
# fi

# echo ""
# echo "Comparison 2: Base → Unlearned"
# echo "----------------------------------------"
# if step_complete "${OUTROOT}/${COMP2}/param_stats" "per_layer.csv"; then
#   echo "  ✓ Already complete — skipping"
# else
#   uv run collect_param_stats.py \
#     --model-a "$BASE" \
#     --model-b "$UNLEARNED" \
#     --device "$PARAM_DEVICE" \
#     --dtype "$PARAM_DTYPE" \
#     --outdir "${OUTROOT}/${COMP2}/param_stats"
# fi

# # ============================================
# # STEP 2: Plot Parameter Statistics
# # ============================================
# echo ""
# echo "=========================================="
# echo "STEP 2: Plotting Parameter Statistics"
# echo "=========================================="

# echo ""
# echo "Plotting Comparison 1..."
# if step_complete "${OUTROOT}/${COMP1}/param_plots" "layer_locality_mlp.png"; then
#   echo "  ✓ Already complete — skipping"
# else
#   uv run plot_param_stats.py \
#     --per-layer-csv "${OUTROOT}/${COMP1}/param_stats/per_layer.csv" \
#     --outdir "${OUTROOT}/${COMP1}/param_plots" \
#     --title "EleutherAI/deep-ignorance-unfiltered → EleutherAI/deep-ignorance-e2e-strong-filter"
# fi

# echo ""
# echo "Plotting Comparison 2..."
# if step_complete "${OUTROOT}/${COMP2}/param_plots" "layer_locality_mlp.png"; then
#   echo "  ✓ Already complete — skipping"
# else
#   uv run plot_param_stats.py \
#     --per-layer-csv "${OUTROOT}/${COMP2}/param_stats/per_layer.csv" \
#     --outdir "${OUTROOT}/${COMP2}/param_plots" \
#     --title "EleutherAI/deep-ignorance-unfiltered → EleutherAI/deep-ignorance-unfiltered-cb-lat"
# fi

# # ============================================
# # STEP 3: Generate Test Datasets
# # ============================================
# echo ""
# echo "=========================================="
# echo "STEP 3: Generating Test Datasets"
# echo "=========================================="
# if [[ -f "$FORGET" && -f "$RETAIN" && "$FORCE" != "1" ]]; then
#   echo "  ✓ Datasets already exist — skipping"
# else
#   uv run create_datasets.py
# fi

# # ============================================
# # STEP 4: Activation Analysis
# # ============================================
# echo ""
# echo "=========================================="
# echo "STEP 4: Collecting Activation Norms"
# echo "=========================================="

# if [[ ! -f "$FORGET" || ! -f "$RETAIN" ]]; then
#   echo "Warning: Activation files missing; skipping activation analysis."
# else
#   echo ""
#   echo "Comparison 1: Base → Filtered"
#   echo "----------------------------------------"
#   if step_complete "${OUTROOT}/${COMP1}/activation_stats" "activation_stats.csv"; then
#     echo "  ✓ Already complete — skipping"
#   else
#     uv run collect_activation_norms.py \
#       --model-a "$BASE" \
#       --model-b "$FILTERED" \
#       --forget-text "$FORGET" \
#       --retain-text "$RETAIN" \
#       --device "$ACTIVATION_DEVICE" \
#       --dtype "$ACTIVATION_DTYPE" \
#       --outdir "${OUTROOT}/${COMP1}/activation_stats"
#   fi

#   echo ""
#   echo "Comparison 2: Base → Unlearned"
#   echo "----------------------------------------"
#   if step_complete "${OUTROOT}/${COMP2}/activation_stats" "activation_stats.csv"; then
#     echo "  ✓ Already complete — skipping"
#   else
#     uv run collect_activation_norms.py \
#       --model-a "$BASE" \
#       --model-b "$UNLEARNED" \
#       --forget-text "$FORGET" \
#       --retain-text "$RETAIN" \
#       --device "$ACTIVATION_DEVICE" \
#       --dtype "$ACTIVATION_DTYPE" \
#       --outdir "${OUTROOT}/${COMP2}/activation_stats"
#   fi
# fi

# # ============================================
# # STEP 5: Plot Activation Norms
# # ============================================
# echo ""
# echo "=========================================="
# echo "STEP 5: Plotting Activation Norms"
# echo "=========================================="
# if step_complete "${OUTROOT}/${COMP1}/activation_plots" "activation_norms_forget.png"; then
#   echo "  ✓ Already complete — skipping"
# else
#   uv run plot_activation_norms.py \
#     --indir "$OUTROOT" \
#     --outdir "$OUTROOT"
# fi

# # ============================================
# # STEP 6: MLP vs Attention Analysis
# # ============================================
# echo ""
# echo "=========================================="
# echo "STEP 6: MLP vs Attention Analysis"
# echo "=========================================="

# echo ""
# echo "Analyzing Comparison 1..."
# if step_complete "${OUTROOT}/${COMP1}/mlp_attn_analysis" "mlp_attn_summary.csv"; then
#   echo "  ✓ Already complete — skipping"
# else
#   uv run analyze_mlp_vs_attn.py \
#     --per-layer-csv "${OUTROOT}/${COMP1}/param_stats/per_layer.csv" \
#     --per-matrix-csv "${OUTROOT}/${COMP1}/param_stats/per_matrix.csv" \
#     --outdir "${OUTROOT}/${COMP1}/mlp_attn_analysis" \
#     --title "E2E Strong Filter: MLP vs Attention"
# fi

# echo ""
# echo "Analyzing Comparison 2..."
# if step_complete "${OUTROOT}/${COMP2}/mlp_attn_analysis" "mlp_attn_summary.csv"; then
#   echo "  ✓ Already complete — skipping"
# else
#   uv run analyze_mlp_vs_attn.py \
#     --per-layer-csv "${OUTROOT}/${COMP2}/param_stats/per_layer.csv" \
#     --per-matrix-csv "${OUTROOT}/${COMP2}/param_stats/per_matrix.csv" \
#     --outdir "${OUTROOT}/${COMP2}/mlp_attn_analysis" \
#     --title "CB-LAT: MLP vs Attention"
# fi

# # ============================================
# # STEP 7: Null Space & Subspace Analysis
# # ============================================
# echo ""
# echo "=========================================="
# echo "STEP 7: Null Space & Subspace Analysis"
# echo "=========================================="
# echo "Note: This is computationally intensive (SVD on 50 weight matrices)"

# echo ""
# echo "Analyzing Comparison 1..."
# if step_complete "${OUTROOT}/${COMP1}/null_space_analysis" "null_space_visualization.png"; then
#   echo "  ✓ Already complete — skipping"
# else
#   uv run null_space_analysis.py \
#     --model-a "$BASE" \
#     --model-b "$FILTERED" \
#     --outdir "${OUTROOT}/${COMP1}/null_space_analysis" \
#     --num-samples 50
# fi

# echo ""
# echo "Analyzing Comparison 2..."
# if step_complete "${OUTROOT}/${COMP2}/null_space_analysis" "null_space_visualization.png"; then
#   echo "  ✓ Already complete — skipping"
# else
#   uv run null_space_analysis.py \
#     --model-a "$BASE" \
#     --model-b "$UNLEARNED" \
#     --outdir "${OUTROOT}/${COMP2}/null_space_analysis" \
#     --num-samples 50
# fi

# # ============================================
# # STEP 8: Activation Separation Analysis
# # ============================================
# echo ""
# echo "=========================================="
# echo "STEP 8: Activation Separation Analysis"
# echo "=========================================="
# echo "Analyzing how well forget/retain activations are separated..."

# echo ""
# echo "Analyzing Comparison 1..."
# if step_complete "${OUTROOT}/${COMP1}/activation_separation" "summary.json"; then
#   echo "  ✓ Already complete — skipping"
# else
#   uv run activation_separation_analysis.py \
#     --model-a "$BASE" \
#     --model-b "$FILTERED" \
#     --forget-text "$FORGET" \
#     --retain-text "$RETAIN" \
#     --device "$ACTIVATION_DEVICE" \
#     --dtype "$ACTIVATION_DTYPE" \
#     --outdir "${OUTROOT}/${COMP1}/activation_separation"
# fi

# echo ""
# echo "Analyzing Comparison 2..."
# if step_complete "${OUTROOT}/${COMP2}/activation_separation" "summary.json"; then
#   echo "  ✓ Already complete — skipping"
# else
#   uv run activation_separation_analysis.py \
#     --model-a "$BASE" \
#     --model-b "$UNLEARNED" \
#     --forget-text "$FORGET" \
#     --retain-text "$RETAIN" \
#     --device "$ACTIVATION_DEVICE" \
#     --dtype "$ACTIVATION_DTYPE" \
#     --outdir "${OUTROOT}/${COMP2}/activation_separation"
# fi

# # ============================================
# # STEP 9: Activation Covariance Analysis
# # ============================================
# echo ""
# echo "=========================================="
# echo "STEP 9: Activation Covariance Analysis"
# echo "=========================================="
# echo "Analyzing covariance spectrum changes..."

# echo ""
# echo "Analyzing Comparison 1..."
# if step_complete "${OUTROOT}/${COMP1}/activation_covariance" "summary.json"; then
#   echo "  ✓ Already complete — skipping"
# else
#   uv run activation_covariance_analysis.py \
#     --model-a "$BASE" \
#     --model-b "$FILTERED" \
#     --forget-text "$FORGET" \
#     --retain-text "$RETAIN" \
#     --device "$ACTIVATION_DEVICE" \
#     --dtype "$ACTIVATION_DTYPE" \
#     --outdir "${OUTROOT}/${COMP1}/activation_covariance"
# fi

# echo ""
# echo "Analyzing Comparison 2..."
# if step_complete "${OUTROOT}/${COMP2}/activation_covariance" "summary.json"; then
#   echo "  ✓ Already complete — skipping"
# else
#   uv run activation_covariance_analysis.py \
#     --model-a "$BASE" \
#     --model-b "$UNLEARNED" \
#     --forget-text "$FORGET" \
#     --retain-text "$RETAIN" \
#     --device "$ACTIVATION_DEVICE" \
#     --dtype "$ACTIVATION_DTYPE" \
#     --outdir "${OUTROOT}/${COMP2}/activation_covariance"
# fi

# ============================================
# STEP 10: MLP Nullspace Alignment
# ============================================
echo ""
echo "=========================================="
echo "STEP 10: MLP Nullspace Alignment Analysis"
echo "=========================================="
echo "Analyzing if MLP updates align with nullspace..."

echo ""
echo "Analyzing Comparison 1..."
if step_complete "${OUTROOT}/${COMP1}/mlp_nullspace" "summary.json"; then
  echo "  ✓ Already complete — skipping"
else
  uv run mlp_nullspace_alignment.py \
    --model-a "$BASE" \
    --model-b "$FILTERED" \
    --device "$PARAM_DEVICE" \
    --dtype "$PARAM_DTYPE" \
    --outdir "${OUTROOT}/${COMP1}/mlp_nullspace"
fi

echo ""
echo "Analyzing Comparison 2..."
if step_complete "${OUTROOT}/${COMP2}/mlp_nullspace" "summary.json"; then
  echo "  ✓ Already complete — skipping"
else
  uv run mlp_nullspace_alignment.py \
    --model-a "$BASE" \
    --model-b "$UNLEARNED" \
    --device "$PARAM_DEVICE" \
    --dtype "$PARAM_DTYPE" \
    --outdir "${OUTROOT}/${COMP2}/mlp_nullspace"
fi

# ============================================
# STEP 11: Row Space Projection Analysis
# ============================================
echo ""
echo "=========================================="
echo "STEP 11: Row Space Projection Analysis"
echo "=========================================="
echo "Analyzing how activations project onto update directions..."

echo ""
echo "Analyzing Comparison 1..."
if step_complete "${OUTROOT}/${COMP1}/row_space_projection" "summary.json"; then
  echo "  ✓ Already complete — skipping"
else
  uv run row_space_projection_analysis.py \
    --model-a "$BASE" \
    --model-b "$FILTERED" \
    --forget-text "$FORGET" \
    --retain-text "$RETAIN" \
    --device "$ACTIVATION_DEVICE" \
    --dtype "$ACTIVATION_DTYPE" \
    --outdir "${OUTROOT}/${COMP1}/row_space_projection"
fi

echo ""
echo "Analyzing Comparison 2..."
if step_complete "${OUTROOT}/${COMP2}/row_space_projection" "summary.json"; then
  echo "  ✓ Already complete — skipping"
else
  uv run row_space_projection_analysis.py \
    --model-a "$BASE" \
    --model-b "$UNLEARNED" \
    --forget-text "$FORGET" \
    --retain-text "$RETAIN" \
    --device "$ACTIVATION_DEVICE" \
    --dtype "$ACTIVATION_DTYPE" \
    --outdir "${OUTROOT}/${COMP2}/row_space_projection"
fi

# ============================================
# STEP 12: Local Lipschitzness Analysis
# ============================================
echo ""
echo "=========================================="
echo "STEP 12: Local Lipschitzness Analysis"
echo "=========================================="
echo "Analyzing local smoothness changes..."

echo ""
echo "Analyzing Comparison 1..."
if step_complete "${OUTROOT}/${COMP1}/lipschitzness" "summary.json"; then
  echo "  ✓ Already complete — skipping"
else
  uv run local_lipschitzness_analysis.py \
    --model-a "$BASE" \
    --model-b "$FILTERED" \
    --forget-text "$FORGET" \
    --retain-text "$RETAIN" \
    --device "$ACTIVATION_DEVICE" \
    --dtype "$ACTIVATION_DTYPE" \
    --outdir "${OUTROOT}/${COMP1}/lipschitzness"
fi

echo ""
echo "Analyzing Comparison 2..."
if step_complete "${OUTROOT}/${COMP2}/lipschitzness" "summary.json"; then
  echo "  ✓ Already complete — skipping"
else
  uv run local_lipschitzness_analysis.py \
    --model-a "$BASE" \
    --model-b "$UNLEARNED" \
    --forget-text "$FORGET" \
    --retain-text "$RETAIN" \
    --device "$ACTIVATION_DEVICE" \
    --dtype "$ACTIVATION_DTYPE" \
    --outdir "${OUTROOT}/${COMP2}/lipschitzness"
fi

# ============================================
# COMPLETION
# ============================================
echo ""
echo "=========================================="
echo "        PIPELINE COMPLETE!"
echo "=========================================="
echo ""
echo "All results saved under: ${OUTROOT}/"
echo ""
echo "  <comparison>/"
echo "    param_stats/           per_matrix.csv, per_layer.csv"
echo "    param_plots/           Layer locality, stable rank PNGs"
echo "    activation_stats/      activation_stats.csv"
echo "    activation_plots/      Activation norms, diffs PNGs"
echo "    mlp_attn_analysis/     summary CSV + plots"
echo "    null_space_analysis/   null_space_results.csv + plots"
echo "    activation_separation/ separation metrics + plots"
echo "    activation_covariance/ covariance spectra + plots"
echo "    mlp_nullspace/         alignment metrics + plots"
echo "    row_space_projection/  projection metrics + plots"
echo "    lipschitzness/         Lipschitz estimates + plots"
echo ""
echo "Tip: rerun with --force to regenerate all results."
echo ""