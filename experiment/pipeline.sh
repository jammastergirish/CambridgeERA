#!/usr/bin/env bash
set -euo pipefail

# Always run from the project root (parent of experiment/)
cd "$(dirname "$0")/.."

clear && printf '\e[3J'

# ---- Force flag: pass --force to rerun completed steps ----
FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
  echo "[pipeline] --force: will rerun all steps regardless of existing results"
fi

# Load .env (HF_TOKEN, WANDB_API_KEY, etc.) if present
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Group all W&B runs from this pipeline invocation together
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-pipeline_$(date +%s)}"

# Configuration — single output root for all results
OUTROOT="${OUTROOT:-outputs}"

# Models (UNLEARNED can be overridden: UNLEARNED=user/model ./experiment/pipeline.sh)
BASE="EleutherAI/deep-ignorance-unfiltered"
FILTERED="EleutherAI/deep-ignorance-e2e-strong-filter"
UNLEARNED="${UNLEARNED:-EleutherAI/deep-ignorance-unfiltered-cb-lat}"

# Comparison names (derived from model IDs: / → _)
COMP1="EleutherAI_deep-ignorance-unfiltered__to__EleutherAI_deep-ignorance-e2e-strong-filter"
COMP2="EleutherAI_deep-ignorance-unfiltered__to__${UNLEARNED//\//_}"

# Per-model folder names (for analyses that run once per model, not per comparison)
MODEL_BASE="EleutherAI_deep-ignorance-unfiltered"
MODEL_FILTERED="EleutherAI_deep-ignorance-e2e-strong-filter"
MODEL_UNLEARNED="${UNLEARNED//\//_}"

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

# ============================================
# STEP 0: Benchmark Evaluation (per-model)
# ============================================
echo "=========================================="
echo "STEP 0: Benchmark Evaluation (MMLU, WMDP, HellaSwag, TruthfulQA)"
echo "=========================================="
echo "Quick sanity check — identifies collapsed models before expensive diagnostics."
echo "(Results stored per-model, not per-comparison)"

echo ""
echo "Model: $BASE"
echo "----------------------------------------"
if step_complete "${OUTROOT}/${MODEL_BASE}/evals" "summary.json"; then
  echo "  ✓ Already complete — skipping"
else
  uv run experiment/eval.py \
    --model "$BASE" \
    --device "$ACTIVATION_DEVICE" \
    --dtype "$ACTIVATION_DTYPE"
fi

echo ""
echo "Model: $FILTERED"
echo "----------------------------------------"
if step_complete "${OUTROOT}/${MODEL_FILTERED}/evals" "summary.json"; then
  echo "  ✓ Already complete — skipping"
else
  uv run experiment/eval.py \
    --model "$FILTERED" \
    --device "$ACTIVATION_DEVICE" \
    --dtype "$ACTIVATION_DTYPE"
fi

echo ""
echo "Model: $UNLEARNED"
echo "----------------------------------------"
if step_complete "${OUTROOT}/${MODEL_UNLEARNED}/evals" "summary.json"; then
  echo "  ✓ Already complete — skipping"
else
  uv run experiment/eval.py \
    --model "$UNLEARNED" \
    --device "$ACTIVATION_DEVICE" \
    --dtype "$ACTIVATION_DTYPE"
fi

# ============================================
# STEP 1: Parameter Statistics
# ============================================
echo "=========================================="
echo "STEP 1: Parameter Statistics (Collect + Plot)"
echo "=========================================="

echo ""
echo "Comparison 1: Base → Filtered"
echo "----------------------------------------"
if step_complete "${OUTROOT}/${COMP1}/param_stats" "per_layer.csv"; then
  echo "  ✓ Already complete — skipping"
else
  uv run experiment/param_stats.py \
    --model-a "$BASE" \
    --model-b "$FILTERED" \
    --device "$PARAM_DEVICE" \
    --dtype "$PARAM_DTYPE" \
    --title "EleutherAI/deep-ignorance-unfiltered → EleutherAI/deep-ignorance-e2e-strong-filter"
fi

echo ""
echo "Comparison 2: Base → Unlearned"
echo "----------------------------------------"
if step_complete "${OUTROOT}/${COMP2}/param_stats" "per_layer.csv"; then
  echo "  ✓ Already complete — skipping"
else
  uv run experiment/param_stats.py \
    --model-a "$BASE" \
    --model-b "$UNLEARNED" \
    --device "$PARAM_DEVICE" \
    --dtype "$PARAM_DTYPE" \
    --title "$BASE → $UNLEARNED"
fi

# ============================================
# STEP 2: Generate Test Datasets
# ============================================
echo ""
echo "=========================================="
echo "STEP 2: Generating Test Datasets"
echo "=========================================="
if [[ -f "$FORGET" && -f "$RETAIN" && "$FORCE" != "1" ]]; then
  echo "  ✓ Datasets already exist — skipping"
else
  uv run create_datasets.py
fi

# ============================================
# STEP 3: Activation Norms
# ============================================
echo ""
echo "=========================================="
echo "STEP 3: Activation Norms"
echo "=========================================="

if [[ ! -f "$FORGET" || ! -f "$RETAIN" ]]; then
  echo "Warning: Activation files missing; skipping activation analysis."
else
  echo ""
  echo "Comparison 1: Base → Filtered"
  echo "----------------------------------------"
  if step_complete "${OUTROOT}/${COMP1}/activation_stats" "activation_stats.csv"; then
    echo "  ✓ Already complete — skipping"
  else
    uv run experiment/activation_norms.py \
      --model-a "$BASE" \
      --model-b "$FILTERED" \
      --forget-text "$FORGET" \
      --retain-text "$RETAIN" \
      --device "$ACTIVATION_DEVICE" \
      --dtype "$ACTIVATION_DTYPE" \
      --title "E2E Strong Filter: Activation Norms"
  fi

  echo ""
  echo "Comparison 2: Base → Unlearned"
  echo "----------------------------------------"
  if step_complete "${OUTROOT}/${COMP2}/activation_stats" "activation_stats.csv"; then
    echo "  ✓ Already complete — skipping"
  else
    uv run experiment/activation_norms.py \
      --model-a "$BASE" \
      --model-b "$UNLEARNED" \
      --forget-text "$FORGET" \
      --retain-text "$RETAIN" \
      --device "$ACTIVATION_DEVICE" \
      --dtype "$ACTIVATION_DTYPE" \
      --title "${UNLEARNED##*/}: Activation Norms"
  fi
fi

# ============================================
# STEP 4: MLP vs Attention Analysis
# ============================================
echo ""
echo "=========================================="
echo "STEP 4: MLP vs Attention Analysis"
echo "=========================================="

echo ""
echo "Analyzing Comparison 1..."
if step_complete "${OUTROOT}/${COMP1}/mlp_attn_analysis" "mlp_attn_summary.csv"; then
  echo "  ✓ Already complete — skipping"
else
  uv run experiment/analyze_mlp_vs_attn.py \
    --per-layer-csv "${OUTROOT}/${COMP1}/param_stats/per_layer.csv" \
    --per-matrix-csv "${OUTROOT}/${COMP1}/param_stats/per_matrix.csv" \
    --outdir "${OUTROOT}/${COMP1}/mlp_attn_analysis" \
    --title "E2E Strong Filter: MLP vs Attention"
fi

echo ""
echo "Analyzing Comparison 2..."
if step_complete "${OUTROOT}/${COMP2}/mlp_attn_analysis" "mlp_attn_summary.csv"; then
  echo "  ✓ Already complete — skipping"
else
  uv run experiment/analyze_mlp_vs_attn.py \
    --per-layer-csv "${OUTROOT}/${COMP2}/param_stats/per_layer.csv" \
    --per-matrix-csv "${OUTROOT}/${COMP2}/param_stats/per_matrix.csv" \
    --outdir "${OUTROOT}/${COMP2}/mlp_attn_analysis" \
    --title "${UNLEARNED##*/}: MLP vs Attention"
fi

# ============================================
# STEP 5: Null Space & Subspace Analysis
# ============================================
echo ""
echo "=========================================="
echo "STEP 5: Null Space & Subspace Analysis"
echo "=========================================="
echo "Note: This is computationally intensive (SVD on 50 weight matrices)"

echo ""
echo "Analyzing Comparison 1..."
if step_complete "${OUTROOT}/${COMP1}/null_space_analysis" "null_space_visualization.png"; then
  echo "  ✓ Already complete — skipping"
else
  uv run experiment/null_space_analysis.py \
    --model-a "$BASE" \
    --model-b "$FILTERED" \
    --num-samples 50
fi

echo ""
echo "Analyzing Comparison 2..."
if step_complete "${OUTROOT}/${COMP2}/null_space_analysis" "null_space_visualization.png"; then
  echo "  ✓ Already complete — skipping"
else
  uv run experiment/null_space_analysis.py \
    --model-a "$BASE" \
    --model-b "$UNLEARNED" \
    --num-samples 50
fi

# ============================================
# STEP 6: Activation Separation Analysis
# ============================================
echo ""
echo "=========================================="
echo "STEP 6: Activation Separation Analysis"
echo "=========================================="
echo "Analyzing how well forget/retain activations are separated..."

echo ""
echo "Analyzing Comparison 1..."
if step_complete "${OUTROOT}/${COMP1}/activation_separation" "summary.json"; then
  echo "  ✓ Already complete — skipping"
else
  uv run experiment/activation_separation_analysis.py \
    --model-a "$BASE" \
    --model-b "$FILTERED" \
    --forget-text "$FORGET" \
    --retain-text "$RETAIN" \
    --device "$ACTIVATION_DEVICE" \
    --dtype "$ACTIVATION_DTYPE"
fi

echo ""
echo "Analyzing Comparison 2..."
if step_complete "${OUTROOT}/${COMP2}/activation_separation" "summary.json"; then
  echo "  ✓ Already complete — skipping"
else
  uv run experiment/activation_separation_analysis.py \
    --model-a "$BASE" \
    --model-b "$UNLEARNED" \
    --forget-text "$FORGET" \
    --retain-text "$RETAIN" \
    --device "$ACTIVATION_DEVICE" \
    --dtype "$ACTIVATION_DTYPE"
fi

# ============================================
# STEP 7: Activation Covariance Analysis
# ============================================
echo ""
echo "=========================================="
echo "STEP 7: Activation Covariance Analysis"
echo "=========================================="
echo "Analyzing covariance spectrum changes..."

echo ""
echo "Analyzing Comparison 1..."
if step_complete "${OUTROOT}/${COMP1}/activation_covariance" "summary.json"; then
  echo "  ✓ Already complete — skipping"
else
  uv run experiment/activation_covariance_analysis.py \
    --model-a "$BASE" \
    --model-b "$FILTERED" \
    --forget-text "$FORGET" \
    --retain-text "$RETAIN" \
    --device "$ACTIVATION_DEVICE" \
    --dtype "$ACTIVATION_DTYPE"
fi

echo ""
echo "Analyzing Comparison 2..."
if step_complete "${OUTROOT}/${COMP2}/activation_covariance" "summary.json"; then
  echo "  ✓ Already complete — skipping"
else
  uv run experiment/activation_covariance_analysis.py \
    --model-a "$BASE" \
    --model-b "$UNLEARNED" \
    --forget-text "$FORGET" \
    --retain-text "$RETAIN" \
    --device "$ACTIVATION_DEVICE" \
    --dtype "$ACTIVATION_DTYPE"
fi

# ============================================
# STEP 8: MLP Nullspace Alignment
# ============================================
echo ""
echo "=========================================="
echo "STEP 8: MLP Nullspace Alignment Analysis"
echo "=========================================="
echo "Analyzing if MLP updates align with nullspace..."

echo ""
echo "Analyzing Comparison 1..."
if step_complete "${OUTROOT}/${COMP1}/mlp_nullspace_alignment" "summary.json"; then
  echo "  ✓ Already complete — skipping"
else
  uv run experiment/mlp_nullspace_alignment.py \
    --model-a "$BASE" \
    --model-b "$FILTERED" \
    --device "$PARAM_DEVICE" \
    --dtype "$PARAM_DTYPE"
fi

echo ""
echo "Analyzing Comparison 2..."
if step_complete "${OUTROOT}/${COMP2}/mlp_nullspace_alignment" "summary.json"; then
  echo "  ✓ Already complete — skipping"
else
  uv run experiment/mlp_nullspace_alignment.py \
    --model-a "$BASE" \
    --model-b "$UNLEARNED" \
    --device "$PARAM_DEVICE" \
    --dtype "$PARAM_DTYPE"
fi

# ============================================
# STEP 9: Row Space Projection Analysis
# ============================================
echo ""
echo "=========================================="
echo "STEP 9: Row Space Projection Analysis"
echo "=========================================="
echo "Analyzing how activations project onto update directions..."

echo ""
echo "Analyzing Comparison 1..."
if step_complete "${OUTROOT}/${COMP1}/row_space_projection" "summary.json"; then
  echo "  ✓ Already complete — skipping"
else
  uv run experiment/row_space_projection_analysis.py \
    --model-a "$BASE" \
    --model-b "$FILTERED" \
    --forget-text "$FORGET" \
    --retain-text "$RETAIN" \
    --device "$ACTIVATION_DEVICE" \
    --dtype "$ACTIVATION_DTYPE"
fi

echo ""
echo "Analyzing Comparison 2..."
if step_complete "${OUTROOT}/${COMP2}/row_space_projection" "summary.json"; then
  echo "  ✓ Already complete — skipping"
else
  uv run experiment/row_space_projection_analysis.py \
    --model-a "$BASE" \
    --model-b "$UNLEARNED" \
    --forget-text "$FORGET" \
    --retain-text "$RETAIN" \
    --device "$ACTIVATION_DEVICE" \
    --dtype "$ACTIVATION_DTYPE"
fi

# ============================================
# STEP 10: Local Lipschitzness Analysis
# ============================================
echo ""
echo "=========================================="
echo "STEP 10: Local Lipschitzness Analysis"
echo "=========================================="
echo "Analyzing local smoothness changes..."

echo ""
echo "Analyzing Comparison 1..."
if step_complete "${OUTROOT}/${COMP1}/lipschitzness_analysis" "summary.json"; then
  echo "  ✓ Already complete — skipping"
else
  uv run experiment/local_lipschitzness_analysis.py \
    --model-a "$BASE" \
    --model-b "$FILTERED" \
    --forget-text "$FORGET" \
    --retain-text "$RETAIN" \
    --device "$ACTIVATION_DEVICE" \
    --dtype "$ACTIVATION_DTYPE"
fi

echo ""
echo "Analyzing Comparison 2..."
if step_complete "${OUTROOT}/${COMP2}/lipschitzness_analysis" "summary.json"; then
  echo "  ✓ Already complete — skipping"
else
  uv run experiment/local_lipschitzness_analysis.py \
    --model-a "$BASE" \
    --model-b "$UNLEARNED" \
    --forget-text "$FORGET" \
    --retain-text "$RETAIN" \
    --device "$ACTIVATION_DEVICE" \
    --dtype "$ACTIVATION_DTYPE"
fi

# ============================================
# STEP 11: Linear Probe Analysis (per-model)
# ============================================
echo ""
echo "=========================================="
echo "STEP 11: Linear Probe Analysis"
echo "=========================================="
echo "Training per-layer linear probes to locate forget-set knowledge..."
echo "(Results stored per-model, not per-comparison)"

echo ""
echo "Model: $BASE"
echo "----------------------------------------"
if step_complete "${OUTROOT}/${MODEL_BASE}/linear_probes" "summary.json"; then
  echo "  ✓ Already complete — skipping"
else
  uv run experiment/linear_probe_analysis.py \
    --model "$BASE" \
    --forget-text "$FORGET" \
    --retain-text "$RETAIN" \
    --device "$ACTIVATION_DEVICE" \
    --dtype "$ACTIVATION_DTYPE"
fi

echo ""
echo "Model: $FILTERED"
echo "----------------------------------------"
if step_complete "${OUTROOT}/${MODEL_FILTERED}/linear_probes" "summary.json"; then
  echo "  ✓ Already complete — skipping"
else
  uv run experiment/linear_probe_analysis.py \
    --model "$FILTERED" \
    --forget-text "$FORGET" \
    --retain-text "$RETAIN" \
    --device "$ACTIVATION_DEVICE" \
    --dtype "$ACTIVATION_DTYPE"
fi

echo ""
echo "Model: $UNLEARNED"
echo "----------------------------------------"
if step_complete "${OUTROOT}/${MODEL_UNLEARNED}/linear_probes" "summary.json"; then
  echo "  ✓ Already complete — skipping"
else
  uv run experiment/linear_probe_analysis.py \
    --model "$UNLEARNED" \
    --forget-text "$FORGET" \
    --retain-text "$RETAIN" \
    --device "$ACTIVATION_DEVICE" \
    --dtype "$ACTIVATION_DTYPE"
fi

# ============================================
# STEP 12: Layer-wise WMDP Accuracy (per-model)
# ============================================
echo ""
echo "=========================================="
echo "STEP 12: Layer-wise WMDP Accuracy (Logit + Tuned Lens)"
echo "=========================================="
echo "Measuring WMDP-Bio MCQ accuracy at every transformer layer..."
echo "(Results stored per-model, not per-comparison)"

for LENS in logit tuned; do
  echo ""
  echo "--- Lens: ${LENS} ---"

  echo ""
  echo "Model: $BASE"
  echo "----------------------------------------"
  if step_complete "${OUTROOT}/${MODEL_BASE}/wmdp_${LENS}_lens" "summary.json"; then
    echo "  ✓ Already complete — skipping"
  else
    uv run experiment/layerwise_wmdp_accuracy.py \
      --model "$BASE" \
      --lens "$LENS" \
      --device "$ACTIVATION_DEVICE" \
      --dtype "$ACTIVATION_DTYPE"
  fi

  echo ""
  echo "Model: $FILTERED"
  echo "----------------------------------------"
  if step_complete "${OUTROOT}/${MODEL_FILTERED}/wmdp_${LENS}_lens" "summary.json"; then
    echo "  ✓ Already complete — skipping"
  else
    uv run experiment/layerwise_wmdp_accuracy.py \
      --model "$FILTERED" \
      --lens "$LENS" \
      --device "$ACTIVATION_DEVICE" \
      --dtype "$ACTIVATION_DTYPE"
  fi

  echo ""
  echo "Model: $UNLEARNED"
  echo "----------------------------------------"
  if step_complete "${OUTROOT}/${MODEL_UNLEARNED}/wmdp_${LENS}_lens" "summary.json"; then
    echo "  ✓ Already complete — skipping"
  else
    uv run experiment/layerwise_wmdp_accuracy.py \
      --model "$UNLEARNED" \
      --lens "$LENS" \
      --device "$ACTIVATION_DEVICE" \
      --dtype "$ACTIVATION_DTYPE"
  fi
done

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
echo "    mlp_nullspace_alignment/ alignment metrics + plots"
echo "    row_space_projection/  projection metrics + plots"
echo "    lipschitzness_analysis/  Lipschitz estimates + plots"
echo ""
echo "  <model>/"
echo "    evals/                 summary.json (MMLU, WMDP, HellaSwag, TruthfulQA)"
echo "    linear_probes/         probe_results.csv, summary.json + plot"
echo "    wmdp_logit_lens/       wmdp_lens_results.csv, summary.json + plot"
echo "    wmdp_tuned_lens/       wmdp_lens_results.csv, summary.json + plot"
echo ""
echo "Tip: rerun with --force to regenerate all results."
echo ""