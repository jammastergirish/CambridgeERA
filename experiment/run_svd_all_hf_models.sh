#!/usr/bin/env bash
# Run singular_value_spectrum_analysis.py for all unlearned models on
# huggingface.co/girishgupta against TWO baselines:
#   1. EleutherAI/deep-ignorance-unfiltered        (base model)
#   2. EleutherAI/deep-ignorance-e2e-strong-filter  (filtered model)
#
# Usage (from project root):
#   ./experiment/run_svd_all_hf_models.sh [--force]

set -uo pipefail

cd "$(dirname "$0")/.."

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
  echo "[run_svd] --force: will rerun all steps regardless of existing results"
fi

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-fp32}"

BASELINES=(
  "EleutherAI/deep-ignorance-unfiltered"
  "EleutherAI/deep-ignorance-e2e-strong-filter"
)

MODELS=(
  "girishgupta/deep-ignorance-unfiltered_unlearned_wt_dist"
  "girishgupta/deep-ignorance-unfiltered_unlearned_simnpo"
  "girishgupta/deep-ignorance-unfiltered_unlearned_rmu"
  "girishgupta/deep-ignorance-unfiltered_unlearned_npo"
  "girishgupta/deep-ignorance-unfiltered_unlearned_lat"
  "girishgupta/deep-ignorance-unfiltered_unlearned_grad_diff"
  "girishgupta/deep-ignorance-unfiltered_unlearned_ga_simple"
  "girishgupta/deep-ignorance-unfiltered_unlearned_ga"
  "girishgupta/deep-ignorance-unfiltered_unlearned_dpo"
  "girishgupta/deep-ignorance-unfiltered_unlearned_cb_lat"
)

NUM_BASELINES=${#BASELINES[@]}
NUM_MODELS=${#MODELS[@]}
TOTAL=$((NUM_BASELINES * NUM_MODELS))
PASSED=0
FAILED=0
RUN=0

for MODEL_A in "${BASELINES[@]}"; do
  echo ""
  echo "########################################"
  echo "# Baseline: $MODEL_A"
  echo "########################################"

  for i in "${!MODELS[@]}"; do
    MODEL_B="${MODELS[$i]}"
    ((RUN++))

    MODEL_A_SLUG="${MODEL_A//\//_}"
    MODEL_B_SLUG="${MODEL_B//\//_}"
    OUTDIR="outputs/${MODEL_A_SLUG}__to__${MODEL_B_SLUG}/sv_spectrum"
    SENTINEL="${OUTDIR}/sv_spectrum.png"

    echo ""
    echo "========================================"
    echo "[$RUN/$TOTAL] SVD spectrum: $MODEL_A → $MODEL_B"
    echo "----------------------------------------"

    if [[ "$FORCE" == "0" ]] && [[ -f "$SENTINEL" ]]; then
      echo "  ✓ Already complete — skipping"
      ((PASSED++))
      continue
    fi

    if uv run experiment/singular_value_spectrum_analysis.py \
      --model-a "$MODEL_A" \
      --model-b "$MODEL_B" \
      --device "$DEVICE" \
      --dtype "$DTYPE" \
      --outdir "$OUTDIR" \
      --title "${MODEL_B##*/}: SV Spectrum (vs ${MODEL_A##*/})"; then
      echo "[OK] $MODEL_B"
      ((PASSED++))
    else
      echo "[FAIL] $MODEL_B (exit $?). Continuing..."
      ((FAILED++))
    fi
  done
done

echo ""
echo "========================================"
echo "Done. Passed: $PASSED / $TOTAL, Failed: $FAILED / $TOTAL"
echo "========================================"
