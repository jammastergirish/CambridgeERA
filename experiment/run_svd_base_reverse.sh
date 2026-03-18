#!/usr/bin/env bash
# Run singular_value_spectrum_analysis.py for all unlearned models on
# huggingface.co/girishgupta against the BASE model, in REVERSE order.
# Intended to run on a second GPU while run_svd_all_hf_models.sh runs forward.
#
# Usage (from project root):
#   DEVICE=cuda:1 ./experiment/run_svd_base_reverse.sh [--force]

set -uo pipefail

cd "$(dirname "$0")/.."

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
  echo "[run_svd_reverse] --force: will rerun all steps regardless of existing results"
fi

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

DEVICE="${DEVICE:-cuda:1}"
DTYPE="${DTYPE:-fp32}"

MODEL_A="EleutherAI/deep-ignorance-unfiltered"

# Same models as the main script, but in REVERSE order
MODELS=(
  "girishgupta/deep-ignorance-unfiltered_unlearned_cb_lat"
  "girishgupta/deep-ignorance-unfiltered_unlearned_dpo"
  "girishgupta/deep-ignorance-unfiltered_unlearned_ga"
  "girishgupta/deep-ignorance-unfiltered_unlearned_ga_simple"
  "girishgupta/deep-ignorance-unfiltered_unlearned_grad_diff"
  "girishgupta/deep-ignorance-unfiltered_unlearned_lat"
  "girishgupta/deep-ignorance-unfiltered_unlearned_npo"
  "girishgupta/deep-ignorance-unfiltered_unlearned_rmu"
  "girishgupta/deep-ignorance-unfiltered_unlearned_simnpo"
  "girishgupta/deep-ignorance-unfiltered_unlearned_wt_dist"
)

TOTAL=${#MODELS[@]}
PASSED=0
FAILED=0

for i in "${!MODELS[@]}"; do
  MODEL_B="${MODELS[$i]}"
  N=$((i + 1))

  MODEL_A_SLUG="${MODEL_A//\//_}"
  MODEL_B_SLUG="${MODEL_B//\//_}"
  OUTDIR="outputs/${MODEL_A_SLUG}__to__${MODEL_B_SLUG}/sv_spectrum"
  SENTINEL="${OUTDIR}/sv_spectrum.png"

  echo ""
  echo "========================================"
  echo "[$N/$TOTAL] SVD spectrum: $MODEL_A → $MODEL_B"
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

echo ""
echo "========================================"
echo "Done. Passed: $PASSED / $TOTAL, Failed: $FAILED / $TOTAL"
echo "========================================"
