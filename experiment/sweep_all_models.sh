#!/usr/bin/env bash
set -euo pipefail

# Run the full analysis pipeline for every unlearned model on HuggingFace.
#
# The Base→Filtered comparison and per-model steps for Base/Filtered are
# shared across all runs and will be skipped automatically once completed.
#
# Usage:
#   ./experiment/sweep_all_models.sh            # run all models
#   ./experiment/sweep_all_models.sh --force     # rerun everything
#   ./experiment/sweep_all_models.sh ga dpo rmu  # run only these methods

cd "$(dirname "$0")/.."

# ---- Parse args ----
FORCE=""
METHODS=()
for arg in "$@"; do
  if [[ "$arg" == "--force" ]]; then
    FORCE="--force"
  else
    METHODS+=("$arg")
  fi
done

# All 12 unlearned models on HuggingFace (girishgupta/*)
ALL_MODELS=(
  "girishgupta/EleutherAI_deep-ignorance-unfiltered__ga_simple"
  "girishgupta/EleutherAI_deep-ignorance-unfiltered__ga"
  "girishgupta/EleutherAI_deep-ignorance-unfiltered__grad_diff"
  "girishgupta/EleutherAI_deep-ignorance-unfiltered__dpo"
  "girishgupta/EleutherAI_deep-ignorance-unfiltered__npo"
  "girishgupta/EleutherAI_deep-ignorance-unfiltered__simnpo"
  "girishgupta/EleutherAI_deep-ignorance-unfiltered__rmu"
  "girishgupta/EleutherAI_deep-ignorance-unfiltered__cb"
  "girishgupta/EleutherAI_deep-ignorance-unfiltered__cb_lat"
  "girishgupta/EleutherAI_deep-ignorance-unfiltered__lat"
  "girishgupta/EleutherAI_deep-ignorance-unfiltered__wt_dist"
  "girishgupta/EleutherAI_deep-ignorance-unfiltered__wt_dist_reg"
)

# Filter to requested methods if any were specified
if [[ ${#METHODS[@]} -gt 0 ]]; then
  MODELS=()
  for method in "${METHODS[@]}"; do
    for model in "${ALL_MODELS[@]}"; do
      if [[ "$model" == *"__${method}" ]]; then
        MODELS+=("$model")
      fi
    done
  done
  if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "Error: no models matched methods: ${METHODS[*]}" >&2
    echo "Available: ga_simple ga grad_diff dpo npo simnpo rmu cb cb_lat lat wt_dist wt_dist_reg" >&2
    exit 1
  fi
else
  MODELS=("${ALL_MODELS[@]}")
fi

echo "=========================================="
echo "   SWEEP: ${#MODELS[@]} UNLEARNED MODEL(S)"
echo "=========================================="
echo ""
for m in "${MODELS[@]}"; do
  echo "  • $m"
done
echo ""

TOTAL=${#MODELS[@]}
CURRENT=0
FAILED=()

for model in "${MODELS[@]}"; do
  CURRENT=$((CURRENT + 1))
  method="${model##*__}"

  echo ""
  echo "=========================================="
  echo "  [$CURRENT/$TOTAL] $method"
  echo "  $model"
  echo "=========================================="
  echo ""

  if UNLEARNED="$model" ./experiment/pipeline.sh $FORCE; then
    echo ""
    echo "  ✓ $method complete"
  else
    echo ""
    echo "  ✗ $method FAILED (continuing with next model)"
    FAILED+=("$model")
  fi
done

# ---- Summary ----
echo ""
echo "=========================================="
echo "   SWEEP COMPLETE"
echo "=========================================="
echo ""
echo "  Ran: $TOTAL model(s)"
echo "  Failed: ${#FAILED[@]}"

if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo ""
  echo "  Failed models:"
  for f in "${FAILED[@]}"; do
    echo "    ✗ $f"
  done
  exit 1
fi
