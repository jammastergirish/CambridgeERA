#!/usr/bin/env bash
set -euo pipefail

OUTROOT="${OUTROOT:-outputs}"
FORGET="${FORGET_TEXT:-data/forget.txt}"
RETAIN="${RETAIN_TEXT:-data/retain.txt}"

if [[ ! -f "$FORGET" || ! -f "$RETAIN" ]]; then
  echo "Activation files missing; skipping."
  exit 0
fi

DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-auto}"

# Baseline model
MODEL_A="EleutherAI/deep-ignorance-unfiltered"

# Comparison 1: Base → Filtered
MODEL_B="EleutherAI/deep-ignorance-e2e-strong-filter"
COMP1="EleutherAI_deep-ignorance-unfiltered__to__EleutherAI_deep-ignorance-e2e-strong-filter"
echo ""
echo "===== Comparing: $MODEL_A -> $MODEL_B ====="
uv run --script collect_activation_norms.py \
  --model-a "$MODEL_A" \
  --model-b "$MODEL_B" \
  --forget-text "$FORGET" \
  --retain-text "$RETAIN" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --outdir "${OUTROOT}/${COMP1}/activation_stats"

# Comparison 2: Base → Unlearned
MODEL_B="EleutherAI/deep-ignorance-unfiltered-cb-lat"
COMP2="EleutherAI_deep-ignorance-unfiltered__to__EleutherAI_deep-ignorance-unfiltered-cb-lat"
echo ""
echo "===== Comparing: $MODEL_A -> $MODEL_B ====="
uv run --script collect_activation_norms.py \
  --model-a "$MODEL_A" \
  --model-b "$MODEL_B" \
  --forget-text "$FORGET" \
  --retain-text "$RETAIN" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --outdir "${OUTROOT}/${COMP2}/activation_stats"
