#!/usr/bin/env bash
set -euo pipefail

OUTROOT="${OUTROOT:-outputs}"

BASE="EleutherAI/deep-ignorance-unfiltered"
FILTERED="EleutherAI/deep-ignorance-e2e-strong-filter"
UNLEARNED="EleutherAI/deep-ignorance-unfiltered-cb-lat"

# Safer on Mac: CPU + fp16 weights to reduce RAM; computations cast to fp32 internally.
DEVICE="${DEVICE:-cpu}"
DTYPE="${DTYPE:-fp16}"

# Comparison 1: Base → Filtered
COMP1="EleutherAI_deep-ignorance-unfiltered__to__EleutherAI_deep-ignorance-e2e-strong-filter"
uv run --script collect_param_stats.py \
  --model-a "$BASE" \
  --model-b "$FILTERED" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --outdir "${OUTROOT}/${COMP1}/param_stats"

# Comparison 2: Base → Unlearned
COMP2="EleutherAI_deep-ignorance-unfiltered__to__EleutherAI_deep-ignorance-unfiltered-cb-lat"
uv run --script collect_param_stats.py \
  --model-a "$BASE" \
  --model-b "$UNLEARNED" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --outdir "${OUTROOT}/${COMP2}/param_stats"
