#!/usr/bin/env bash
set -euo pipefail

OUTROOT="${OUTROOT:-outputs}"

BASE="EleutherAI/deep-ignorance-unfiltered"
FILTERED="EleutherAI/deep-ignorance-e2e-strong-filter"
UNLEARNED="EleutherAI/deep-ignorance-unfiltered-cb-lat"

# Safer on Mac: CPU + fp16 weights to reduce RAM; computations cast to fp32 internally.
DEVICE="${DEVICE:-cpu}"
DTYPE="${DTYPE:-fp16}"

uv run --script collect_param_stats.py \
  --model-a "$BASE" \
  --model-b "$FILTERED" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --outdir "${OUTROOT}/param_stats"

uv run --script collect_param_stats.py \
  --model-a "$BASE" \
  --model-b "$UNLEARNED" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --outdir "${OUTROOT}/param_stats"
