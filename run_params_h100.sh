#!/usr/bin/env bash
set -euo pipefail

OUTROOT="${OUTROOT:-outputs}"

BASE="EleutherAI/deep-ignorance-unfiltered"
FILTERED="EleutherAI/deep-ignorance-e2e-strong-filter"
UNLEARNED="EleutherAI/deep-ignorance-unfiltered-cb-lat"

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"

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
