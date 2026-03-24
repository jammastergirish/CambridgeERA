#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_A="${MODEL_A:-EleutherAI/deep-ignorance-unfiltered}"
CROSS_METHOD_OUTDIR="${OUTROOT:-outputs}/cross_method_basin"

uv run experiment/cross_method_basin_comparison.py \
  --model-a "$MODEL_A" \
  --outdir "$CROSS_METHOD_OUTDIR"
