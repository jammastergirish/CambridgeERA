#!/usr/bin/env bash
set -euo pipefail

OUTROOT="${OUTROOT:-outputs}"

BASE="EleutherAI/deep-ignorance-unfiltered"
FILTERED="EleutherAI/deep-ignorance-e2e-strong-filter"
UNLEARNED="EleutherAI/deep-ignorance-unfiltered-cb-lat"

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"

# 1. Base vs Filtered (ceiling — maximum structural difference)
#    Normalize by Filtered (the target / gold standard)
echo "=== Comparison 1/3: Base vs Filtered ==="
uv run --script collect_weight_comparison.py \
  --model-a "$BASE" \
  --model-b "$FILTERED" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --outdir "${OUTROOT}/weight_comparison"

# 2. Unlearned vs Base (how much did unlearning change?)
#    Normalize by Base (the starting point)
echo "=== Comparison 2/3: Unlearned vs Base ==="
uv run --script collect_weight_comparison.py \
  --model-a "$UNLEARNED" \
  --model-b "$BASE" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --outdir "${OUTROOT}/weight_comparison"

# 3. Unlearned vs Filtered (how close to gold standard?)
#    Normalize by Filtered (the target)
echo "=== Comparison 3/3: Unlearned vs Filtered ==="
uv run --script collect_weight_comparison.py \
  --model-a "$UNLEARNED" \
  --model-b "$FILTERED" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --outdir "${OUTROOT}/weight_comparison"

echo "=== All comparisons complete ==="
