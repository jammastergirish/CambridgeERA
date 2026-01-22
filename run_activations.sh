#!/usr/bin/env bash
set -euo pipefail

OUTROOT="${OUTROOT:-outputs}"
FORGET="${FORGET_TEXT:-data/forget.txt}"
RETAIN="${RETAIN_TEXT:-data/retain.txt}"

if [[ ! -f "$FORGET" || ! -f "$RETAIN" ]]; then
  echo "Activation files missing; skipping."
  exit 0
fi

MODELS=(
  "EleutherAI/deep-ignorance-unfiltered"
  "EleutherAI/deep-ignorance-e2e-strong-filter"
  "EleutherAI/deep-ignorance-unfiltered-cb-lat"
)

DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-auto}"

uv run --script collect_activation_norms.py \
  --models "${MODELS[@]}" \
  --forget-text "$FORGET" \
  --retain-text "$RETAIN" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --outdir "${OUTROOT}/activation_norms"
