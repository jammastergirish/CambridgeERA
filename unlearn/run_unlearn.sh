#!/usr/bin/env bash
# Run the unlearning pipeline for a given method.
#
# Usage:
#   ./run_unlearn.sh ga_simple     # Pure Gradient Ascent (forget only)
#   ./run_unlearn.sh ga            # Gradient Ascent + retain
#   ./run_unlearn.sh grad_diff     # Gradient Difference
#   ./run_unlearn.sh dpo           # Direct Preference Optimization
#   ./run_unlearn.sh npo           # Negative Preference Optimization
#   ./run_unlearn.sh simnpo        # Simple NPO (reference-free)
#   ./run_unlearn.sh rmu           # Representation Misdirection
#   ./run_unlearn.sh cb            # Circuit Breakers
#   ./run_unlearn.sh lat           # Latent Adversarial Training
#   ./run_unlearn.sh cb_lat        # CB + LAT combined
#   ./run_unlearn.sh wt_dist       # Weight Distortion (Gaussian noise + retain FT)
#   ./run_unlearn.sh wt_dist_reg   # Weight Distance Regularization
#
# Environment overrides:
#   BASE, DEVICE, DTYPE, OUTROOT, LR, EPOCHS, BATCH_SIZE, MAX_LENGTH, BETA,
#   ALPHA, STEERING_COEFF, LAYER_ID, FORGET_WEIGHT, LAT_EPS, LAT_STEPS,
#   WT_NOISE_STD, WT_REG_LAMBDA, EVAL_SPLIT
set -euo pipefail

METHOD="${1:?Usage: $0 <ga_simple|ga|grad_diff|dpo|npo|simnpo|rmu|cb|lat|cb_lat|wt_dist|wt_dist_reg>}"
BASE="${BASE:-EleutherAI/deep-ignorance-unfiltered}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-auto}"
OUTROOT="${OUTROOT:-unlearned_models}"

# Build comparison folder name
SAFE_BASE="${BASE//\//_}"
COMP="${SAFE_BASE}__${METHOD}"

echo "=== Unlearning: method=${METHOD}  model=${BASE} ==="
echo "=== Output: ${OUTROOT}/${COMP} ==="

uv run --script unlearn/unlearn.py \
  --model "$BASE" \
  --method "$METHOD" \
  --forget-data data/forget.txt \
  --retain-data data/retain.txt \
  --outdir "${OUTROOT}/${COMP}" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  ${LR:+--lr "$LR"} \
  ${EPOCHS:+--epochs "$EPOCHS"} \
  ${BATCH_SIZE:+--batch-size "$BATCH_SIZE"} \
  ${MAX_LENGTH:+--max-length "$MAX_LENGTH"} \
  ${BETA:+--beta "$BETA"} \
  ${ALPHA:+--alpha "$ALPHA"} \
  ${STEERING_COEFF:+--steering-coeff "$STEERING_COEFF"} \
  ${LAYER_ID:+--layer-id "$LAYER_ID"} \
  ${FORGET_WEIGHT:+--forget-weight "$FORGET_WEIGHT"} \
  ${LAT_EPS:+--lat-eps "$LAT_EPS"} \
  ${LAT_STEPS:+--lat-steps "$LAT_STEPS"} \
  ${WT_NOISE_STD:+--wt-noise-std "$WT_NOISE_STD"} \
  ${WT_REG_LAMBDA:+--wt-reg-lambda "$WT_REG_LAMBDA"} \
  ${EVAL_SPLIT:+--eval-split "$EVAL_SPLIT"} \
  --seed 42

