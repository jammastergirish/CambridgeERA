#!/usr/bin/env bash
# Example: Run NPO and SimNPO unlearning on the deep-ignorance-unfiltered model.
#
# Usage:
#   ./run_unlearn_local.sh          # Run both NPO and SimNPO
#   DEVICE=mps ./run_unlearn_local.sh   # Use MPS on Mac
set -euo pipefail

export BASE="EleutherAI/deep-ignorance-unfiltered"
export DEVICE="${DEVICE:-auto}"
export DTYPE="${DTYPE:-auto}"
export OUTROOT="${OUTROOT:-outputs}"

echo "========================================"
echo " Unlearning: NPO"
echo "========================================"
./run_unlearn.sh npo

echo ""
echo "========================================"
echo " Unlearning: SimNPO"
echo "========================================"
./run_unlearn.sh simnpo
