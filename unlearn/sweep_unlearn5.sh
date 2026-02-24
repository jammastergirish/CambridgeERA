#!/usr/bin/env bash
# Fifth sweep: batch size ablation for NPO and SimNPO.
#
# Best configs from sweeps 3-4:
#   simnpo ep3 lr5e-05 bs4 b0.1 rw1.0  → WMDP Cat 0.3189, MMLU 0.4113 (best overall)
#   npo    ep3 lr5e-05 bs4 b0.1 rw1.0  → WMDP Cat 0.3464, MMLU 0.4054
#
# Goal: test whether larger batches improve gradient quality and push WMDP lower.
# Larger batches → less noisy gradients → possibly more precise forgetting.
# Note: LR is NOT scaled with batch size here; the existing lr5e-05 is already
# near the collapse ceiling so scaling up would likely cause collapse.
#
# 8 runs total (4 per method × 2 batch sizes).

set -euo pipefail

cd "$(dirname "$0")/.."

export BASE="${BASE:-EleutherAI/deep-ignorance-unfiltered}"
export DEVICE="${DEVICE:-auto}"
export DTYPE="${DTYPE:-auto}"
export EVAL_SPLIT=0.1
export MAX_LENGTH=512
export NO_SAVE=1

echo "=========================================================="
echo "Starting FIFTH hyperparameter sweep"
echo "Batch size ablation for NPO and SimNPO at optimal LR"
echo "Model: $BASE"
echo "=========================================================="

# ---------------------------------------------------------------
# SimNPO — optimal: ep3 lr5e-05 b0.1 rw1.0 (baseline bs=4)
# ---------------------------------------------------------------
for bs in "8" "16"; do
    for ep in "2" "3"; do
        echo -e "\n>>> [simnpo] EPOCHS=${ep}, LR=5e-05, BS=${bs}, BETA=0.1, RETAIN_WEIGHT=1.0 <<<"
        EPOCHS=$ep LR=5e-05 BATCH_SIZE=$bs BETA=0.1 RETAIN_WEIGHT=1.0 \
            ./unlearn/run_unlearn.sh simnpo
    done
done

# ---------------------------------------------------------------
# NPO — optimal: ep3 lr5e-05 b0.1 rw1.0 (baseline bs=4)
# ---------------------------------------------------------------
for bs in "8" "16"; do
    for ep in "1" "3"; do
        echo -e "\n>>> [npo] EPOCHS=${ep}, LR=5e-05, BS=${bs}, BETA=0.1, RETAIN_WEIGHT=1.0 <<<"
        EPOCHS=$ep LR=5e-05 BATCH_SIZE=$bs BETA=0.1 RETAIN_WEIGHT=1.0 \
            ./unlearn/run_unlearn.sh npo
    done
done

echo "=========================================================="
echo "Sweep 5 completed successfully!"
echo "=========================================================="
