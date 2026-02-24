#!/usr/bin/env bash
# Fourth sweep: exploiting the NPO/SimNPO breakthrough at lr5e-05.
#
# Findings from sweeps 1-3:
#   - simnpo ep3 lr5e-05 beats the strong-filter on ALL WMDP metrics (Cat: 0.3189 vs 0.4006)
#   - lr=1e-04 collapses MMLU (<0.30); ceiling confirmed between 5e-05 and 1e-04
#   - CB/CB_LAT: broad layers preserve MMLU but barely unlearn; lr5e-05 never tested
#   - DPO at lr5e-05 never tested despite being same family as NPO/SimNPO
#   - retain_weight has only ever been 1.0; lowering it trades MMLU for more forgetting
#
# Goals (24 runs total):
#   1) Extend SimNPO/NPO epoch sweep at lr5e-05 (ep4, ep5)         [4 runs]
#   2) Probe intermediate LR 7e-05 for SimNPO and NPO              [6 runs]
#   3) DPO at lr5e-05 with both beta values (0.1 and 0.5)          [6 runs]
#   4) Retain-weight ablation (0.5, 0.1) for top methods            [4 runs]
#   5) CB/CB_LAT at lr5e-05 with broad layers (WMDP vs collapse)   [4 runs]

set -euo pipefail

cd "$(dirname "$0")/.."

export BASE="${BASE:-EleutherAI/deep-ignorance-unfiltered}"
export DEVICE="${DEVICE:-auto}"
export DTYPE="${DTYPE:-auto}"
export BATCH_SIZE=4
export EVAL_SPLIT=0.1
export MAX_LENGTH=512
export NO_SAVE=1

LAYER_ID_BROAD="5,10,15,20,25,30"

echo "=========================================================="
echo "Starting FOURTH hyperparameter sweep"
echo "Exploiting the NPO/SimNPO lr5e-05 breakthrough"
echo "Model: $BASE"
echo "=========================================================="

# ---------------------------------------------------------------
# 1. SimNPO: extend epoch sweep at lr5e-05 (ep3 is current best)
# ---------------------------------------------------------------
for ep in "4" "5"; do
    echo -e "\n>>> [simnpo] LR=5e-05, EPOCHS=${ep}, RETAIN_WEIGHT=1.0 <<<"
    EPOCHS=$ep LR=5e-05 BETA=0.1 RETAIN_WEIGHT=1.0 ./unlearn/run_unlearn.sh simnpo
done

# ---------------------------------------------------------------
# 2. NPO: extend epoch sweep at lr5e-05 (ep3 currently best Cat)
# ---------------------------------------------------------------
for ep in "4" "5"; do
    echo -e "\n>>> [npo] LR=5e-05, EPOCHS=${ep}, RETAIN_WEIGHT=1.0 <<<"
    EPOCHS=$ep LR=5e-05 BETA=0.1 RETAIN_WEIGHT=1.0 ./unlearn/run_unlearn.sh npo
done

# ---------------------------------------------------------------
# 3. Probe intermediate LR 7e-05 — between best (5e-05) and cliff (1e-04)
#    May unlock better WMDP/MMLU Pareto point for both methods
# ---------------------------------------------------------------
for ep in "1" "2" "3"; do
    echo -e "\n>>> [simnpo] LR=7e-05, EPOCHS=${ep} <<<"
    EPOCHS=$ep LR=7e-05 BETA=0.1 RETAIN_WEIGHT=1.0 ./unlearn/run_unlearn.sh simnpo

    echo -e "\n>>> [npo] LR=7e-05, EPOCHS=${ep} <<<"
    EPOCHS=$ep LR=7e-05 BETA=0.1 RETAIN_WEIGHT=1.0 ./unlearn/run_unlearn.sh npo
done

# ---------------------------------------------------------------
# 4. DPO at lr5e-05 — entire promising LR range never tested for DPO.
#    Also sweep beta=0.5 which improved DPO at lr1e-05.
# ---------------------------------------------------------------
for ep in "1" "2" "3"; do
    for beta in "0.1" "0.5"; do
        echo -e "\n>>> [dpo] LR=5e-05, EPOCHS=${ep}, BETA=${beta} <<<"
        EPOCHS=$ep LR=5e-05 BETA=$beta ./unlearn/run_unlearn.sh dpo
    done
done

# ---------------------------------------------------------------
# 5. Retain-weight ablation for top methods at lr5e-05
#    retain_weight=1.0 has been the only value tried; lower values
#    allow more aggressive forgetting at some MMLU cost
# ---------------------------------------------------------------
for rw in "0.5" "0.1"; do
    echo -e "\n>>> [simnpo] LR=5e-05, EPOCHS=3, RETAIN_WEIGHT=${rw} <<<"
    EPOCHS=3 LR=5e-05 BETA=0.1 RETAIN_WEIGHT=$rw ./unlearn/run_unlearn.sh simnpo

    echo -e "\n>>> [npo] LR=5e-05, EPOCHS=3, RETAIN_WEIGHT=${rw} <<<"
    EPOCHS=3 LR=5e-05 BETA=0.1 RETAIN_WEIGHT=$rw ./unlearn/run_unlearn.sh npo
done

# ---------------------------------------------------------------
# 6. CB / CB_LAT at lr5e-05 with broad layers
#    Previous: lr3e-05 broad layers barely unlearns (Cat ~0.52).
#    Hypothesis: lr5e-05 broad layers might move WMDP without collapse,
#    as the stable operating regime is wider for circuit-breaking.
# ---------------------------------------------------------------
for ep in "1" "3"; do
    echo -e "\n>>> [cb] LR=5e-05, EPOCHS=${ep}, LAYER_ID=${LAYER_ID_BROAD} <<<"
    EPOCHS=$ep LR=5e-05 LAYER_ID=$LAYER_ID_BROAD ALPHA=100.0 STEERING_COEFF=20.0 \
        ./unlearn/run_unlearn.sh cb

    echo -e "\n>>> [cb_lat] LR=5e-05, EPOCHS=${ep}, LAYER_ID=${LAYER_ID_BROAD} <<<"
    EPOCHS=$ep LR=5e-05 LAYER_ID=$LAYER_ID_BROAD ALPHA=100.0 STEERING_COEFF=20.0 \
        ./unlearn/run_unlearn.sh cb_lat
done

echo "=========================================================="
echo "Sweep 4 completed successfully!"
echo "=========================================================="
