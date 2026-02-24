#!/usr/bin/env bash
# parallel_sweep.sh — run any sweep script with parallel GPU group dispatch.
#
# Usage:
#   ./unlearn/parallel_sweep.sh ./unlearn/sweep_unlearn2.sh
#   GPUS_PER_JOB=4 ./unlearn/parallel_sweep.sh ./unlearn/sweep_unlearn3.sh
#
# How it works:
#   1. Detects GPU count; probes model VRAM to choose the minimum group size
#      that fits the model (rounded to a power of two that divides NUM_GPUS).
#   2. Temporarily replaces unlearn/run_unlearn.sh with a lightweight shim.
#      The shim:
#        a. Spins until a GPU group is free (using atomic mkdir as a lock).
#        b. Starts the real job in the background, holding the group lock.
#        c. Returns immediately so the sweep script queues the next config.
#   3. Runs the sweep script — different configs execute on different GPU groups.
#   4. Waits for all background jobs to complete, then restores run_unlearn.sh.
#
# Result: with 8× A40s and a model requiring 4 GPUs, 2 sweep configs run at
# once instead of serially — doubling throughput with no changes to any
# existing sweep script.
#
# Override group size (skips VRAM probe):
#   GPUS_PER_JOB=4 ./unlearn/parallel_sweep.sh ./unlearn/sweep_unlearn2.sh

set -euo pipefail

SWEEP_SCRIPT="${1:?Usage: $0 <sweep_script>}"

# Ensure we're in the project root (same convention as all other scripts).
cd "$(dirname "$0")/.."

BASE="${BASE:-EleutherAI/deep-ignorance-unfiltered}"
DTYPE="${DTYPE:-auto}"
REAL_RUN="$(pwd)/unlearn/run_unlearn.sh"

# ---------------------------------------------------------------------------
# 1. Detect GPU count
# ---------------------------------------------------------------------------
NUM_GPUS=0
if command -v nvidia-smi &>/dev/null; then
  NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 0)
fi

if [[ "$NUM_GPUS" -le 1 ]]; then
  echo "[parallel_sweep] Only ${NUM_GPUS} GPU(s) detected — running sweep directly."
  exec bash "$SWEEP_SCRIPT"
fi

# ---------------------------------------------------------------------------
# 2. Determine GPU group size (GPUs per job)
# ---------------------------------------------------------------------------
if [[ -n "${GPUS_PER_JOB:-}" ]]; then
  GROUP_SIZE="$GPUS_PER_JOB"
  echo "[parallel_sweep] GPUS_PER_JOB=${GROUP_SIZE} (manual override)"
else
  echo "[parallel_sweep] Probing VRAM requirements for ${BASE} (~30 s)..."

  # Load the model with device_map=auto and report how many GPUs it spread to.
  PROBE=$(python - "$BASE" "$DTYPE" <<'PYEOF' 2>/dev/null || echo "ERROR")
import sys, torch
model_id, dtype_str = sys.argv[1], sys.argv[2]
dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32, "auto": torch.bfloat16}
pt_dtype = dtype_map.get(dtype_str, torch.bfloat16)
from transformers import AutoModelForCausalLM
try:
    m = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=pt_dtype, low_cpu_mem_usage=True)
    gpus_used = {p.device.index for p in m.parameters() if p.device.type == "cuda"}
    print(len(gpus_used))
    del m
    torch.cuda.empty_cache()
except Exception as e:
    print("ERROR", file=sys.stderr)
    print("ERROR")
PYEOF

  if [[ "$PROBE" == ERROR* ]] || [[ -z "$PROBE" ]]; then
    echo "[parallel_sweep] Probe failed; defaulting to all ${NUM_GPUS} GPUs per job."
    GROUP_SIZE="$NUM_GPUS"
  else
    MIN_GPUS="$PROBE"
    GROUP_SIZE="$NUM_GPUS"  # safe fallback
    for gs in 1 2 4 8 16; do
      if [[ "$gs" -ge "$MIN_GPUS" ]] && (( NUM_GPUS % gs == 0 )); then
        GROUP_SIZE="$gs"
        break
      fi
    done
    echo "[parallel_sweep] Model needs ${MIN_GPUS} GPU(s); group size → ${GROUP_SIZE}"
  fi
fi

NUM_GROUPS=$(( NUM_GPUS / GROUP_SIZE ))
echo "[parallel_sweep] ${NUM_GROUPS} parallel job(s) × ${GROUP_SIZE} GPU(s) each"

if [[ "$NUM_GROUPS" -le 1 ]]; then
  echo "[parallel_sweep] Only one group fits — running sweep directly."
  exec bash "$SWEEP_SCRIPT"
fi

# ---------------------------------------------------------------------------
# 3. Create shared state directory, jobs tracking file, and shim script
# ---------------------------------------------------------------------------
STATE_DIR=$(mktemp -d /tmp/parallel_sweep_XXXXXX)
JOBS_FILE="$STATE_DIR/jobs.pids"
touch "$JOBS_FILE"

# Each GPU group gets a lock directory:  $STATE_DIR/group_N  (exists = busy)
# mkdir is atomic on POSIX — used as a lightweight group lock.
# When a background job finishes, it rmdir's the lock (releasing the group).

SHIM="$STATE_DIR/shim.sh"
# Bake the runtime values into the shim at creation time.
cat > "$SHIM" << SHIM_EOF
#!/usr/bin/env bash
# Auto-generated shim — do not edit directly.
# Waits for a free GPU group then starts the real job in the background.
set -euo pipefail

_NUM_GROUPS=$NUM_GROUPS
_GROUP_SIZE=$GROUP_SIZE
_STATE_DIR="$STATE_DIR"
_JOBS_FILE="$JOBS_FILE"
_REAL="$REAL_RUN.bak"   # real run_unlearn.sh was backed up here

while true; do
  for g in \$(seq 0 \$(( _NUM_GROUPS - 1 ))); do
    lock_dir="\$_STATE_DIR/group_\${g}"

    # atomic: if mkdir succeeds we own this group slot
    if mkdir "\$lock_dir" 2>/dev/null; then
      start=\$(( g * _GROUP_SIZE ))
      end=\$(( start + _GROUP_SIZE - 1 ))
      GPU_LIST=\$(seq -s, "\$start" "\$end")

      echo "[parallel_sweep] Group \${g} (CUDA_VISIBLE_DEVICES=\${GPU_LIST}): \$*"

      # Start the real job in the background.
      # The subshell holds the lock (lock_dir) and releases it via trap on exit.
      (
        trap "rmdir '\$lock_dir' 2>/dev/null || true" EXIT
        CUDA_VISIBLE_DEVICES="\$GPU_LIST" DEVICE=auto bash "\$_REAL" "\$@"
      ) &

      JOB_PID=\$!
      echo "\$JOB_PID" >> "\$_JOBS_FILE"
      exit 0   # return immediately — sweep script queues next config
    fi
  done

  sleep 2  # all groups busy; poll again
done
SHIM_EOF
chmod +x "$SHIM"

# ---------------------------------------------------------------------------
# 4. Swap shim into place; restore original on any exit
# ---------------------------------------------------------------------------
BACKUP="${REAL_RUN}.bak"
cp "$REAL_RUN" "$BACKUP"
cp "$SHIM" "$REAL_RUN"

cleanup() {
  # Always restore the original run_unlearn.sh, even if we were interrupted.
  if [[ -f "$BACKUP" ]]; then
    mv -f "$BACKUP" "$REAL_RUN"
  fi
  rm -rf "$STATE_DIR"
  echo "[parallel_sweep] run_unlearn.sh restored."
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# 5. Run the sweep script
# Each call to run_unlearn.sh hits the shim, which returns immediately after
# queuing the job — so the sweep loops through all configs quickly.
# ---------------------------------------------------------------------------
echo "[parallel_sweep] ======================================================="
echo "[parallel_sweep] Starting: $SWEEP_SCRIPT"
echo "[parallel_sweep] ======================================================="
bash "$SWEEP_SCRIPT"
echo "[parallel_sweep] All configs queued. Waiting for background jobs..."

# ---------------------------------------------------------------------------
# 6. Wait for every background job and collect exit codes
# ---------------------------------------------------------------------------
ALL_OK=1
while IFS= read -r pid; do
  [[ -z "$pid" ]] && continue
  if kill -0 "$pid" 2>/dev/null; then
    if ! wait "$pid"; then
      echo "[parallel_sweep] WARNING: job PID ${pid} failed."
      ALL_OK=0
    fi
  fi
done < "$JOBS_FILE"

if [[ "$ALL_OK" -eq 1 ]]; then
  echo "[parallel_sweep] All jobs completed successfully ✓"
else
  echo "[parallel_sweep] One or more jobs failed — check output above."
  exit 1
fi
