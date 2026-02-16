#!/usr/bin/env bash
set -euo pipefail

for method in ga_simple ga grad_diff dpo npo simnpo rmu cb lat cb_lat; do
  ./unlearn/run_unlearn.sh "$method"
done
echo "All methods complete."