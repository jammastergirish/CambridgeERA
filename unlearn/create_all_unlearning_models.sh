#!/usr/bin/env bash
set -euo pipefail

for method in ga_simple ga grad_diff dpo npo simnpo rmu cb lat cb_lat wt_dist wt_dist_reg; do
  ./unlearn/run_unlearn.sh "$method"
done
echo "All methods complete."