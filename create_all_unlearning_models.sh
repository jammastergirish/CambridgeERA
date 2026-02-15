#!/usr/bin/env bash
set -euo pipefail

for method in ga_simple ga grad_diff dpo npo simnpo rmu cb lat cb_lat; do
  ./run_unlearn.sh "$method"
done
echo "All methods complete."