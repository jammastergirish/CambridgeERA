#!/usr/bin/env bash
set -euo pipefail

COMP1="EleutherAI_deep-ignorance-unfiltered__to__EleutherAI_deep-ignorance-e2e-strong-filter"
COMP2="EleutherAI_deep-ignorance-unfiltered__to__EleutherAI_deep-ignorance-unfiltered-cb-lat"

uv run plot_param_stats.py \
  --per-layer-csv "outputs/${COMP1}/param_stats/per_layer.csv" \
  --outdir "plots/${COMP1}/param_plots" \
  --title "EleutherAI/deep-ignorance-unfiltered → EleutherAI/deep-ignorance-e2e-strong-filter"
  
uv run plot_param_stats.py \
  --per-layer-csv "outputs/${COMP2}/param_stats/per_layer.csv" \
  --outdir "plots/${COMP2}/param_plots" \
  --title "EleutherAI/deep-ignorance-unfiltered → EleutherAI/deep-ignorance-unfiltered-cb-lat"