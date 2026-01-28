uv run plot_param_stats.py \
  --per-layer-csv outputs/param_stats/EleutherAI_deep-ignorance-unfiltered__to__EleutherAI_deep-ignorance-e2e-strong-filter/per_layer.csv \
  --outdir plots/EleutherAI_deep-ignorance-unfiltered__to__EleutherAI_deep-ignorance-e2e-strong-filter \
  --title "EleutherAI/deep-ignorance-unfiltered → EleutherAI/deep-ignorance-e2e-strong-filter"
  
uv run plot_param_stats.py \
  --per-layer-csv outputs/param_stats/EleutherAI_deep-ignorance-unfiltered__to__EleutherAI_deep-ignorance-unfiltered-cb-lat/per_layer.csv \
  --outdir plots/EleutherAI_deep-ignorance-unfiltered__to__EleutherAI_deep-ignorance-unfiltered-cb-lat \
  --title "EleutherAI/deep-ignorance-unfiltered → EleutherAI/deep-ignorance-unfiltered-cb-lat"