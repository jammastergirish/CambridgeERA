clear
rm -rf outputs plots
./run_params_local.sh
./plot_param_stats.sh
uv run create_datasets.py
./run_activations.sh
./plot_activation_norms.sh
