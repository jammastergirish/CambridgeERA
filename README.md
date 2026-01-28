# Model Diffs Analysis Tools

This directory contains scripts for analyzing the differences between PyTorch models (e.g., a baseline vs. a fine-tuned/unlearned version). It focuses on parameter statistics (Frobenius norms, Stable Rank) and activation norms.

## Quick Start

Use the provided bash scripts to run the analysis. These scripts handle arguments and environment setups for you.

### 1. Data Generation (Required for Activations)
Before running activation analysis, you must generate the "Forget" and "Retain" datasets.

```bash
uv run create_datasets.py
# Creates data/forget.txt and data/retain.txt
```

### 2. Parameter Statistics (Local / Mac)
Use `run_params_local.sh` to compare models on your local machine (e.g. Mac). It defaults to CPU/MPS and safe dtypes to avoid OOM.

```bash
# Edit variables in the script first if needed (BASE, FILTERED, UNLEARNED)
./run_params_local.sh
```

### 3. Parameter Statistics (H100 / Cluster)
Use `run_params_h100.sh` for high-performance runs on a GPU cluster.

```bash
# Edit variables in the script first if needed
./run_params_h100.sh
```

### 4. Activation Norms
Use `run_activations.sh` to compute activation norms.

```bash
# Requires data/forget.txt and data/retain.txt (from step 1)
./run_activations.sh
```

### 5. Visualization
Generate plots from the computed statistics.

```bash
# Plot parameter statistics (Frobenius norms, Stable Rank)
./plot_param_stats.sh

# Plot activation norms
./plot_activation_norms.sh
```

### 6. Full Pipeline
Run the entire workflow end-to-end (Local/Mac optimized).

```bash
./pipeline.sh
```

---

## Script Details

### `collect_param_stats.py`
**Purpose**: Calculates how much the weights have changed and how "complex" those changes are.

**Key Concepts:**
1.  **Linear Only**: It scans for `nn.Linear` modules (ignoring LayerNorms, Biases, and Embeddings).
2.  **Spectral Norm (via Power Iteration)**: Computing the true singular values (SVD) of large matrices (e.g., 4096x11008) is slow. This script uses **Power Iteration** to approximate the largest singular value ($\|A\|_2$) efficiently on the GPU.
3.  **Stable Rank**: defined as $r_{stable} = \frac{\|A\|_F^2}{\|A\|_2^2}$.
    *   **Low Stable Rank (near 1)**: The change is "spiky" or low-rank; it affects only a few specific directions in the feature space.
    *   **High Stable Rank**: The change is "blurry" or isotropic; it affects many directions intimately.
4.  **Memory Efficient (Streaming)**: This script uses a **SmartLoader** to read parameters one-by-one from disk (supporting `safetensors` and `pytorch_model.bin`). It does **not** load the full model into RAM, making it safe to run 70B+ param model analysis locally.

**Outputs:**
*   `per_matrix.csv`: Granular stats for every single weight matrix (e.g. `model.layers.5.mlp.gate_proj.weight`).
*   `per_layer.csv`: Aggregated stats.
    *   `dW_fro_layer`: Root-sum-square of all Frobenius norms in that layer (like a "Layer Norm" of the parameter stats).
    *   `mean_dW_stable_rank`: Average complexity of changes in that layer.

### `collect_activation_norms.py`
**Purpose**: Checks if the model is "activitating" more or less strongly on specific data (Forget set vs Retain set).

**Process:**
1.  **Tokenization**: Reads lines from text files and tokenizes them.
2.  **Forward Pass**: Runs the model in `inference_mode` and captures the **hidden states** at every layer.
3.  **L2 Norm Calculation**:
    *   For each token, it computes the L2 norm of the hidden vector: $\|h\|_2 = \sqrt{\sum h_i^2}$.
    *   It ignores padding tokens (using the attention mask).
4.  **Averaging**: accurately computes the mean norm per layer across all valid tokens in the dataset.

**Interpretation:**
*   If `mean_norm` drops significantly on the "Forget" set but stays same on "Retain", the unlearning was effective (the model is less confident/active on the target).
*   If `mean_norm` explodes, the model might be broken (lobotomized).  