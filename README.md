# Model Diffs Analysis Tools

This directory contains scripts for analyzing the differences between PyTorch models (e.g., a baseline vs. a fine-tuned/unlearned version). It focuses on parameter statistics (Frobenius norms, Stable Rank) and activation norms.

## Quick Start

Run the full analysis pipeline (Data Generation → Parameter Stats → Activations → Plotting):

```bash
./pipeline.sh
```
*Note: This script cleans the `outputs/` and `plots/` directories and runs all steps end-to-end.*

---

## Pipeline Breakdown

Here is a detailed explanation of every file executed by `pipeline.sh`:

### 1. `run_params_local.sh`
**Action**: Runs `collect_param_stats.py` twice to compare the baseline model against two variations.
*   **Models Compared**:
    1.  `deep-ignorance-unfiltered` vs `deep-ignorance-e2e-strong-filter`
    2.  `deep-ignorance-unfiltered` vs `deep-ignorance-unfiltered-cb-lat` (Unlearned)
*   **Output**: `outputs/<comparison>/param_stats/per_matrix.csv` and `per_layer.csv`

### 2. `plot_param_stats.sh`
**Action**: Calls `plot_param_stats.py` to visualize the CSVs from step 1.
*   **Plots Generated**:
    *   `layer_locality_*.png`: Magnitude of changes per layer (Frobenius norm).
    *   `stable_rank_*.png`: Complexity of changes per layer (Stable Rank).
*   **Output Directory**: `plots/<comparison>/param_plots/`

### 3. `create_datasets.py`
**Action**: logic to download and filter raw datasets from HuggingFace (`cais/wmdp`, `wikitext`).
*   **Forget Set**: 500 questions from WMDP-Bio (Hazardous Bio knowledge).
*   **Retain Set**: 500 generic samples from Wikitext-2 (General capabilities).
*   **Output**: `data/forget.txt`, `data/retain.txt`.

### 4. `run_activations.sh`
**Action**: Runs `collect_activation_norms.py` for each model pair comparison.
*   **Comparisons**:
    1.  `deep-ignorance-unfiltered` → `deep-ignorance-e2e-strong-filter`
    2.  `deep-ignorance-unfiltered` → `deep-ignorance-unfiltered-cb-lat`
*   **Metrics Computed**:
    *   **Absolute norms**: L2 norm of hidden states for each model.
    *   **Activation diffs**: L1 and L2 norms of the hidden state difference ($\Delta h = h_{after} - h_{before}$).
*   **Output**: `outputs/<comparison>/activation_stats/activation_stats.csv`

### 5. `plot_activation_norms.sh`
**Action**: Calls `plot_activation_norms.py` to visualize activation statistics.
*   **Plots Generated**:
    *   `activation_norms_*.png`: Compares model A vs model B absolute activation magnitude.
    *   `activation_diffs_*.png`: Shows L1 and L2 norms of the activation difference.
*   **Output**: `plots/<comparison>/activation_plots/`

---

## Script Details & Technical Reference

### 1. `collect_param_stats.py`
**Purpose**: Calculates parameter-level statistics to quantify model changes ($\Delta W = W_{fine-tuned} - W_{base}$).

#### Key Concepts
*   **Frobenius Norm ($\|A\|_F$)**:
    *   The Euclidean norm of the flattened matrix: $\sqrt{\sum_{i,j} A_{ij}^2}$.
    *   Measures the *magnitude* of the change.
*   **Spectral Norm ($\|A\|_2$)**:
    *   The largest singular value ($\sigma_{max}$) of the matrix.
    *   Measures the maximum "stretch" the matrix can apply to a vector.
    *   *Implementation Note*: Computing full SVD for large matrices (e.g., 4096x11008) is slow. This script uses **Power Iteration** to efficiently approximate the largest singular value on the GPU.
    *   *Determinism*: Power Iteration starts with a random vector. The script uses a fixed random seed (default `--seed 42`) to ensuring results are bit-for-bit identical across runs.
*   **Stable Rank ($r_{stable}$)**:
    *   **Formula**: $r_{stable} = \frac{\|A\|_F^2}{\|A\|_2^2}$.
    *   **Interpretation**: A proxy for the "effective rank" of the matrix.
        *   **Low Stable Rank ($\approx 1$)**: The change is "spiky" or "rank-1". All the energy is concentrated in one specific direction (singular vector). This often indicates a precise, surgical edit.
        *   **High Stable Rank**: The change is "blurry" or isotropic. The energy is spread out across many dimensions. This often indicates noise or a general "drift" in the weights.

#### Outputs
**`outputs/<comparison>/param_stats/per_matrix.csv`**
Granular stats for every single weight matrix scaned.
| Column | Description |
| :--- | :--- |
| `name` | Parameter name (e.g., `layers.10.mlp.gate_proj.weight`) |
| `layer` | The integer layer index extracted from the name |
| `group` | Coarse grouping (`attn` for attention, `mlp` for feed-forward) |
| `shape0` | Matrix dimension 0 (rows/output features) |
| `shape1` | Matrix dimension 1 (cols/input features) |
| `dW_fro` | Frobenius norm of the difference: $\|\Delta W\|_F$ |
| `dW_stable_rank` | Stable rank of the difference: $r_{stable}(\Delta W)$ |
| `W_stable_rank` | Stable rank of the original base weights: $r_{stable}(W)$ |

**`outputs/<comparison>/param_stats/per_layer.csv`**
Aggregated statistics per layer.
| Column | Description |
| :--- | :--- |
| `layer` | The integer layer index |
| `group` | Coarse grouping (`attn` vs `mlp`) |
| `dW_fro_layer` | Root-sum-square of Frobenius norms in that layer ($\sqrt{\sum \|\Delta W_i\|_F^2}$). Like a "Layer Norm" for parameter changes. |
| `mean_dW_stable_rank` | Average stable rank of changes in that layer. |
| `count_mats` | Number of parameter matrices aggregated in this group/layer. |

---

### 2. `collect_activation_norms.py`
**Purpose**: Measures both absolute activation magnitudes and the difference in activations between model pairs on specific datasets.

#### Key Concepts
*   **Absolute Activation Norm ($\|h\|_2$)**:
    *   The L2 norm of the hidden state vector at each position.
    *   Measures the "signal strength" or "confidence" of the model's internal representation.
*   **Activation Difference ($\Delta h = h_{after} - h_{before}$)**:
    *   The difference in hidden states between two models on the **same input**.
    *   Measures how much the model's internal representations *changed* due to fine-tuning/unlearning.
*   **L1 vs L2 Norms of $\Delta h$**:
    *   **L1 ($\|\Delta h\|_1$)**: Sum of absolute differences. Treats all dimensions equally.
    *   **L2 ($\|\Delta h\|_2$)**: Euclidean distance. Penalizes large changes in individual dimensions.
    *   If L1 >> L2 (relatively): changes are spread across many dimensions.
    *   If L2 is large relative to L1: changes are concentrated in a few dimensions.

#### Memory-Efficient Implementation
The script uses **disk caching** to avoid loading two models simultaneously:
1.  Run **model A** on all texts, cache hidden states to temp directory.
2.  Run **model B**, load cached hidden states batch-by-batch, compute diffs.
3.  Cleanup temp files.

#### Data Sources
*   **Forget Set (`data/forget.txt`)**: Generated (via `create_datasets.py`) from **WMDP-Bio**. Hazardous biological knowledge to "forget".
*   **Retain Set (`data/retain.txt`)**: Generated from **Wikitext-2**. General English to preserve capabilities.

#### Outputs
**`outputs/<comparison>/activation_stats/activation_stats.csv`**
| Column | Description |
| :--- | :--- |
| `layer` | Layer index (0 to N) |
| `split` | Dataset split (`forget` or `retain`) |
| `model_a_norm_L1` | Mean L1 norm of hidden states for **model A** (baseline): $\mathbb{E}[\|h\|_1]$ |
| `model_a_norm_L2` | Mean L2 norm of hidden states for **model A** (baseline): $\mathbb{E}[\|h\|_2]$ |
| `model_b_norm_L1` | Mean L1 norm of hidden states for **model B** (target) |
| `model_b_norm_L2` | Mean L2 norm of hidden states for **model B** (target) |
| `mean_dh_L1` | Mean L1 norm of the activation difference: $\mathbb{E}[\|\Delta h\|_1]$ |
| `mean_dh_L2` | Mean L2 norm of the activation difference: $\mathbb{E}[\|\Delta h\|_2]$ |

#### Interpretation
*   **L1 vs L2 norms**:
    *   **L1** ($\|h\|_1 = \sum |h_i|$): Total activation mass across all dimensions. Treats every dimension equally.
    *   **L2** ($\|h\|_2 = \sqrt{\sum h_i^2}$): Geometric magnitude. Dominated by the largest activations.
    *   If L1 changes but L2 doesn't: the change is spread across many small dimensions.
    *   If L2 changes but L1 doesn't: the change is concentrated in a few dominant features.
*   **Effective Unlearning**:
    *   On **`forget`** split: Large `mean_dh_L1`/`mean_dh_L2` → representations changed significantly (good!).
    *   On **`retain`** split: Small `mean_dh_L1`/`mean_dh_L2` → representations stayed similar (good!).
*   **Absolute norm comparison**: If `model_b_norm_*` drops on forget but stays stable on retain, the model is selectively suppressing hazardous knowledge.

---

### 3. Plots
The scripts in `plots/` visualize the CSVs generated above:

*   **Layer Locality (`layer_locality_*.png`)**
    *   **Source Data**: `per_layer.csv`
    *   **Column Plotted**: `dW_fro_layer` (filtered by group `attn` or `mlp`)
    *   **Interpretation**:
        *   *X-axis*: Layer Index.
        *   *Y-axis*: Aggregated Frobenius Norm of parameters in that layer.
        *   *Meaning*: Shows **where** the model changed. A spike at layer 5 means the weights in layer 5 were modified significantly more than others.

*   **Edit Dimensionality (`stable_rank_*.png`)**
    *   **Source Data**: `per_layer.csv`
    *   **Column Plotted**: `mean_dW_stable_rank` (filtered by group `attn` or `mlp`)
    *   **Interpretation**:
        *   *X-axis*: Layer Index.
        *   *Y-axis*: Average Stable Rank of the difference matrices ($\Delta W$).
        *   *Meaning*: Shows **complexity** of the change.
            *   **~1.0**: Surgical, low-rank update (affects specific features).
            *   **High**: Broad, isotropic noise (affects all features).

*   **Activation Magnitude (`activation_norms_*.png`)**
    *   **Source Data**: `activation_stats.csv`
    *   **Columns Plotted**: L1 (left) and L2 (right) side-by-side, model A vs model B per panel
    *   **Interpretation**:
        *   *X-axis*: Layer Index.
        *   *Y-axis*: Average norm of hidden states.
        *   *Meaning*: Compares signal strength between models. Drop on Forget but stable on Retain = successful unlearning.

*   **Activation Diffs (`activation_diffs_*.png`)**
    *   **Source Data**: `activation_stats.csv`
    *   **Columns Plotted**: `mean_dh_L1` (left), `mean_dh_L2` (right) side-by-side
    *   **Interpretation**:
        *   *X-axis*: Layer Index.
        *   *Y-axis*: Mean norm of activation difference ($\Delta h$).
        *   *Meaning*: Shows how much internal representations changed. High on Forget + Low on Retain = targeted unlearning.  

---

## 6. Unlearning Pipeline

Run unlearning experiments with **10 different methods** via a single script. Includes automatic train/eval splitting and tqdm progress bars.

### Available Methods

#### Loss-based methods

| Method | Description | Ref Model? | Key Args |
|--------|-------------|:---:|----------|
| `ga_simple` | **Pure Gradient Ascent** on forget set only | No | — |
| `ga` | **Gradient Ascent** on forget + Gradient Descent on retain | No | — |
| `grad_diff` | **Gradient Difference** — weighted forget/retain NLL | No | `--forget-weight` |
| `dpo` | **Direct Preference Optimization** — forget=rejected, retain=chosen | Yes | `--beta` |
| `npo` | **Negative Preference Optimization** — DPO-inspired | Yes | `--beta` |
| `simnpo` | **Simple NPO** — reference-free variant of NPO | No | `--beta` |

#### Representation-level methods

| Method | Description | Ref Model? | Key Args |
|--------|-------------|:---:|----------|
| `rmu` | **Representation Misdirection** — MSE toward random targets | No | `--layer-id`, `--steering-coeff`, `--alpha` |
| `cb` | **Circuit Breakers** — cosine-similarity rerouting | No | `--layer-id`, `--steering-coeff`, `--alpha` |
| `lat` | **Latent Adversarial Training** — adversarial perturbation in hidden states | No | `--layer-id`, `--lat-eps`, `--lat-steps` |
| `cb_lat` | **CB + LAT combined** — Circuit Breakers with adversarial robustness | No | all CB + LAT args |

### Quick Start

```bash
# Run a single method (uses default model: EleutherAI/deep-ignorance-unfiltered)
./run_unlearn.sh ga

# Override defaults via environment variables
DEVICE=mps EPOCHS=3 LR=2e-5 ./run_unlearn.sh npo

# Or call the Python script directly
uv run --script unlearn.py \
  --model EleutherAI/deep-ignorance-unfiltered \
  --method simnpo \
  --outdir outputs/my_experiment/unlearned_model \
  --device auto --dtype auto --epochs 1
```

### Evaluation Split

By default, 10% of forget/retain data is held out for evaluation. After training, the script reports:
- **forget_NLL** — should be high (model forgot hazardous knowledge)
- **retain_NLL** — should be low (model still works on general text)
- **gap** — bigger = better unlearning

```bash
# Custom eval split (20%)
uv run --script unlearn.py --model ... --method ga --eval-split 0.2 --outdir ...

# Disable eval split (use all data for training)
uv run --script unlearn.py --model ... --method ga --eval-split 0 --outdir ...
```

### Output & Integration

Unlearned models are saved to `outputs/<model>__<method>/unlearned_model/` and can be fed directly into the analysis pipeline as `--model-b`:

```bash
# Parameter stats: compare original vs. unlearned
uv run --script collect_param_stats.py \
  --model-a EleutherAI/deep-ignorance-unfiltered \
  --model-b outputs/EleutherAI_deep-ignorance-unfiltered__ga/unlearned_model \
  --device auto --dtype auto --outdir outputs/EleutherAI_deep-ignorance-unfiltered__ga/param_stats

# Activation norms: measure representation changes on forget vs. retain data
uv run --script collect_activation_norms.py \
  --model-a EleutherAI/deep-ignorance-unfiltered \
  --model-b outputs/EleutherAI_deep-ignorance-unfiltered__ga/unlearned_model \
  --device auto --dtype auto --outdir outputs/EleutherAI_deep-ignorance-unfiltered__ga/activation_stats
```