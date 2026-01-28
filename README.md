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

If you want to run steps individually or understand what `pipeline.sh` is doing, here is the breakdown:

### 1. Parameter Analysis
```bash
./run_params_local.sh
```
Comparses the baseline model against the fine-tuned versions. computes Frobenius norms and Stable Ranks for every weight matrix.
*   **Input**: HuggingFace models.
*   **Output**: `outputs/param_stats/...` (CSVs with raw stats).

### 2. Plot Parameter Stats
```bash
./plot_param_stats.sh
```
Visualizes the CSVs generated in step 1.
*   **Output**: `plots/filtered/*.png`, `plots/unlearned/*.png`.

### 3. Data Generation
```bash
uv run create_datasets.py
```
Downloads WMDP-Bio (hazardous knowledge) and Wikitext (general knowledge) datasets.
*   **Output**: `data/forget.txt`, `data/retain.txt`.

### 4. Activation Analysis
```bash
./run_activations.sh
```
Runs the models on the datasets from step 3 and records the average hidden-state norms for every layer.
*   **Output**: `outputs/activation_norms/activation_norms.csv`.

### 5. Plot Activations
```bash
./plot_activation_norms.sh
```
Visualizes the activation trends to show "Unlearning Gaps".
*   **Output**: `plots/activations/*.png`.

---

## Script Details

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
*   **Stable Rank ($r_{stable}$)**:
    *   **Formula**: $r_{stable} = \frac{\|A\|_F^2}{\|A\|_2^2}$.
    *   **Interpretation**: A proxy for the "effective rank" of the matrix.
        *   **Low Stable Rank ($\approx 1$)**: The change is "spiky" or "rank-1". All the energy is concentrated in one specific direction (singular vector). This often indicates a precise, surgical edit.
        *   **High Stable Rank**: The change is "blurry" or isotropic. The energy is spread out across many dimensions. This often indicates noise or a general "drift" in the weights.

#### Outputs
**`outputs/param_stats/<model_pair>/per_matrix.csv`**
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

**`outputs/param_stats/<model_pair>/per_layer.csv`**
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
**Purpose**: Measures the magnitude or "confidence" of the model's internal representations on specific datasets.

#### Data Sources
*   **Forget Set (`data/forget.txt`)**: Generated (via `create_datasets.py`) from **WMDP-Bio**. These are questions about hazardous biological knowledge that we want the model to "forget".
*   **Retain Set (`data/retain.txt`)**: Generated from **Wikitext-2**. These are general English texts that we want the model to preserve capabilities on.

#### Outputs
**`outputs/activation_norms/activation_norms.csv`**
| Column | Description |
| :--- | :--- |
| `model` | The HF model ID or path |
| `split` | Dataset split (`forget` or `retain`) |
| `layer` | Layer index (0 to N) |
| `mean_norm` | The average L2 norm of the hidden states in that layer: $\mathbb{E}[\sqrt{\sum h_i^2}]$ |

#### Interpretation
*   **Effective Unlearning**: You want `mean_norm` to **drop** on the `forget` split (less activation/confidence on hazardous topics) while staying **constant** on the `retain` split (general capabilities preserved).
*   **Lobotomy**: If `mean_norm` drops to zero or explodes everywhere, the model is broken.

---

### 3. Plots
The scripts in `plots/` visualize the CSVs generated above. Here is exactly where the data comes from:

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

*   **Activation Profiles (`activation_norms_*.png`)**
    *   **Source Data**: `activation_norms.csv`
    *   **Column Plotted**: `mean_norm`
    *   **Interpretation**:
        *   *X-axis*: Layer Index.
        *   *Y-axis*: Average L2 norm of hidden states.
        *   *Meaning*: Shows signal propagation strength. "Unlearning" is successful if the curve **drops** for the Forget Set (Blue) but stays **stable** for the Retain Set (Orange).  