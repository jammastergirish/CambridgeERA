# Model Diffs Analysis Tools

This directory contains scripts for analyzing the differences between PyTorch models (e.g., a baseline vs. a fine-tuned/unlearned version). It focuses on parameter statistics (Frobenius norms, Stable Rank) and activation norms.

## Recent Improvements

- **Enhanced Training**: Gradient accumulation, configurable gradient clipping, periodic validation
- **Memory Optimization**: Half-precision activation caching (50% I/O reduction), smart parameter streaming
- **Robustness**: Architecture fallback warnings, automatic device/dtype selection
- **Documentation**: Detailed mathematical explanations for all 10 unlearning methods
- **Reproducibility**: Fixed random seeds throughout pipeline

## ⚠️ Important: Experimental Reproducibility

When running experiments for research or comparison:

1. **Gradient Accumulation**: While useful for memory-constrained systems, gradient accumulation (`--grad-accum-steps > 1`) is NOT equivalent to true larger batch training. The optimizer statistics and convergence behavior may differ.

2. **FP16 Caching**: The `--cache-fp16` flag trades precision for speed. For exact reproducibility across systems, always use the same caching mode.

3. **Recommended for Experiments**:
   - Use default settings: `--grad-accum-steps 1` (no accumulation)
   - Omit `--cache-fp16` (use full precision)
   - Document all settings in your experimental logs
   - The scripts will emit warnings when optimization modes are active

4. **Use Optimizations Only When**:
   - Running on memory-constrained systems
   - Doing exploratory analysis (not final experiments)
   - Speed is more important than exact reproducibility

## Experimental Best Practices

### 1. Dataset Considerations

**Data Leakage**: Ensure forget and retain sets are truly disjoint:
```python
# Check for overlap (add to your preprocessing)
forget_set = set(open('data/forget.txt').readlines())
retain_set = set(open('data/retain.txt').readlines())
overlap = forget_set & retain_set
assert len(overlap) == 0, f"Found {len(overlap)} overlapping samples!"
```

**Dataset Size Balance**: Unbalanced datasets can bias results:
- Keep forget/retain sets roughly equal sized
- Or use weighted sampling if they must differ
- Document exact counts in your results

**Domain Shift**: Ensure retain set represents the model's general domain:
- Bad: Forget=biology, Retain=poetry (domain shift confounds results)
- Good: Forget=harmful biology, Retain=general text including safe biology

### 2. Evaluation Metrics

**Beyond NLL**: Single metrics can be misleading:
```bash
# Generate samples to check for mode collapse
python -c "from transformers import pipeline; \
  p = pipeline('text-generation', model='outputs/unlearned_model'); \
  print(p('The DNA sequence', max_length=50, num_return_sequences=5))"
```

**Membership Inference**: Test if the model truly "forgot":
- Higher confidence on retain than forget = good
- But also check: can an attacker still detect if text was in forget set?

**Downstream Tasks**: Measure capability preservation:
- Run standard benchmarks (MMLU, HellaSwag, etc.) before/after
- Significant drops indicate over-forgetting

### 3. Statistical Rigor

**Multiple Seeds**: Single runs can be misleading:
```bash
# Run with multiple seeds
for seed in 42 1337 2024; do
  uv run --script unlearn.py \
    --method ga --seed $seed \
    --outdir outputs/ga_seed${seed}
done
```

**Significance Testing**: Use appropriate statistical tests:
- Paired t-test for before/after comparisons
- Bootstrap confidence intervals for small sample sizes
- Report variance, not just means

**Early Stopping**: Avoid overfitting to your evaluation set:
```bash
# Use validation for early stopping, hold out separate test set
--eval-split 0.2  # 20% for validation
# Keep another 10% completely untouched for final evaluation
```

### 4. Hyperparameter Selection

**Grid Search Pitfalls**:
- Don't optimize hyperparameters on your test set
- Use a separate validation set for hyperparameter tuning
- Report results on held-out test set with fixed hyperparameters

**Method-Specific Defaults May Not Transfer**:
```python
# Learning rates that work for GA might be terrible for DPO
methods_lr = {
    'ga': 1e-5,
    'dpo': 5e-7,  # Often needs lower LR due to reference model
    'rmu': 1e-6,  # Representation methods might need different scale
}
```

### 5. Computational Considerations

**Hardware Variability**: Different hardware can affect results:
- FP16 on A100 ≠ FP16 on V100 (different tensor cores)
- MPS (Apple Silicon) may have different numerics than CUDA
- Document exact hardware: GPU model, driver version, CUDA version

**Batch Size Effects**: Batch size affects more than just memory:
- Larger batches → different gradient noise → different convergence
- Some methods (especially DPO) are sensitive to batch size
- Keep batch size constant across all comparison experiments

### 6. Reporting Guidelines

**What to ALWAYS Report**:
```yaml
# In your paper/report, always include:
Model:
  base: "EleutherAI/pythia-2.8b"
  parameters: 2.8B
  dtype: bfloat16

Dataset:
  forget_size: 500
  retain_size: 500
  forget_domain: "WMDP-bio"
  retain_domain: "WikiText-2"

Training:
  method: "ga"
  learning_rate: 1e-5
  batch_size: 4
  epochs: 1
  gradient_clip: 1.0
  seed: 42

Hardware:
  device: "NVIDIA A100 40GB"
  cuda: "11.8"
  pytorch: "2.1.0"

Results:
  forget_nll: 5.23 ± 0.15  # Mean ± std over 3 seeds
  retain_nll: 2.87 ± 0.08
  runtime: "23 minutes"
```

### 7. Common Pitfalls to Avoid

**Catastrophic Forgetting**: Model forgets everything, not just target:
- Always monitor retain_NLL during training
- If retain_NLL increases significantly, reduce learning rate

**Superficial Unlearning**: Model just adds noise to outputs:
- Check if model can still complete prompts coherently
- Test with paraphrases of forget data

**Cherry-picking**: Selecting favorable examples:
- Use fixed, predetermined test sets
- Report aggregate statistics, not just best cases

### 8. Ablation Studies

Essential ablations for any unlearning paper:
```bash
# 1. Learning rate sensitivity
for lr in 1e-6 1e-5 1e-4; do
  uv run --script unlearn.py --method ga --lr $lr ...
done

# 2. Data size scaling
for n in 100 500 1000; do
  head -n $n data/forget.txt > data/forget_${n}.txt
  uv run --script unlearn.py --forget-data data/forget_${n}.txt ...
done

# 3. Layer targeting (for RMU/CB/LAT)
for layers in "5,6,7" "10,11,12" "15,16,17"; do
  uv run --script unlearn.py --method rmu --layer-id "$layers" ...
done
```

### 9. Reproducibility Checklist

Before publishing results:
- [ ] Code runs from clean clone: `git clone ... && cd ... && uv run --script unlearn.py ...`
- [ ] Random seeds fixed and documented
- [ ] Exact package versions recorded: `pip freeze > requirements.txt`
- [ ] Data available or recreation steps provided
- [ ] Results reproducible on different machines (within numerical tolerance)

### 10. Ethical Considerations

**Unlearning Verification**: Can you actually verify forgetting?
- Consider adversarial actors trying to recover "forgotten" information
- Test with various prompting strategies (few-shot, chain-of-thought, etc.)

**Selective vs. Broad**: Document the trade-off:
- Targeted unlearning: Preserves capabilities but may leave traces
- Broad unlearning: More thorough but damages general performance

## Quick Start

Before running experiments, check your setup:
```bash
python check_experiment.py  # Runs sanity checks for common issues
```

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

Run unlearning experiments with **10 different methods** via a single script. Includes automatic train/eval splitting, tqdm progress bars, and numerous optimization features.

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

### Mathematical Details of Unlearning Methods

#### Loss-based Methods

**Gradient Ascent (GA)**: The simplest approach - maximize loss on forget data to "corrupt" the model's predictions.
- `ga_simple`: L = -NLL(forget) — Pure gradient ascent
- `ga`: L = -NLL(forget) + NLL(retain) — Balanced forget/retain
- `grad_diff`: L = NLL(retain) - α·NLL(forget) — Weighted difference

**Direct Preference Optimization (DPO)**: Frames unlearning as preference learning without reward models.
- **Loss**: -log σ(β·(log π(retain)/π_ref(retain) - log π(forget)/π_ref(forget)))
- **Intuition**: Increase likelihood ratio for retain data (preferred), decrease for forget data (dispreferred)
- **β parameter**: Inverse temperature controlling preference strength (lower = stronger)

**Negative Preference Optimization (NPO)**: Variant that explicitly pushes down forget likelihood.
- **Loss**: -(2/β)·E[log σ(-β·log(π(forget)/π_ref(forget)))] + NLL(retain)
- **Intuition**: Sigmoid saturates when π(forget) << π_ref(forget), preventing over-optimization
- **SimNPO**: Reference-free variant using absolute log-probs instead of ratios

#### Representation-level Methods

**Representation Misdirection (RMU)**: Steers internal representations using MSE loss.
- **Forget**: ||h_forget - α·r||² where r is random unit vector — corrupts understanding
- **Retain**: ||h_retain - h_original||² — preserves capabilities
- **Key idea**: Random targets act as "attractors" pulling forget representations into meaningless directions

**Circuit Breakers (CB)**: Rewires model by changing activation flow directions using cosine similarity.
- **Forget**: -cos(h_forget, α·r) — align with random directions
- **Retain**: 1 - cos(h_retain, h_original) — preserve original "circuits"
- **Advantage**: Scale-invariant (focuses on direction not magnitude)

**Latent Adversarial Training (LAT)**: Makes unlearning robust against adversarial recovery.
1. **Inner loop**: Find perturbation δ* = argmin_δ L_forget(x + δ) s.t. ||δ||∞ ≤ ε
   - Simulates adversary trying to recover forgotten knowledge
   - Uses projected gradient descent: δ ← clip(δ - ε·sign(∇_δ L), [-ε, ε])
2. **Outer loop**: Train to forget even WITH optimal perturbation
   - L = -NLL_forget(x + δ*) + NLL_retain(x)
   - Prevents brittle unlearning that's easily reversed

### Enhanced Features (New)

#### Training Optimizations
- **Gradient Accumulation** (`--grad-accum-steps`): Simulate larger batch sizes on limited memory (set to 1 to disable)
- **Configurable Gradient Clipping** (`--grad-clip`): Adjust or disable gradient clipping (default: 1.0, 0 to disable)
- **Periodic Validation** (`--eval-interval`): Monitor forget/retain NLL during training for early stopping
- **Fixed Random Seeds** (`--seed`): Ensures reproducible results across runs

#### Memory Optimizations
- **Half-precision activation caching**: Optionally reduces disk I/O by 50% during activation analysis (`--cache-fp16`)
- **Smart parameter loading**: Streams model shards one at a time to avoid OOM
- **Gradient checkpointing**: Automatically enabled during training to save GPU memory

#### Robustness Features
- **Architecture fallback warnings**: Clear warnings when LAT/CB-LAT encounter unsupported architectures
- **Automatic dtype selection**: Chooses optimal precision based on hardware (bf16 for CUDA, fp16 for MPS)
- **Device auto-detection**: Automatically uses best available accelerator (CUDA > MPS > CPU)

### Quick Start

```bash
# Run a single method (uses default model: EleutherAI/deep-ignorance-unfiltered)
./run_unlearn.sh ga

# Override defaults via environment variables
DEVICE=mps EPOCHS=3 LR=2e-5 ./run_unlearn.sh npo

# Advanced: Use gradient accumulation for larger effective batch size
uv run --script unlearn.py \
  --model EleutherAI/deep-ignorance-unfiltered \
  --method simnpo \
  --batch-size 2 \
  --grad-accum-steps 4 \  # Effective batch size = 8
  --grad-clip 0.5 \        # Stronger clipping for stability
  --eval-interval 50 \     # Validate every 50 steps
  --outdir outputs/my_experiment/unlearned_model \
  --device auto --dtype auto --epochs 1

# Disable gradient accumulation and clipping for simple training
uv run --script unlearn.py \
  --model EleutherAI/deep-ignorance-unfiltered \
  --method ga \
  --grad-accum-steps 1 \   # No accumulation (default)
  --grad-clip 0 \           # No gradient clipping
  --outdir outputs/simple_test/unlearned_model
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

# Enable periodic validation during training
uv run --script unlearn.py --model ... --method ga --eval-interval 100 --outdir ...
```

### Complete Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | (required) | Base model HF ID or local path |
| `--method` | (required) | Unlearning method (see methods table) |
| `--forget-data` | `data/forget.txt` | Path to forget prompts |
| `--retain-data` | `data/retain.txt` | Path to retain prompts |
| `--outdir` | (required) | Output directory for unlearned model |
| `--device` | `auto` | Device (auto/cuda/mps/cpu) |
| `--dtype` | `auto` | Data type (auto/fp32/fp16/bf16) |
| `--lr` | `1e-5` | Learning rate |
| `--epochs` | `1` | Number of training epochs |
| `--batch-size` | `4` | Batch size per gradient step |
| `--grad-accum-steps` | `1` | Gradient accumulation steps |
| `--max-length` | `512` | Maximum sequence length |
| `--beta` | `0.1` | Inverse temperature for DPO/NPO/SimNPO |
| `--alpha` | `100.0` | Retain weight for RMU/CB |
| `--steering-coeff` | `20.0` | Steering coefficient for RMU/CB |
| `--layer-id` | `"5,6,7"` | Target layers for RMU/CB/LAT |
| `--forget-weight` | `1.0` | Forget weight for GradDiff |
| `--lat-eps` | `0.1` | Perturbation budget for LAT |
| `--lat-steps` | `5` | Adversarial steps for LAT |
| `--eval-split` | `0.1` | Validation data fraction |
| `--eval-interval` | `0` | Validate every N steps (0=disable) |
| `--grad-clip` | `1.0` | Gradient clipping norm (0=disable) |
| `--seed` | `42` | Random seed for reproducibility |

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

# Use half-precision caching to reduce disk I/O by 50%
uv run --script collect_activation_norms.py \
  --model-a EleutherAI/deep-ignorance-unfiltered \
  --model-b outputs/EleutherAI_deep-ignorance-unfiltered__ga/unlearned_model \
  --forget-text data/forget.txt \
  --retain-text data/retain.txt \
  --cache-fp16 \  # Enable FP16 caching
  --device auto --dtype auto --outdir outputs/fast_activation_stats
```