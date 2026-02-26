# Subliminal Learning Enhancements to Unlearning Analysis Pipeline

This document describes the new analysis capabilities added to the unlearning repository, directly inspired by techniques developed in subliminal learning research (Experiments 2b, 3b, 5b from the takehome project).

## Overview of New Capabilities

The subliminal learning research revealed powerful techniques for understanding training dynamics and factor interactions that were missing from traditional unlearning analysis. These enhancements address critical gaps:

- **Dynamic Analysis**: Traditional unlearning analysis focused on static weight comparisons. The new tools track training dynamics and optimization stability.
- **Factor Decomposition**: Replaced qualitative analysis with quantified variance decomposition across experimental factors.
- **Hyperparameter Optimization**: Systematic visualization of trade-off curves rather than trial-and-error parameter selection.
- **Robustness Testing**: Evaluation across different input distributions to identify brittleness.

## New Analysis Steps

### Step 13: Gradient Dynamics Analysis
**File**: `experiment/gradient_dynamics_analysis.py`

**Inspiration**: Subliminal learning Experiment 2b revealed that real MNIST produced 13× larger gradient norms than uniform noise, causing instability.

**What it does**: Tracks per-batch gradient norms across layers during simulated unlearning training to reveal optimization stability patterns.

**Key insights expected**:
- Robust methods (CB-LAT, filtering) → controlled, stable gradient norms
- Brittle methods (simple GA) → large, variable, potentially exploding gradients
- Optimal hyperparameters → consistent gradient magnitudes across training

**Usage**:
```bash
uv run experiment/gradient_dynamics_analysis.py \
  --model-a EleutherAI/deep-ignorance-unfiltered \
  --model-b EleutherAI/deep-ignorance-unfiltered-cb-lat \
  --outdir outputs/gradient_analysis \
  --method ga \
  --num-steps 100
```

**Outputs**:
- `gradient_dynamics.csv`: Per-step gradient norms by component
- `gradient_analysis_summary.json`: Stability metrics and comparisons
- `plots/gradient_dynamics.png`: Gradient norm timelines and distributions
- `plots/gradient_heatmap_layers.png`: Layer-wise gradient norm heatmap

### Step 14: Distance from Initialization Analysis
**File**: `experiment/distance_from_init_analysis.py`

**Inspiration**: Subliminal learning Experiment 3b showed peak performance at specific distances from initialization (≈4-5 units), revealing the "alignment basin" concept.

**What it does**: Measures how far unlearned models drift from their pretrained initialization, identifying optimal "Goldilocks zones" for parameter updates.

**Key insights expected**:
- Robust methods stay within optimal distance from pretrained weights
- Brittle methods may show excessive drift or insufficient change
- Distance correlates with tamper-resistance (harder to fine-tune back)

**Usage**:
```bash
uv run experiment/distance_from_init_analysis.py \
  --init-model EleutherAI/deep-ignorance-unfiltered \
  --target-model EleutherAI/deep-ignorance-unfiltered-cb-lat \
  --outdir outputs/distance_analysis
```

**Outputs**:
- `distance_per_matrix.csv`: Per-parameter distance metrics
- `distance_per_component.csv`: Component-level aggregates
- `distance_per_layer.csv`: Layer-wise analysis
- `distance_summary.json`: Key metrics and most-changed components
- `plots/distance_analysis.png`: Multi-panel distance visualization
- `plots/distance_heatmap.png`: Layer × component distance heatmap

### Step 15: Input Distribution Sensitivity Analysis
**File**: `experiment/input_distribution_sensitivity.py`

**Inspiration**: Subliminal learning Experiment 2 showed that input distribution critically affects learning stability—structured inputs hurt while unstructured inputs helped.

**What it does**: Tests unlearning robustness across different forget/retain data distributions to identify brittle methods that only work on specific input types.

**Key insights expected**:
- Methods may work well on synthetic data but fail on real-world distributions
- Domain shifts in retain data can break unlearning effectiveness
- Adversarial input crafting can expose weaknesses

**Usage**:
```bash
uv run experiment/input_distribution_sensitivity.py \
  --base-model EleutherAI/deep-ignorance-unfiltered \
  --unlearned-model EleutherAI/deep-ignorance-unfiltered-cb-lat \
  --outdir outputs/distribution_sensitivity \
  --forget-distributions random_tokens shuffled_wmdp adversarial_wmdp \
  --retain-distributions original domain_shift length_shift
```

**Outputs**:
- `distribution_sensitivity_results.csv`: Performance across all combinations
- `distribution_analysis_summary.json`: Best/worst combinations and summary stats
- `plots/distribution_sensitivity_analysis.png`: Heatmaps of effectiveness across distributions

## Standalone Analysis Tools

### Multi-Factor Variance Decomposition
**File**: `experiment/factor_decomposition_analysis.py`

**Inspiration**: Subliminal learning Experiment 5b decomposed variance into animal identity (68.1%), prompt template (14.6%), geometric factors (0%), revealing which factors truly matter.

**What it does**: Decomposes unlearning effectiveness variance across experimental factors using linear regression, random forests, and ANOVA.

**Usage**:
```bash
uv run experiment/factor_decomposition_analysis.py \
  --results-csv unlearn/analysis/sweep_results.csv \
  --target-cols wmdp_accuracy mmlu_accuracy \
  --outdir outputs/factor_analysis
```

**Key outputs**:
- R² decomposition showing which factors explain variance
- Factor importance rankings across multiple methods
- Statistical significance tests (ANOVA p-values)
- Interaction effect analysis

### Goldilocks Curve Visualization
**File**: `experiment/goldilocks_curve_analysis.py`

**Inspiration**: Subliminal learning Experiment 3b's scatter plot directly revealed the inverted-U relationship between distance from initialization and performance.

**What it does**: Creates systematic visualizations of hyperparameter-performance relationships, identifying optimal "Goldilocks zones."

**Usage**:
```bash
uv run experiment/goldilocks_curve_analysis.py \
  --results-csv unlearn/analysis/sweep_results.csv \
  --hyperparameter-cols learning_rate epochs batch_size \
  --target-cols wmdp_accuracy mmlu_accuracy \
  --outdir outputs/goldilocks_analysis
```

**Key outputs**:
- 1D Goldilocks curves for each hyperparameter
- 2D Pareto frontier plots (WMDP vs MMLU trade-offs)
- 3D surface plots for multiple hyperparameters
- Optimal hyperparameter values with confidence intervals

### Post-Hoc Analysis Runner
**File**: `experiment/run_post_analysis.py`

**What it does**: Convenient wrapper to run factor decomposition and Goldilocks analysis on existing experimental results.

**Usage**:
```bash
# Run both analyses with auto-detection
uv run experiment/run_post_analysis.py \
  --mode both \
  --results-csv unlearn/analysis/sweep_results.csv \
  --outdir post_analysis_results

# Run specific analysis with manual column specification
uv run experiment/run_post_analysis.py \
  --mode goldilocks \
  --results-csv sweep_results.csv \
  --hyperparameter-cols lr epochs method \
  --target-cols wmdp_bio_robust mmlu
```

## Integration with Existing Pipeline

The new analysis steps are fully integrated into the main pipeline (`experiment/pipeline.sh`):

1. **Steps 13-15 run automatically** after the existing 12 steps
2. **Same device/dtype configuration** as other analyses
3. **Same skipping logic** - completed steps are automatically detected and skipped
4. **Same output structure** under `outputs/<comparison>/`

Run the enhanced pipeline:
```bash
# Run all analyses including new steps
./experiment/pipeline.sh

# Run specific comparisons with new steps
UNLEARNED=your-model ./experiment/pipeline.sh
```

## Expected Impact on Unlearning Research

### 1. **Optimization Stability Detection**
Gradient dynamics analysis will reveal whether "brittle" unlearning methods are characterized by unstable optimization patterns, providing an early warning system for methods that may fail under fine-tuning attacks.

### 2. **Tamper-Resistance Quantification**
Distance-from-initialization tracking provides a direct metric for tamper-resistance—models that move further from pretraining should be harder to recover via adversarial fine-tuning.

### 3. **Systematic Hyperparameter Optimization**
Goldilocks curves replace trial-and-error hyperparameter selection with systematic visualization of trade-off spaces, accelerating method development.

### 4. **Robustness Evaluation Standards**
Distribution sensitivity analysis establishes rigorous evaluation standards—methods that only work on specific data distributions are identified as brittle.

### 5. **Mechanistic Understanding**
Factor decomposition reveals which aspects of unlearning methods actually matter, enabling researchers to focus effort on the factors that drive effectiveness rather than pursuing irrelevant optimizations.

## Comparison with Original Subliminal Learning Insights

| Subliminal Learning Finding | Unlearning Application | Expected Discovery |
|---|---|---|
| Real MNIST → 13× larger gradients → instability | Real vs synthetic forget data → different gradient patterns | Some unlearning methods may be unstable on real-world data |
| Peak at distance ≈4-5 from init → Goldilocks zone | Optimal parameter drift → tamper-resistance | Quantified relationship between parameter change and robustness |
| Animal identity = 68.1% variance → dominant factor | Method vs hyperparams vs architecture → factor importance | Which aspects of unlearning design actually matter most |
| Geometric factors = 0% magnitude prediction | Weight similarities vs effectiveness → mechanistic insights | Whether geometric intuitions about unlearning are correct |
| Base vs instruct = binary prerequisite | Different model families/sizes → transferability | Which unlearning insights generalize across model types |

## Technical Notes

### Dependencies
All new scripts use the existing `utils.py` infrastructure and require the same dependencies as the main pipeline. New requirements:
- `scipy` (for curve fitting)
- `statsmodels` (for ANOVA)
- `seaborn` (for enhanced visualizations)

### Performance Considerations
- **Gradient dynamics**: Lightweight—only computes gradients, doesn't update weights
- **Distance analysis**: Parameter-loading only—no forward passes required
- **Distribution sensitivity**: Most expensive—requires inference on multiple datasets
- **Factor/Goldilocks**: Analysis-only—operates on existing result CSVs

### Extensibility
Each analysis tool is designed to be:
- **Modular**: Can be run independently or as part of the pipeline
- **Configurable**: Extensive CLI options for customization
- **Extensible**: Clear APIs for adding new metrics or visualizations
- **Reusable**: Compatible with any unlearning experimental setup

This enhancement transforms the unlearning repository from a static analysis toolkit into a comprehensive framework for understanding optimization dynamics, factor interactions, and robustness patterns—directly applying the mechanistic insights discovered through subliminal learning research.