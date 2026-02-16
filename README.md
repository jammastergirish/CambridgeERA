# What Makes Unlearning Brittle? A Mechanistic Study of Parameter-Space Interventions

This repository contains a pipeline for creating unlearned large language models using various unlearning algorihtms, and then analyzing how these methods alter models in both parameter and activation spaces. The goal is to identify mechanistic signatures that distinguish deep representational change from shallow parameter patching.

## Initial Proposal

The field of machine unlearning proposes a range of post-training interventions intended to remove sensitive or harmful information from large language models. However, these methods are brittle under further fine-tuning or adversarial tampering, and there is very limited mechanistic understanding of why unlearning approaches are so shallow. I would like to study unlearning methods through the lens of mechanistic interpretability, treating them as targeted parameter-space interventions whose internal effects can be directly analyzed. 

I analyze unlearning’s internal effects using various model-agnostic diagnostics that characterize how much models change, where those changes occur, and how high-dimensional they are, using layer-wise norms and effective rank, alongside activation-level diagnostics on “forget” and “retain” datasets. Initial results compare baseline, pretraining-filtered (demonstrating actual ignorance), and post-training unlearned models. This reveals distinct regimes: pretraining-time filtering induces large, high-rank, distributed updates consistent with deep representational change, while post-training methods such as Circuit Breaking with Latent Adversarial Training produce small, low-rank, attention-localized edits that leave feature computation largely intact. Activation norms appear to show that these differences are not explained by global suppression.

The project will expand across a broader set of unlearning algorithms (e.g., RMU, gradient-based, latent adversarial), and will also study how training dynamics—such as optimizer choice—shape the geometry of unlearning updates. Additional diagnostics may include representation-level analyses such as sparse autoencoders. By identifying mechanistic signatures associated with brittleness and partial robustness—such as update rank, depth, and distribution—this work aims to equip researchers with the information needed to design more effective approaches to tamper-resistant unlearning, rather than claiming success for existing methods.

---

## Datasets

`uv run create_datasets.py` creates two text datasets that serve in training for unlearning, and as *probes* for activation-level analyses:

| Dataset | Source | Purpose |
|---|---|---|
| `forget.txt` | WMDP-Bio questions | Text the model *should* have forgotten — the "target" of unlearning |
| `retain.txt` | WikiText-2 | Benign text the model *should* still handle well — the "control" |

These are analogous to stimulus and control conditions in an experiment. Every activation-level diagnostic (Steps 8–12) runs on *both* datasets, measuring whether interventions selectively affect forget-domain processing while preserving retain-domain processing.

---

## Experiment

### Quick Start

- Add `HF_TOKEN`and `WANDB_API_KEY` to `.env`.
- Ensure the `uv` package manager is installed.
- Run:

```bash
./pipeline.sh
```
---

### The Experimental Setup

The pipeline performs a **controlled experiment** with three models sharing identical architecture. 

| Model | Role | What happened to it |
|---|---|---|
| `deep-ignorance-unfiltered` | **Base** (control) | Trained on everything, including WMDP-Bio hazardous content |
| `deep-ignorance-e2e-strong-filter` | **Filtered** (gold standard) | Trained from scratch with hazardous data *removed before training* |
| `deep-ignorance-unfiltered-XXXXXX` | **Unlearned** (intervention) | Same as Base, but post-hoc unlearned |

Every diagnostic runs **twice** — once for each comparison — always using the Base model as the reference:

```
Comparison 1:  Base → Filtered     (What does genuine ignorance look like?)
Comparison 2:  Base → Unlearned    (What does post-hoc unlearning look like?)
```

By contrasting these two comparisons, you can distinguish *deep representational change* (filtering) from *shallow parameter patching* (unlearning).

---

### The Diagnostics: What and Why

The 13 steps form a hierarchy from **coarse to fine**, and from **weight-space to activation-space**:

```mermaid
graph TD
    A["Weight-Space Diagnostics<br/>(Steps 1-2, 6-7, 10)"] --> C["How much did parameters change?<br/>Where? In what directions?"]
    B["Activation-Space Diagnostics<br/>(Steps 4-5, 8-9, 11-12)"] --> D["How do those changes affect<br/>what the model computes?"]
    F["Knowledge Localization<br/>(Step 13)"] --> G["Where is the forget-set knowledge<br/>encoded in each model?"]
    C --> E["Mechanistic Signature<br/>of the Intervention"]
    D --> E
    G --> E
```

---

#### Weight-Space Diagnostics

These examine `ΔW = W_modified − W_base` directly — treating the intervention as a matrix perturbation.

#### Steps 1–2: Parameter Statistics (`collect_param_stats.py` + `plot_param_stats.py`)

**Question:** *How large is the intervention, and where is it concentrated?*

For every weight matrix `W` in the model, this computes:

| Metric | Formula | What it tells you |
|---|---|---|
| **Relative Frobenius norm** of ΔW | ‖ΔW‖_F / ‖W‖_F | Normalized magnitude of change — what fraction of the original weight moved? Comparable across layers regardless of matrix size. |
| **Frobenius norm** of ΔW | ‖ΔW‖_F = √(Σᵢⱼ ΔWᵢⱼ²) | Raw total magnitude (unnormalized; also recorded for completeness) |
| **Spectral norm** of ΔW | σ₁(ΔW) / σ₁(W) | Relative worst-case amplification — how much did the dominant singular direction shift? High spectral + low stable rank = a sharp rank-1 spike. |
| **Stable rank** of ΔW | ‖ΔW‖²_F / ‖ΔW‖²₂ | Effective dimensionality of the update. A rank-1 perturbation (e.g., LoRA-style) gives stable rank ≈ 1. A full-rank rewrite gives stable rank ≈ min(m,n). |
| **Stable rank** of W | Same, on original | Baseline dimensionality for comparison |
| **Empirical rank** (opt-in: `--empirical-rank`) | min k s.t. Σᵢᵏ σᵢ² ≥ 0.99·Σ σᵢ² | Discrete count of dimensions capturing 99% of variance (requires full SVD, so slow, so we default to not do this) |

These are aggregated per layer and split into **MLP vs Attention** groups, then plotted. The layer locality plot uses the **relative** Frobenius norm so layers are directly comparable; a separate spectral norm plot shows worst-case amplification per layer.

**Why this matters:** If unlearning produces low-rank, localized updates (small relative ‖ΔW‖_F concentrated in a few layers) while filtering produces high-rank, distributed updates, that's direct evidence that unlearning is a *shallow patch* rather than a *deep restructuring*. The stable rank quantifies this precisely — it's the "soft" version of matrix rank, robust to noise.

---

##### Step 6: MLP vs Attention Breakdown (`analyze_mlp_vs_attn.py`)

**Question:** *Are the changes concentrated in MLP (knowledge storage) or Attention (routing/composition)?*

Takes the per-matrix stats from Step 1 and computes the ratio of MLP change to Attention change at each layer. Addresses the mechanistic hypothesis that knowledge is primarily stored in MLP layers (the "key-value memory" view from [Geva et al](https://arxiv.org/pdf/2012.14913).), while attention layers handle routing.

**Why this matters:** If unlearning only modifies attention layers, it might be redirecting *routing around* the knowledge rather than erasing it — explaining why adversarial fine-tuning can recover the information.

---

##### Step 7: Null Space & Subspace Analysis (`null_space_analysis.py`)

**Question:** *Is the update low-rank, and do the principal subspaces shift?*

For 50 sampled weight matrices, computes full SVD and measures:

| Metric | What it tells you |
|---|---|
| **Top-10 SV variance ratio** | What fraction of ΔW's energy is in its top 10 singular directions? High → very low-rank update. |
| **Effective rank** | How many singular values needed to capture 99% of variance |
| **Subspace alignment** (Grassmann distance) | Do the top-k singular vectors of W_base and W_modified span similar subspaces? High alignment → the intervention didn't change *what directions* the matrix uses, only *how much* it uses them. |

**Why this matters:** A low-rank ΔW with high subspace alignment means the unlearning intervention is a small perturbation within the existing computational manifold — it didn't rewire the representations, it just nudged the gains. This is precisely the geometric signature of brittleness: a small counter-perturbation (fine-tuning) can undo it.

---

##### Step 10: MLP Nullspace Alignment (`mlp_nullspace_alignment.py`)

**Question:** *Does ΔW lie in the nullspace of the original W?*

Decomposes each MLP update ΔW into components that lie in the **column space** vs. **null space** of the original weight matrix W.

- **Nullspace component** ("off-manifold"): Changes orthogonal to what W originally computed. These add new directions without disrupting existing computations.
- **Column space component** ("on-manifold"): Changes that directly interfere with existing computations.

**Why this matters:** If unlearning updates are primarily in the nullspace, the model's existing computations are barely disturbed — the "unlearned" knowledge may still flow through the same channels, just with a small additive correction that's easy to remove. True knowledge erasure should require on-manifold changes that destroy the original computation.

---

#### Activation-Space Diagnostics

These run the model on actual text and measure *what it computes*, not just what its parameters look like. All activation scripts cap input at `--max-samples 500` texts per split by default to keep runtimes manageable (override with e.g. `--max-samples 1000` for more statistical power).

#### Steps 4–5: Activation Norms (`collect_activation_norms.py` + `plot_activation_norms.py`)

**Question:** *Does the intervention globally suppress or amplify activations?*

For each layer, computes the mean L1 and L2 norms of hidden states per token, plus the norm of the *difference* in activations (‖h_modified − h_base‖). Run on both forget and retain texts.

| Norm | Formula (per token) | What it captures |
|---|---|---|
| **L1** | Σᵢ |hᵢ| | Total activation mass — sensitive to diffuse, low-magnitude changes across many dimensions |
| **L2** | √(Σᵢ hᵢ²) | Activation magnitude — sensitive to large spikes in individual dimensions |

Both are averaged across all tokens (weighted by attention mask). They are **not** divided by hidden dimension since all models share the same architecture, so they are directly comparable across models and layers.

**Why this matters:** If norms are similar between base and unlearned models but different for the filtered model, it means unlearning doesn't achieve suppression through reducing activation magnitudes — it's doing something more subtle (or less effective). L1 and L2 capture different aspects: L1 is more sensitive to many small changes spread across dimensions, while L2 is dominated by the largest components.

---

##### Step 8: Activation Separation (`activation_separation_analysis.py`)

**Question:** *Can you tell forget-text activations apart from retain-text activations? Does the intervention change this?*

At each layer, extracts the centroid of forget-text activations and retain-text activations, then measures their separation via:

| Metric | What it captures |
|---|---|
| **Cosine distance** between centroids | Direction-based separation |
| **AUC** (linear classifier) | How linearly separable are the two distributions? |
| **Variance ratio** | Between-class vs. within-class variance (like Fisher's discriminant) |

**Why this matters:** If unlearning *increases* the separation between forget and retain activations (pushes them apart), that's evidence the model is actively routing forget-text to a different computational path. If separation stays similar, the model treats both text types the same way internally — it hasn't genuinely distinguished "knowledge to suppress."

---

##### Step 9: Activation Covariance Analysis (`activation_covariance_analysis.py`)

**Question:** *Does the intervention change the shape of the activation distribution?*

Computes the eigenvalue spectrum of the activation covariance matrix at each layer and measures:

| Metric | What it captures |
|---|---|
| **Effective rank** (of covariance) | How many dimensions do activations meaningfully occupy? |
| **Spectral entropy** | How uniform is the energy distribution across dimensions? |
| **Wasserstein distance** | How much did the spectrum change between base and modified model? |

A key output is the **selectivity ratio**: (Wasserstein distance on forget text) / (Wasserstein distance on retain text). High selectivity = the intervention specifically reshapes forget-domain representations while leaving retain-domain representations intact.

**Why this matters:** This captures something the norms miss — two distributions can have identical norms but completely different *shapes*. If filtering fundamentally restructures the covariance (high Wasserstein, changed effective rank) while unlearning barely disturbs it, that's evidence the representations aren't actually changing.

---

##### Step 11: Row Space Projection (`row_space_projection_analysis.py`)

**Question:** *Do activations from forget-text align more with the directions the intervention modified?*

Computes the SVD of ΔW at each MLP layer and measures how much the *input activations* project onto the row space (input-side principal directions) of ΔW.

If forget-text activations have high projection onto ΔW's row space while retain-text activations don't, the update is *precisely targeted* — it modifies exactly the directions that forget-text activates. The **selectivity ratio** quantifies this.

**Why this matters:** This is perhaps the most mechanistically informative diagnostic. It directly tests whether the intervention is *geometrically aligned* with the specific input patterns it needs to suppress. High selectivity + low rank = a surgical intervention that only fires on forget-domain inputs. Low selectivity = a blunt instrument that affects everything equally. And crucially, high selectivity + low rank is also the easiest to undo: just learn a small correction in that same low-dimensional subspace.

---

##### Step 12: Local Lipschitz Analysis (`local_lipschitzness_analysis.py`)

**Question:** *Did the intervention make the model's output more or less sensitive to input perturbations?*

Estimates the local Lipschitz constant by perturbing input embeddings with small noise (ε-balls) and measuring how much the output changes. Also computes gradient norms and output variance under perturbation, separately for forget and retain texts.

| Outcome | Interpretation |
|---|---|
| Forget text becomes **rougher** (higher Lipschitz) | Model is unstable on forget inputs — outputs shift erratically |
| Forget text becomes **smoother** (lower Lipschitz) | Model learned to ignore/suppress forget-domain features |
| Retain text stays **similar** | Intervention didn't damage general capabilities |

**Why this matters:** A model that becomes rougher on forget text hasn't *learned to not know* something — it's in an unstable regime where small pushes (fine-tuning) can tip it back. Smoothness changes are a direct indicator of whether the loss landscape around forget-domain inputs is fundamentally reshaped or just locally perturbed.

---

#### Knowledge Localization

Unlike Steps 1–12 which compare pairs of models, Step 13 analyzes **each model individually**.

##### Step 13: Linear Probe Analysis (`linear_probe_analysis.py`)

**Question:** *At which layer is the forget-set knowledge most linearly readable?*

For each layer, this script extracts the **last-token hidden state** (the most context-rich position in a causal LM) on both forget and retain texts, then trains a logistic regression "probe" to classify whether an activation came from forget or retain text.

| Metric | What it tells you |
|---|---|
| **Probe accuracy** | How well a linear classifier can distinguish forget from retain activations at this layer |
| **Selectivity** | Accuracy minus majority-class baseline — how much the probe exceeds random guessing. High selectivity = the layer linearly encodes the forget/retain distinction. |
| **AUC** (Area Under the ROC Curve) | How well the probe ranks forget vs retain samples, regardless of threshold. 0.5 = random, 1.0 = perfect separation. More robust than accuracy when class sizes are imbalanced. |

Default probe: `LogisticRegression(C=1.0, max_iter=1000)` — adjustable via `--C` and `--max-iter`.

**Why this matters:** This identifies *where* in the network the model stores information that distinguishes hazardous content from benign content. In a well-unlearned model, you'd expect low selectivity everywhere — the model genuinely can't tell the domains apart. In a poorly unlearned model, the probes will still find layers with high selectivity, meaning the knowledge is still encoded and a linear readout can recover it. Comparing probe profiles across BASE, FILTERED, and UNLEARNED reveals whether unlearning actually erased the representation or just hid it from the output head.

> **Note:** Unlike other steps, results are stored **per-model** (not per-comparison) since probes analyze a single model's representations.

---

### The Big Picture

The diagnostics answer an escalating series of questions:

| Level | Question | Steps |
|---|---|---|
| **Magnitude** | How much changed? | 1–2 |
| **Location** | Where — MLP or Attention? Which layers? | 6 |
| **Geometry** | What shape is ΔW? Low-rank? Nullspace-aligned? | 7, 10 |
| **Function** | Do activations actually change on target text? | 4–5, 8–9 |
| **Precision** | Is the change *targeted* at forget-domain inputs? | 11 |
| **Stability** | Is the new behavior robust or fragile? | 12 |
| **Knowledge Localization** | Where is forget-set knowledge encoded? | 13 |

The thesis prediction is that unlearning methods (CB-LAT) will show: small magnitude, attention-localized, low-rank, nullspace-aligned, minimal activation change, low selectivity, and increased roughness — the full mechanistic signature of a brittle intervention. While filtering will show the opposite across every dimension.

---

### Output Structure

All results are saved under a single root (default `outputs/`):

```
outputs/
  <comparison>/                        # Steps 1–12: per model-pair
    param_stats/           per_matrix.csv, per_layer.csv
    param_plots/           Layer locality, stable rank, rank comparison PNGs
    activation_stats/      activation_stats.csv
    activation_plots/      Activation norms, activation diffs PNGs
    mlp_attn_analysis/     summary CSV + plots
    null_space_analysis/   null_space_results.csv + plots
    activation_separation/ separation metrics + plots
    activation_covariance/ covariance spectra + plots
    mlp_nullspace/         alignment metrics + plots
    row_space_projection/  projection metrics + plots
    lipschitzness/         Lipschitz estimates + plots

  <model>/                             # Step 13: per individual model
    linear_probes/         probe_results.csv, summary.json + plot
```

> **Tip:** The pipeline automatically skips steps whose output already exists. Use `./pipeline.sh --force` to regenerate everything.

---

## Unlearning

```bash
# Run unlearning
./create_all_unlearning_models.sh

# Analyze the result(s) per the above experiments
uv run collect_param_stats.py \
  --model-a EleutherAI/deep-ignorance-unfiltered \
  --model-b outputs/EleutherAI_deep-ignorance-unfiltered__ga/unlearned_model \
  --outdir outputs/ga_analysis/param_stats
```

### Available Methods

| Method | Type | Description |
|---|---|---|
| `ga` / `ga_simple` | Loss | Gradient ascent on forget set |
| `grad_diff` | Loss | Weighted forget/retain NLL difference |
| `dpo` / `npo` / `simnpo` | Loss | Preference optimization variants |
| `rmu` | Representation | Steer hidden states toward random targets |
| `cb` | Representation | Cosine-similarity circuit rerouting |
| `lat` | Representation | Latent adversarial training for robustness |
| `cb_lat` | Representation | Circuit Breakers + LAT combined |

See `uv run unlearn.py --help` for full argument reference.

---

## Inference

Run prompts against any model — HuggingFace or local unlearned:

```bash
# Single prompt
uv run infer.py --model EleutherAI/deep-ignorance-unfiltered --prompt "What is biotin?"

# Local unlearned model
uv run infer.py --model unlearned_models/EleutherAI_deep-ignorance-unfiltered__ga --prompt "What is biotin?"

# Side-by-side comparison
uv run infer.py \
  --model EleutherAI/deep-ignorance-unfiltered \
  --model-b EleutherAI/deep-ignorance-unfiltered-cb-lat \
  --prompt "What is biotin?"

# Interactive mode
uv run infer.py --model EleutherAI/deep-ignorance-unfiltered --interactive
```

### Sweep Mode

Run the same prompt through **every model** in `unlearned_models/` and save a comparison CSV:

```bash
# All local unlearned models
uv run infer.py --sweep --prompt "What is biotin?"

# Include the 3 base HF models (unfiltered, filtered, cb-lat)
uv run infer.py --sweep --include-base --prompt "What is biotin?"

# Custom models directory
uv run infer.py --sweep --models-dir path/to/models --prompt "What is biotin?"
```

Output is saved to `outputs/inference/<sha256_hash>.csv` with columns: `prompt`, `model`, `model_path`, `output`. The same prompt always maps to the same filename.

Options: `--max-new-tokens`, `--temperature`, `--top-p`, `--greedy`, `--models-dir`, `--outdir`. See `uv run infer.py --help`.

---

## Appendix A: CSV Column Reference

### `param_stats/per_matrix.csv`

One row per weight matrix in the model.

| Column | Description |
| :--- | :--- |
| `name` | Full parameter name (e.g., `gpt_neox.layers.10.mlp.dense_h_to_4h.weight`) |
| `layer` | Integer layer index extracted from name (`-1` if not layer-specific) |
| `group` | Coarse grouping: `attn` (attention) or `mlp` (feed-forward) |
| `shape0` | Matrix rows (output features) |
| `shape1` | Matrix columns (input features) |
| `dW_fro` | Frobenius norm of the weight difference: ‖ΔW‖_F |
| `W_fro` | Frobenius norm of the original (base) weight: ‖W‖_F |
| `dW_fro_rel` | Relative Frobenius norm: ‖ΔW‖_F / ‖W‖_F (fraction of original weight changed) |
| `dW_spectral` | Spectral norm (largest singular value) of ΔW: σ₁(ΔW) |
| `W_spectral` | Spectral norm of the original (base) weight: σ₁(W) |
| `dW_spectral_rel` | Relative spectral norm: σ₁(ΔW) / σ₁(W) |
| `dW_stable_rank` | Stable rank of ΔW: ‖ΔW‖²_F / ‖ΔW‖²₂ |
| `W_stable_rank` | Stable rank of the original (base) weights |
| `dW_empirical_rank`* | Number of singular values of ΔW capturing 99% of variance |
| `W_empirical_rank`* | Number of singular values of W capturing 99% of variance |

\* *Only present when `--empirical-rank` flag is passed (opt-in, requires full SVD).*

### `param_stats/per_layer.csv`

Aggregated statistics per (layer, group) pair.

| Column | Description |
| :--- | :--- |
| `layer` | Integer layer index |
| `group` | `attn` or `mlp` |
| `dW_fro_layer` | Root-sum-square of Frobenius norms in this group: √(Σ ‖ΔWᵢ‖²_F) |
| `W_fro_layer` | Root-sum-square of original weight Frobenius norms: √(Σ ‖Wᵢ‖²_F) |
| `dW_fro_layer_rel` | Relative change: `dW_fro_layer / W_fro_layer` |
| `max_dW_spectral` | Max spectral norm of ΔW across matrices in this group |
| `max_W_spectral` | Max spectral norm of W across matrices in this group |
| `max_dW_spectral_rel` | Relative spectral norm: `max_dW_spectral / max_W_spectral` |
| `mean_dW_stable_rank` | Mean stable rank of ΔW across matrices in this group |
| `mean_dW_empirical_rank`* | Mean empirical rank of ΔW across matrices in this group |
| `count_mats` | Number of weight matrices aggregated in this group |

\* *Only present when `--empirical-rank` flag is passed.*

### `activation_stats/activation_stats.csv`

One row per (layer, split) combination.

| Column | Description |
| :--- | :--- |
| `layer` | Layer index (0 to N) |
| `split` | Dataset split: `forget` or `retain` |
| `model_a_norm_L1` | Mean L1 norm of hidden states for model A (baseline): E[‖h‖₁] |
| `model_a_norm_L2` | Mean L2 norm of hidden states for model A (baseline): E[‖h‖₂] |
| `model_b_norm_L1` | Mean L1 norm of hidden states for model B (target) |
| `model_b_norm_L2` | Mean L2 norm of hidden states for model B (target) |
| `mean_dh_L1` | Mean L1 norm of the activation difference: E[‖Δh‖₁] |
| `mean_dh_L2` | Mean L2 norm of the activation difference: E[‖Δh‖₂] |
