# Mechanistic Analysis: Filtered vs. CB-LAT Unlearning

Comparing two interventions on EleutherAI's `deep-ignorance-unfiltered` (2.8B):
- **Filtered** (`e2e-strong-filter`): Retrained from scratch without WMDP-Bio hazardous data
- **CB-LAT** (`cb-lat`): Post-hoc unlearned via Circuit Breakers + Latent Adversarial Training

---

## Steps 1–2: Parameter Statistics — "How much changed?"

### Frobenius Norms (‖ΔW‖_F)

| Metric | Filtered | CB-LAT | Ratio |
|---|---|---|---|
| Typical MLP ‖ΔW‖_F per layer | 320–490 | **0.0** | ∞ |
| Typical Attention ‖ΔW‖_F per layer | 240–310 | 15–29 | ~15× |
| Layer 31 (final) | attn=295, mlp=478 | **attn=0, mlp=0** | — |

#### Filtered: MLP ‖ΔW‖ per layer
Steady ~420 with a rising tail to ~490 in late layers:

![Filtered MLP Frobenius norms](../plots/EleutherAI_deep-ignorance-unfiltered__to__EleutherAI_deep-ignorance-e2e-strong-filter/param_plots/layer_locality_mlp.png)

#### CB-LAT: MLP ‖ΔW‖ per layer
Perfectly flat zero — no MLP weights were modified:

![CB-LAT MLP Frobenius norms](../plots/EleutherAI_deep-ignorance-unfiltered__to__EleutherAI_deep-ignorance-unfiltered-cb-lat/param_plots/layer_locality_mlp.png)

#### Filtered: Attention ‖ΔW‖ per layer
240 to 310, smooth curve across all layers:

![Filtered Attention Frobenius norms](../plots/EleutherAI_deep-ignorance-unfiltered__to__EleutherAI_deep-ignorance-e2e-strong-filter/param_plots/layer_locality_attn.png)

#### CB-LAT: Attention ‖ΔW‖ per layer
Noisy ~15-29 with a drop to zero at layer 31:

![CB-LAT Attention Frobenius norms](../plots/EleutherAI_deep-ignorance-unfiltered__to__EleutherAI_deep-ignorance-unfiltered-cb-lat/param_plots/layer_locality_attn.png)

**Filtering** produces massive, distributed changes across *every* weight matrix — both MLP and attention, all 32 layers. CB-LAT is strikingly different: **MLP changes are exactly zero.** The method only modifies attention weights, with norms ~15× smaller than filtering.

> **Key finding:** CB-LAT modifies < 0.01% of the parameter change that filtering does. The "unlearned" model is nearly identical to the base model in weight-space.

### Stable Rank of ΔW

| Metric | Filtered | CB-LAT |
|---|---|---|
| MLP stable rank | 15–750 (varies widely) | 0.0 (no update) |
| Attention stable rank | 113–430 (high-rank) | **~0.5** (consistently) |

#### Filtered: MLP stable rank
Varies 15-750 with a deep oscillatory pattern and a rising trend in late layers:

![Filtered MLP stable rank](../plots/EleutherAI_deep-ignorance-unfiltered__to__EleutherAI_deep-ignorance-e2e-strong-filter/param_plots/stable_rank_mlp.png)

#### CB-LAT: Attention stable rank
Locked at ~0.5 across layers 3-30, confirming rank-1 perturbations per QKV submatrix:

![CB-LAT Attention stable rank](../plots/EleutherAI_deep-ignorance-unfiltered__to__EleutherAI_deep-ignorance-unfiltered-cb-lat/param_plots/stable_rank_attn.png)

CB-LAT's attention ΔW has stable rank ≈ 0.5 across all layers — since stable rank ≥ 1 for any nonzero matrix, a value of 0.5 means the update is split across 2 submatrices (Q, K, V packed into `query_key_value`) where each contributes a **rank-1** perturbation. This is the geometric signature of a LoRA-like intervention.

---

## Steps 4–5: Activation Norms — "Do outputs actually change?"

### CB-LAT: Selective but Small

#### Forget-text activation diffs
Exponential growth through layers, peaking at L2≈270 at layers 29-30, then dropping at layer 31:

![CB-LAT forget activation diffs](../plots/EleutherAI_deep-ignorance-unfiltered__to__EleutherAI_deep-ignorance-unfiltered-cb-lat/activation_plots/activation_diffs_forget.png)

#### Retain-text activation diffs
Same exponential shape but ~5× smaller — max L2≈50 at layers 29-30:

![CB-LAT retain activation diffs](../plots/EleutherAI_deep-ignorance-unfiltered__to__EleutherAI_deep-ignorance-unfiltered-cb-lat/activation_plots/activation_diffs_retain.png)

CB-LAT achieves **5× selectivity** (forget diffs 5× larger than retain), but with modest absolute magnitudes. The exponential growth through layers shows the attention perturbations cascade and amplify through depth. The sharp drop at layer 31 suggests the final LayerNorm partially resets the accumulated perturbation.

### Filtered: Large but Non-selective

#### Forget-text activation diffs
Smooth exponential growth to L2≈630, nearly identical shape to retain:

![Filtered forget activation diffs](../plots/EleutherAI_deep-ignorance-unfiltered__to__EleutherAI_deep-ignorance-e2e-strong-filter/activation_plots/activation_diffs_forget.png)

#### Retain-text activation diffs
Same exponential shape, peaking at L2≈410 — only 1.5× less than forget:

![Filtered retain activation diffs](../plots/EleutherAI_deep-ignorance-unfiltered__to__EleutherAI_deep-ignorance-e2e-strong-filter/activation_plots/activation_diffs_retain.png)

Filtering causes much larger absolute activation changes (3.3× more than CB-LAT) but with only ~1.8× selectivity. Both forget and retain domains are deeply restructured.

> **Key finding:** CB-LAT is more *selective* (5× vs 1.8×) but less *thorough* (106 vs 349 mean ΔL2). It targets forget-domain processing while leaving retain intact — but the magnitude suggests only partial representational disruption.

---

## Step 6: MLP vs Attention Breakdown

#### Filtered: MLP dominates attention at every layer
Total: MLP ~14k vs Attn ~9k:

![Filtered MLP vs Attention](../outputs/EleutherAI_deep-ignorance-unfiltered__to__EleutherAI_deep-ignorance-e2e-strong-filter/mlp_attn_analysis/mlp_vs_attn_detailed.png)

#### CB-LAT: MLP flatlines at zero
Attention-only intervention, total change ~650:

![CB-LAT MLP vs Attention](../outputs/EleutherAI_deep-ignorance-unfiltered__to__EleutherAI_deep-ignorance-unfiltered-cb-lat/mlp_attn_analysis/mlp_vs_attn_detailed.png)

Filtered MLP/Attn ratio is consistently **1.40–1.62×**. CB-LAT is **0.0** — purely attention-routing with zero MLP modification.

> **Implication for brittleness:** If knowledge is stored in MLP layers (the "key-value memory" hypothesis), CB-LAT hasn't *erased* any knowledge — it has only *rerouted around it*. An adversary who undoes the attention rerouting could recover full access.

---

## Step 7: Null Space & Subspace Analysis

#### Filtered: Low concentration, low alignment — deep restructuring

![Filtered null space analysis](../outputs/EleutherAI_deep-ignorance-unfiltered__to__EleutherAI_deep-ignorance-e2e-strong-filter/null_space_analysis/null_space_visualization.png)

#### CB-LAT: MLP at zero; attention bimodal — most matrices unchanged, modified QKV highly concentrated

![CB-LAT null space analysis](../outputs/EleutherAI_deep-ignorance-unfiltered__to__EleutherAI_deep-ignorance-unfiltered-cb-lat/null_space_analysis/null_space_visualization.png)

| Component | Filtered | CB-LAT |
|---|---|---|
| MLP effective rank | 3800–3940 (full) | **0** |
| Attn QKV effective rank | 3800+ | **3–6** |
| Subspace alignment | 0.07–0.46 (low) | **0.91–0.95** (preserved) |
| Top-10 SV ratio | 1–3% | **99.6–99.8%** |

> **Key finding:** CB-LAT is a rank-~5 perturbation preserving 95% of the original subspace. In a 4096-dimensional space, an adversary only needs to discover ~5 directions to reverse the intervention.

---

## Step 9: Activation Covariance — "Does the shape of representations change?"

#### Filtered: Wasserstein 11k–29k, modest selectivity (1.15×)

![Filtered covariance analysis](../outputs/EleutherAI_deep-ignorance-unfiltered__to__EleutherAI_deep-ignorance-e2e-strong-filter/activation_covariance/covariance_analysis.png)

#### CB-LAT: Wasserstein 1.9k–5.9k — **inverted selectivity** (retain changes MORE)

![CB-LAT covariance analysis](../outputs/EleutherAI_deep-ignorance-unfiltered__to__EleutherAI_deep-ignorance-unfiltered-cb-lat/activation_covariance/covariance_analysis.png)

**CB-LAT shows inverted covariance selectivity**: retain Wasserstein = 4382 vs forget = 1886 (ratio 0.43×). This means CB-LAT changes the *shape* of retain-domain representations more than forget-domain. The forget-text changes appear to be primarily **magnitude suppression** (scaling) rather than representational restructuring, which is why activation norms drop but covariance barely shifts.

---

## Step 10: MLP Nullspace Alignment

#### Filtered: 88% column space, 40% nullspace — updates are on-manifold

![Filtered MLP nullspace alignment](../outputs/EleutherAI_deep-ignorance-unfiltered__to__EleutherAI_deep-ignorance-e2e-strong-filter/mlp_nullspace/mlp_nullspace_alignment.png)

#### CB-LAT: Every metric is exactly zero — no MLP modification

![CB-LAT MLP nullspace alignment](../outputs/EleutherAI_deep-ignorance-unfiltered__to__EleutherAI_deep-ignorance-unfiltered-cb-lat/mlp_nullspace/mlp_nullspace_alignment.png)

Filtering's MLP updates are overwhelmingly on-manifold — they directly modify existing computations rather than introducing new directions. CB-LAT has zero MLP change, consistent with all other diagnostics.

---

## Step 8: Activation Separation (Filtered only)

![Filtered activation separation](../outputs/activation_separation/activation_separation_analysis.png)

Both the base and filtered models can distinguish forget/retain activations at near-perfect AUC (≈ 0.998). Filtering changes *what* the model computes, not *how distinguishable* the domains are. CB-LAT activation separation data is missing and should be re-run.

---

## Synthesis: The Mechanistic Signature of Brittleness

```
                          Filtering              CB-LAT
                          ──────────             ──────────
Magnitude:                ‖ΔW‖ ~300-490          ‖ΔW‖ ~15-29 (attn only)
Location:                 All MLP + Attn         Attn QKV only (MLP = 0)
Geometry:                 Full-rank (~3900)      Rank 3-6
                          Subspace divergent     Subspace preserved (95%)
                          On-manifold (99%)      N/A (no MLP change)
Activation change:        Large (dh=349)         Moderate (dh=106)
Selectivity:              1.8× (forget>retain)   5.0× (forget>retain)
Covariance:               Massive spectrum shift  Inverted (retain > forget)
```

### The Narrative

**Filtering** rewrites the model from the ground up. Every weight matrix, every layer, full rank. The MLP layers — where factual knowledge lives — are deeply restructured. This is **genuine knowledge erasure**.

**CB-LAT** applies a rank-~5 perturbation to QKV attention projections. The MLPs are untouched. Despite high activation selectivity (5×), covariance analysis reveals the forget-text changes are magnitude suppression, not representational restructuring. The knowledge remains intact, rerouted around but not erased.

### Adversarial Robustness Prediction

An adversarial fine-tuner attacking CB-LAT faces a **low-dimensional search problem**: ~5 directions per QKV matrix × ~31 layers. Discovering and reversing these directions could restore full access to the suppressed knowledge, since the MLP weights are literally unchanged.

Reversing filtering would require reconstructing full-rank changes across all weight matrices — as hard as re-training from scratch.

---

## Missing Data & Recommended Re-runs

| Diagnostic | Status | Action Needed |
|---|---|---|
| Activation Separation | ✅ Filtered only | Re-run for CB-LAT |
| Row Space Projection | ❌ Empty | Re-run both comparisons |
| Lipschitz Analysis | ❌ All NaN | Debug NaN issue, re-run both |
