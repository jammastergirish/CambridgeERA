#!/usr/bin/env python
# /// script
# dependencies = [
#   "torch",
#   "transformers",
#   "numpy",
#   "matplotlib",
#   "tqdm",
#   "wandb",
#   "pandas",
# ]
# ///

"""
Analyze local Lipschitzness of models on forget/retain data.
Measures how "smooth" the function is locally by estimating gradient norms.
Lower Lipschitz constant = smoother function in that region.
"""

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import resolve_device, resolve_dtype, write_csv, init_wandb, log_csv_as_table, log_plots, finish_wandb


def estimate_local_lipschitz(model, tokenizer, texts, device, dtype,
                            epsilon=0.01, num_samples=5, max_length=512):
    """
    Estimate local Lipschitz constant by perturbing inputs and measuring output changes.

    Uses finite differences in embedding space to estimate:
    L = max ||f(x + δ) - f(x)|| / ||δ||
    """
    model.eval()
    lipschitz_estimates = []

    for text in tqdm(texts[:100], desc="Estimating Lipschitz constants", unit="text", leave=False):  # Sample subset
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=max_length, padding=True).to(device)

        # Get embeddings
        with torch.no_grad():
            if hasattr(model, 'model'):  # LLaMA-style
                embeddings = model.model.embed_tokens(inputs.input_ids)
            elif hasattr(model, 'transformer'):  # GPT-style
                embeddings = model.transformer.wte(inputs.input_ids)
            else:
                continue

            # Get baseline output
            outputs_base = model(inputs_embeds=embeddings, output_hidden_states=True)
            hidden_base = outputs_base.hidden_states[-1]  # Last layer
            logits_base = outputs_base.logits

        # Estimate Lipschitz constant with random perturbations
        max_ratio = 0.0
        for _ in range(num_samples):
            # Generate random perturbation
            noise = torch.randn_like(embeddings) * epsilon
            noise_norm = noise.norm().item()

            # Get perturbed output
            with torch.no_grad():
                outputs_pert = model(inputs_embeds=embeddings + noise,
                                    output_hidden_states=True)
                hidden_pert = outputs_pert.hidden_states[-1]
                logits_pert = outputs_pert.logits

            # Compute output difference
            hidden_diff = (hidden_pert - hidden_base).norm().item()
            logits_diff = (logits_pert - logits_base).norm().item()

            # Estimate local Lipschitz constant
            if noise_norm > 0:
                ratio_hidden = hidden_diff / noise_norm
                ratio_logits = logits_diff / noise_norm
                max_ratio = max(max_ratio, ratio_hidden, ratio_logits)

        lipschitz_estimates.append(max_ratio)

    return lipschitz_estimates


def compute_gradient_norms(model, tokenizer, texts, device, dtype, max_length=512):
    """
    Compute actual gradient norms with respect to inputs.
    This is a more direct measure of local smoothness.
    """
    model.eval()
    gradient_norms = []

    for text in tqdm(texts[:50], desc="Computing gradient norms", unit="text", leave=False):  # Smaller sample
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=max_length, padding=True).to(device)

        # Get embeddings and enable gradients
        if hasattr(model, 'model'):  # LLaMA-style
            embeddings = model.model.embed_tokens(inputs.input_ids)
        elif hasattr(model, 'transformer'):  # GPT-style
            embeddings = model.transformer.wte(inputs.input_ids)
        else:
            continue

        embeddings = embeddings.detach().requires_grad_(True)

        # Forward pass
        outputs = model(inputs_embeds=embeddings)
        loss = outputs.logits.mean()  # Simple scalar objective

        # Compute gradients
        loss.backward()

        if embeddings.grad is not None:
            grad_norm = embeddings.grad.norm().item()
            gradient_norms.append(grad_norm)

        # Clear gradients
        model.zero_grad()

    return gradient_norms


def analyze_output_variance(model, tokenizer, texts, device, dtype,
                           num_perturbations=10, epsilon=0.01, max_length=512):
    """
    Analyze variance in outputs under small input perturbations.
    High variance = less smooth/stable.
    """
    model.eval()
    variances = []

    for text in tqdm(texts[:50], desc="Measuring output variance", unit="text", leave=False):
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=max_length, padding=True).to(device)

        # Get embeddings
        with torch.no_grad():
            if hasattr(model, 'model'):
                embeddings = model.model.embed_tokens(inputs.input_ids)
            elif hasattr(model, 'transformer'):
                embeddings = model.transformer.wte(inputs.input_ids)
            else:
                continue

            # Collect outputs under perturbations
            outputs_list = []
            for _ in range(num_perturbations):
                noise = torch.randn_like(embeddings) * epsilon
                outputs = model(inputs_embeds=embeddings + noise)
                outputs_list.append(outputs.logits.cpu().float().numpy())

        # Compute variance across perturbations
        outputs_array = np.stack(outputs_list)
        variance = np.var(outputs_array, axis=0).mean()
        variances.append(variance)

    return variances


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-a", required=True, help="Baseline model")
    ap.add_argument("--model-b", required=True, help="Unlearned/finetuned model")
    ap.add_argument("--forget-text", default="data/forget.txt")
    ap.add_argument("--retain-text", default="data/retain.txt")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--epsilon", type=float, default=0.01,
                   help="Perturbation magnitude for Lipschitz estimation")
    ap.add_argument("--num-perturbations", type=int, default=10,
                   help="Number of perturbations for variance estimation")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--max-samples", type=int, default=500,
                   help="Max texts per split to process (default: 500)")
    ap.add_argument("--outdir", default="outputs/lipschitzness_analysis")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    init_wandb("local_lipschitzness", args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    # Load texts
    print("[local_lipschitz] Loading forget/retain texts...")
    with open(args.forget_text, "r") as f:
        forget_texts = [line.strip() for line in f if line.strip()]
    with open(args.retain_text, "r") as f:
        retain_texts = [line.strip() for line in f if line.strip()]

    # Cap sample count
    if len(forget_texts) > args.max_samples:
        forget_texts = forget_texts[:args.max_samples]
    if len(retain_texts) > args.max_samples:
        retain_texts = retain_texts[:args.max_samples]

    print(f"[local_lipschitz] Loaded {len(forget_texts)} forget texts, {len(retain_texts)} retain texts (max-samples={args.max_samples})")

    # Load models
    print(f"[local_lipschitz] Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_a)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[local_lipschitz] Loading model A (baseline): {args.model_a}")
    model_a = AutoModelForCausalLM.from_pretrained(args.model_a, torch_dtype=dtype).to(device)

    print(f"[local_lipschitz] Loading model B (target): {args.model_b}")
    model_b = AutoModelForCausalLM.from_pretrained(args.model_b, torch_dtype=dtype).to(device)

    results = {}

    # Analyze Model A
    print("\n[local_lipschitz] === Analyzing Model A (baseline) ===")
    print("  Forget data:")
    forget_lip_a = estimate_local_lipschitz(model_a, tokenizer, forget_texts,
                                           device, dtype, args.epsilon)
    forget_grad_a = compute_gradient_norms(model_a, tokenizer, forget_texts,
                                          device, dtype, args.max_length)
    forget_var_a = analyze_output_variance(model_a, tokenizer, forget_texts,
                                          device, dtype, args.num_perturbations,
                                          args.epsilon, args.max_length)

    print("  Retain data:")
    retain_lip_a = estimate_local_lipschitz(model_a, tokenizer, retain_texts,
                                           device, dtype, args.epsilon)
    retain_grad_a = compute_gradient_norms(model_a, tokenizer, retain_texts,
                                          device, dtype, args.max_length)
    retain_var_a = analyze_output_variance(model_a, tokenizer, retain_texts,
                                          device, dtype, args.num_perturbations,
                                          args.epsilon, args.max_length)

    # Analyze Model B
    print("\n[local_lipschitz] === Analyzing Model B (target) ===")
    print("  Forget data:")
    forget_lip_b = estimate_local_lipschitz(model_b, tokenizer, forget_texts,
                                           device, dtype, args.epsilon)
    forget_grad_b = compute_gradient_norms(model_b, tokenizer, forget_texts,
                                          device, dtype, args.max_length)
    forget_var_b = analyze_output_variance(model_b, tokenizer, forget_texts,
                                          device, dtype, args.num_perturbations,
                                          args.epsilon, args.max_length)

    print("  Retain data:")
    retain_lip_b = estimate_local_lipschitz(model_b, tokenizer, retain_texts,
                                           device, dtype, args.epsilon)
    retain_grad_b = compute_gradient_norms(model_b, tokenizer, retain_texts,
                                          device, dtype, args.max_length)
    retain_var_b = analyze_output_variance(model_b, tokenizer, retain_texts,
                                          device, dtype, args.num_perturbations,
                                          args.epsilon, args.max_length)

    # Save results
    os.makedirs(args.outdir, exist_ok=True)

    # Compute statistics
    results_summary = [
        {
            "model": "A",
            "data": "forget",
            "avg_lipschitz": np.mean(forget_lip_a),
            "std_lipschitz": np.std(forget_lip_a),
            "avg_gradient_norm": np.mean(forget_grad_a),
            "std_gradient_norm": np.std(forget_grad_a),
            "avg_output_variance": np.mean(forget_var_a),
            "std_output_variance": np.std(forget_var_a),
        },
        {
            "model": "A",
            "data": "retain",
            "avg_lipschitz": np.mean(retain_lip_a),
            "std_lipschitz": np.std(retain_lip_a),
            "avg_gradient_norm": np.mean(retain_grad_a),
            "std_gradient_norm": np.std(retain_grad_a),
            "avg_output_variance": np.mean(retain_var_a),
            "std_output_variance": np.std(retain_var_a),
        },
        {
            "model": "B",
            "data": "forget",
            "avg_lipschitz": np.mean(forget_lip_b),
            "std_lipschitz": np.std(forget_lip_b),
            "avg_gradient_norm": np.mean(forget_grad_b),
            "std_gradient_norm": np.std(forget_grad_b),
            "avg_output_variance": np.mean(forget_var_b),
            "std_output_variance": np.std(forget_var_b),
        },
        {
            "model": "B",
            "data": "retain",
            "avg_lipschitz": np.mean(retain_lip_b),
            "std_lipschitz": np.std(retain_lip_b),
            "avg_gradient_norm": np.mean(retain_grad_b),
            "std_gradient_norm": np.std(retain_grad_b),
            "avg_output_variance": np.mean(retain_var_b),
            "std_output_variance": np.std(retain_var_b),
        },
    ]

    write_csv(
        os.path.join(args.outdir, "lipschitzness_summary.csv"),
        results_summary,
        ["model", "data", "avg_lipschitz", "std_lipschitz",
         "avg_gradient_norm", "std_gradient_norm",
         "avg_output_variance", "std_output_variance"]
    )

    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: Lipschitz constants comparison
    ax = axes[0, 0]
    x = np.array([0, 1, 3, 4])
    means = [np.mean(forget_lip_a), np.mean(retain_lip_a),
             np.mean(forget_lip_b), np.mean(retain_lip_b)]
    stds = [np.std(forget_lip_a), np.std(retain_lip_a),
            np.std(forget_lip_b), np.std(retain_lip_b)]
    colors = ['red', 'blue', 'darkred', 'darkblue']
    labels = ['Forget (A)', 'Retain (A)', 'Forget (B)', 'Retain (B)']

    ax.bar(x, means, yerr=stds, color=colors, alpha=0.6, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel("Lipschitz Constant")
    ax.set_title("Local Lipschitzness (Lower = Smoother)")
    ax.grid(alpha=0.3, axis='y')

    # Plot 2: Gradient norms
    ax = axes[0, 1]
    means_grad = [np.mean(forget_grad_a), np.mean(retain_grad_a),
                  np.mean(forget_grad_b), np.mean(retain_grad_b)]
    stds_grad = [np.std(forget_grad_a), np.std(retain_grad_a),
                 np.std(forget_grad_b), np.std(retain_grad_b)]

    ax.bar(x, means_grad, yerr=stds_grad, color=colors, alpha=0.6, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Input Gradient Norms")
    ax.grid(alpha=0.3, axis='y')

    # Plot 3: Output variance
    ax = axes[0, 2]
    means_var = [np.mean(forget_var_a), np.mean(retain_var_a),
                 np.mean(forget_var_b), np.mean(retain_var_b)]
    stds_var = [np.std(forget_var_a), np.std(retain_var_a),
                np.std(forget_var_b), np.std(retain_var_b)]

    ax.bar(x, means_var, yerr=stds_var, color=colors, alpha=0.6, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel("Output Variance")
    ax.set_title("Variance Under Perturbation")
    ax.grid(alpha=0.3, axis='y')

    # Plot 4: Change in Lipschitzness
    ax = axes[1, 0]
    forget_lip_change = np.mean(forget_lip_b) - np.mean(forget_lip_a)
    retain_lip_change = np.mean(retain_lip_b) - np.mean(retain_lip_a)

    ax.bar([0, 1], [forget_lip_change, retain_lip_change],
           color=['red', 'blue'], alpha=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Forget', 'Retain'])
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel("Δ Lipschitz (B - A)")
    ax.set_title("Change in Smoothness")
    ax.grid(alpha=0.3, axis='y')

    # Plot 5: Distribution comparison
    ax = axes[1, 1]
    ax.hist(forget_lip_a, bins=20, alpha=0.3, label='Forget (A)', color='red')
    ax.hist(forget_lip_b, bins=20, alpha=0.3, label='Forget (B)', color='darkred')
    ax.hist(retain_lip_a, bins=20, alpha=0.3, label='Retain (A)', color='blue')
    ax.hist(retain_lip_b, bins=20, alpha=0.3, label='Retain (B)', color='darkblue')
    ax.set_xlabel("Lipschitz Constant")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Local Lipschitz Constants")
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 6: Summary
    ax = axes[1, 2]
    ax.axis('off')

    # Compute relative changes
    forget_lip_ratio = np.mean(forget_lip_b) / (np.mean(forget_lip_a) + 1e-10)
    retain_lip_ratio = np.mean(retain_lip_b) / (np.mean(retain_lip_a) + 1e-10)

    summary_text = f"""
    Lipschitzness Analysis Summary:

    Model A (Baseline):
    - Forget Lipschitz: {np.mean(forget_lip_a):.3f} ± {np.std(forget_lip_a):.3f}
    - Retain Lipschitz: {np.mean(retain_lip_a):.3f} ± {np.std(retain_lip_a):.3f}

    Model B (Unlearned):
    - Forget Lipschitz: {np.mean(forget_lip_b):.3f} ± {np.std(forget_lip_b):.3f}
    - Retain Lipschitz: {np.mean(retain_lip_b):.3f} ± {np.std(retain_lip_b):.3f}

    Changes (B/A ratio):
    - Forget: {forget_lip_ratio:.2f}x
    - Retain: {retain_lip_ratio:.2f}x

    Interpretation:
    Forget became {'smoother' if forget_lip_ratio < 0.9 else 'rougher' if forget_lip_ratio > 1.1 else 'similar'}
    Retain became {'smoother' if retain_lip_ratio < 0.9 else 'rougher' if retain_lip_ratio > 1.1 else 'similar'}
    {'✓ Selective smoothing on forget' if forget_lip_ratio < 0.9 and retain_lip_ratio > 0.95 else '✗ Non-selective changes'}
    """

    ax.text(0.05, 0.5, summary_text, fontsize=9, family='monospace',
            verticalalignment='center')

    plt.suptitle(f"Local Lipschitzness Analysis\n{args.model_a} → {args.model_b}")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "lipschitzness_analysis.png"), dpi=150)
    plt.close()

    # Save detailed results
    np.savez_compressed(
        os.path.join(args.outdir, "lipschitz_details.npz"),
        forget_lip_a=forget_lip_a,
        forget_lip_b=forget_lip_b,
        retain_lip_a=retain_lip_a,
        retain_lip_b=retain_lip_b,
        forget_grad_a=forget_grad_a,
        forget_grad_b=forget_grad_b,
        retain_grad_a=retain_grad_a,
        retain_grad_b=retain_grad_b,
        forget_var_a=forget_var_a,
        forget_var_b=forget_var_b,
        retain_var_a=retain_var_a,
        retain_var_b=retain_var_b,
    )

    # Save summary JSON
    summary = {
        "model_a": args.model_a,
        "model_b": args.model_b,
        "forget_lipschitz_change": float(forget_lip_change),
        "retain_lipschitz_change": float(retain_lip_change),
        "forget_smoother": bool(forget_lip_ratio < 0.9),
        "selective_smoothing": bool(forget_lip_ratio < 0.9 and retain_lip_ratio > 0.95),
        "epsilon": args.epsilon,
        "num_perturbations": args.num_perturbations,
    }

    with open(os.path.join(args.outdir, "lipschitz_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[local_lipschitz] ✓ Results saved to {args.outdir}")
    print(f"[local_lipschitz] Forget Lipschitz change: {forget_lip_change:.3f}")
    print(f"[local_lipschitz] Retain Lipschitz change: {retain_lip_change:.3f}")
    print(f"[local_lipschitz] Forget became {'smoother' if forget_lip_ratio < 0.9 else 'rougher' if forget_lip_ratio > 1.1 else 'similar'}")
    log_csv_as_table(os.path.join(args.outdir, "lipschitzness_summary.csv"), "lipschitzness_summary")
    log_plots(args.outdir, "lipschitzness")
    finish_wandb()


if __name__ == "__main__":
    main()