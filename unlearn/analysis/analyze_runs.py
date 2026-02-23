# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "wandb",
#     "pandas",
#     "python-dotenv",
#     "tqdm",
# ]
# ///

import wandb
import pandas as pd
from dotenv import load_dotenv

def main():
    load_dotenv()
    api = wandb.Api()

    project_name = "cambridge_era"
    
    print(f"Fetching runs from project '{project_name}'...")
    try:
        runs = api.runs(project_name)
    except Exception as e:
        print(f"Error fetching runs: {e}")
        print("Please check your WANDB_API_KEY and ensure the project name ({project_name}) is correct.")
        return

    data = []
    
    from tqdm import tqdm
    for run in tqdm(runs, desc="Processing runs"):
        if run.state != "finished":
            continue
            
        # Get method from config
        method = run.config.get("hyperparameters", {}).get("method", "unknown")
        
        # Get eval metrics from summary
        # W&B flattens keys so we'll look for:
        # eval_bench/mmlu/acc
        # eval_bench/wmdp_bio_robust_rewritten/acc
        # eval_bench/wmdp_bio_cloze_verified/acc
        # eval_bench/wmdp_bio_categorized_mcqa/acc
        
        mmlu = run.summary.get("eval_bench/mmlu/acc", None)
        wmdp_1 = run.summary.get("eval_bench/wmdp_bio_robust_rewritten/acc", None)
        wmdp_2 = run.summary.get("eval_bench/wmdp_bio_cloze_verified/acc", None)
        wmdp_3 = run.summary.get("eval_bench/wmdp_bio_categorized_mcqa/acc", None)
        
        # We only want to include runs that have evaluation data
        if mmlu is None and wmdp_1 is None and wmdp_2 is None and wmdp_3 is None:
            continue
            
        is_base = "__ep" not in run.name
        
        data.append({
            "Run ID": run.id,
            "Name": run.name,
            "Method": method,
            "MMLU": mmlu,
            "WMDP (Robust)": wmdp_1,
            "WMDP (Cloze)": wmdp_2,
            "WMDP (Categorized)": wmdp_3,
            "Loss": run.summary.get("train/loss", None),
            "IsBase": is_base
        })

    if not data:
        print("No finished runs with evaluation metrics found.")
        return

    df = pd.DataFrame(data)
    
    # Print the count of valid runs processed
    print(f"\nSuccessfully processed {len(df)} runs with evaluation metrics out of the total runs.")
    
    baselines_df = df[df["IsBase"]]
    sweeps_df = df[~df["IsBase"]]
    
    # Format the columns for printing
    cols = ["Name", "MMLU", "WMDP (Robust)", "WMDP (Cloze)", "WMDP (Categorized)"]
    
    print("=====================================")
    print("             BASELINES               ")
    print("=====================================")
    if not baselines_df.empty:
        baselines_df = baselines_df.sort_values(by="Name")
        print(baselines_df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A"))
    else:
        print("No baseline runs found.")
        print("\nTo generate and log baselines to W&B, run these commands from the project root:")
        print("  uv run experiment/eval.py --model EleutherAI/deep-ignorance-unfiltered \\")
        print("      --wandb-project cambridge_era --wandb-name EleutherAI/deep-ignorance-unfiltered")
        print()
        print("  uv run experiment/eval.py --model EleutherAI/deep-ignorance-e2e-strong-filter \\")
        print("      --wandb-project cambridge_era --wandb-name EleutherAI/deep-ignorance-e2e-strong-filter")
        
    print("\n--- Best Models By Method ---")
    
    metrics = ["MMLU", "WMDP (Robust)", "WMDP (Cloze)", "WMDP (Categorized)"]
    # We want high MMLU (retain general knowledge) and LOW WMDP (forget hazardous knowledge)
    
    sweeps_df = sweeps_df.sort_values(by=["Method", "WMDP (Categorized)", "MMLU"], ascending=[True, True, False])
    
    for method, group in sweeps_df.groupby("Method"):
        print(f"\n=====================================")
        print(f"Method: {method}")
        print(f"=====================================")
        
        # Let's display the top 3 for the method
        # Sort by best (lowest) WMDP Categorized, then highest MMLU
        best_runs = group.sort_values(by=["WMDP (Categorized)", "MMLU"], ascending=[True, False]).head(5)
        
        print(best_runs[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A"))

if __name__ == "__main__":
    main()
