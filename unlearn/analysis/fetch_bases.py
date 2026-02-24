import wandb
from dotenv import load_dotenv

load_dotenv()
api = wandb.Api()
runs = api.runs("cambridge_era")

for r in runs:
    if r.state != "finished": continue
    name = r.name
    # if it's not a sweep run it might be the base model eval
    if "__ep" not in name:
        print(f"Name: {name}")
        print(f"MMLU: {r.summary.get('eval_bench/mmlu/acc')}")
        print(f"WMDP R: {r.summary.get('eval_bench/wmdp_bio_robust_rewritten/acc')}")
        print(f"WMDP C: {r.summary.get('eval_bench/wmdp_bio_cloze_verified/acc_norm')}")
        print(f"WMDP Cat: {r.summary.get('eval_bench/wmdp_bio_categorized_mcqa/acc')}")
        print("---")
