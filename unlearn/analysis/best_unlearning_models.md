## Baselines

No baseline runs found.

To generate baselines, run:
```bash
uv run experiment/eval.py --model EleutherAI/deep-ignorance-unfiltered \
    --wandb-project cambridge_era --wandb-name EleutherAI/deep-ignorance-unfiltered
```

## Best Models By Method

*Ranked by Score = MMLU - WMDP (Robust)*

### tar

| Name | Score | MMLU | WMDP (Robust) | WMDP (Cloze) | WMDP (Categorized) |
| --- | --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/tar__ta1.0_tlr1e-05_tep1_ml1024 | -0.0378 | 0.2295 | 0.2673 | 0.2491 | 0.2467 |
| EleutherAI_deep-ignorance-unfiltered/tar__ta1.0_tlr1e-05_tep1_ml2048 | -0.0378 | 0.2295 | 0.2673 | 0.2491 | 0.2467 |
| EleutherAI_deep-ignorance-unfiltered/tar__ta1.0_tlr1e-05_tep1_ml4096 | -0.0378 | 0.2295 | 0.2673 | 0.2491 | 0.2467 |
| EleutherAI_deep-ignorance-unfiltered/tar__ta0.5_tlr1e-05_tep1_ml2048 | -0.0378 | 0.2295 | 0.2673 | 0.2491 | 0.2467 |
| EleutherAI_deep-ignorance-unfiltered/tar__ta1.0_tlr1e-05_tep1_ml2048 | -0.0378 | 0.2295 | 0.2673 | 0.2491 | 0.2467 |

