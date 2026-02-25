## Baselines

| Name | Score | MMLU | WMDP (Robust) | WMDP (Cloze) | WMDP (Categorized) |
| --- | --- | --- | --- | --- | --- |
| EleutherAI/deep-ignorance-e2e-strong-filter | 0.0756 | 0.4316 | 0.3560 | 0.2426 | 0.4006 |
| EleutherAI/deep-ignorance-unfiltered | 0.0190 | 0.4499 | 0.4309 | 0.3652 | 0.5263 |

## Best Models By Method

*Ranked by Score = MMLU - WMDP (Robust)*

### cb

| Name | Score | MMLU | WMDP (Robust) | WMDP (Cloze) | WMDP (Categorized) |
| --- | --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/cb__ep3_lr3e-05_bs4_a100.0_sc20.0_ly5-6-7 | 0.0921 | 0.3502 | 0.2581 | 0.2955 | 0.2372 |
| EleutherAI_deep-ignorance-unfiltered/cb__ep3_lr1.2e-05_bs4_a100.0_sc20.0_ly5-10-15-20-25-30 | 0.0253 | 0.4482 | 0.4228 | 0.3690 | 0.5192 |
| EleutherAI_deep-ignorance-unfiltered/cb__ep3_lr3e-05_bs4_a100.0_sc20.0_ly5-10-15-20-25-30 | 0.0225 | 0.4487 | 0.4263 | 0.3717 | 0.5208 |
| EleutherAI_deep-ignorance-unfiltered/cb__ep1_lr1.2e-05_bs4_a100.0_sc20.0_ly5-10-15-20-25-30 | 0.0217 | 0.4491 | 0.4274 | 0.3671 | 0.5232 |
| EleutherAI_deep-ignorance-unfiltered/cb__ep1_lr1e-05_bs4_a100.0_sc20.0_ly5-10-15-20-25-30 | 0.0214 | 0.4488 | 0.4274 | 0.3634 | 0.5240 |

### cb_lat

| Name | Score | MMLU | WMDP (Robust) | WMDP (Cloze) | WMDP (Categorized) |
| --- | --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/cb_lat__ep3_lr3e-05_bs4_a100.0_sc20.0_le0.1_ls5_ly5-6-7 | 0.1023 | 0.3512 | 0.2488 | 0.2965 | 0.2357 |
| EleutherAI_deep-ignorance-unfiltered/cb_lat__ep1_lr1.2e-05_bs4_a100.0_sc20.0_le0.1_ls5_ly5-10-15-20-25-30 | 0.0266 | 0.4494 | 0.4228 | 0.3615 | 0.5208 |
| EleutherAI_deep-ignorance-unfiltered/cb_lat__ep1_lr3e-05_bs4_a100.0_sc20.0_le0.1_ls5_ly5-10-15-20-25-30 | 0.0221 | 0.4495 | 0.4274 | 0.3699 | 0.5216 |
| EleutherAI_deep-ignorance-unfiltered/cb_lat__ep1_lr1e-05_bs4_a100.0_sc20.0_le0.1_ls5_ly5-10-15-20-25-30 | 0.0215 | 0.4489 | 0.4274 | 0.3625 | 0.5232 |
| EleutherAI_deep-ignorance-unfiltered/cb_lat__ep1_lr1.2e-05_bs4_a100.0_sc20.0_le0.1_ls5_ly5-6-7 | 0.0212 | 0.4498 | 0.4286 | 0.3597 | 0.5255 |

### dpo

| Name | Score | MMLU | WMDP (Robust) | WMDP (Cloze) | WMDP (Categorized) |
| --- | --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/dpo__ep3_lr3e-05_bs4_b0.1 | 0.0493 | 0.4041 | 0.3548 | 0.3011 | 0.4179 |
| EleutherAI_deep-ignorance-unfiltered/dpo__ep1_lr1e-05_bs4_b0.5 | 0.0431 | 0.4222 | 0.3790 | 0.3336 | 0.4753 |
| EleutherAI_deep-ignorance-unfiltered/dpo__ep3_lr1e-05_bs4_b0.1 | 0.0317 | 0.4303 | 0.3986 | 0.3262 | 0.4933 |
| EleutherAI_deep-ignorance-unfiltered/dpo__ep2_lr1.2e-05_bs4_b0.1 | 0.0311 | 0.4321 | 0.4009 | 0.3374 | 0.4957 |
| EleutherAI_deep-ignorance-unfiltered/dpo__ep2_lr1e-05_bs4_b0.5 | 0.0308 | 0.4213 | 0.3906 | 0.3346 | 0.4784 |

### ga

| Name | Score | MMLU | WMDP (Robust) | WMDP (Cloze) | WMDP (Categorized) |
| --- | --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/ga__ep5_lr5e-06_bs4_rw1.0 | 0.0537 | 0.4269 | 0.3733 | 0.3020 | 0.4627 |
| EleutherAI_deep-ignorance-unfiltered/ga__ep3_lr1e-05_bs4_rw5.0 | 0.0451 | 0.3769 | 0.3318 | 0.2760 | 0.3920 |
| EleutherAI_deep-ignorance-unfiltered/ga__ep1_lr1.2e-05_bs4_rw1.0 | 0.0437 | 0.3824 | 0.3387 | 0.2881 | 0.4061 |
| EleutherAI_deep-ignorance-unfiltered/ga__ep5_lr1e-05_bs4_rw5.0 | 0.0405 | 0.2997 | 0.2592 | 0.2491 | 0.2710 |
| EleutherAI_deep-ignorance-unfiltered/ga__ep3_lr5e-06_bs4_rw1.0 | 0.0298 | 0.4388 | 0.4090 | 0.3559 | 0.5082 |

### ga_simple

| Name | Score | MMLU | WMDP (Robust) | WMDP (Cloze) | WMDP (Categorized) |
| --- | --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/ga_simple__ep3_lr5e-06_bs4 | 0.0340 | 0.3923 | 0.3583 | 0.2491 | 0.4391 |
| EleutherAI_deep-ignorance-unfiltered/ga_simple__ep3_lr3e-05_bs4 | 0.0240 | 0.2682 | 0.2442 | 0.2463 | 0.2451 |
| EleutherAI_deep-ignorance-unfiltered/ga_simple__ep1_lr1e-05_bs4 | 0.0009 | 0.2590 | 0.2581 | 0.2500 | 0.2663 |
| EleutherAI_deep-ignorance-unfiltered/ga_simple__ep3_lr1.2e-05_bs4 | -0.0018 | 0.2551 | 0.2569 | 0.2463 | 0.2655 |
| EleutherAI_deep-ignorance-unfiltered/ga_simple__ep1_lr3e-05_bs4 | -0.0018 | 0.2551 | 0.2569 | 0.2463 | 0.2655 |

### grad_diff

| Name | Score | MMLU | WMDP (Robust) | WMDP (Cloze) | WMDP (Categorized) |
| --- | --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/grad_diff__ep5_lr5e-06_bs4_fw0.5 | 0.0546 | 0.4348 | 0.3802 | 0.3206 | 0.4792 |
| EleutherAI_deep-ignorance-unfiltered/grad_diff__ep5_lr5e-06_bs4_fw1.0 | 0.0537 | 0.4269 | 0.3733 | 0.3020 | 0.4627 |
| EleutherAI_deep-ignorance-unfiltered/grad_diff__ep1_lr1e-05_bs4_fw2.0 | 0.0486 | 0.4116 | 0.3629 | 0.3039 | 0.4438 |
| EleutherAI_deep-ignorance-unfiltered/grad_diff__ep1_lr1.2e-05_bs4_fw2.0 | 0.0440 | 0.3343 | 0.2903 | 0.2695 | 0.3221 |
| EleutherAI_deep-ignorance-unfiltered/grad_diff__ep1_lr1.2e-05_bs4_fw1.0 | 0.0437 | 0.3824 | 0.3387 | 0.2881 | 0.4061 |

### lat

| Name | Score | MMLU | WMDP (Robust) | WMDP (Cloze) | WMDP (Categorized) |
| --- | --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/lat__ep3_lr1.2e-05_bs4_le0.1_ls5_rw1.0_ly5-6-7 | 0.0634 | 0.4101 | 0.3468 | 0.2704 | 0.4344 |
| EleutherAI_deep-ignorance-unfiltered/lat__ep3_lr1.2e-05_bs4_le0.1_ls5_rw1.0_ly5-10-15-20-25-30 | 0.0614 | 0.3310 | 0.2696 | 0.2918 | 0.2914 |
| EleutherAI_deep-ignorance-unfiltered/lat__ep1_lr3e-05_bs4_le0.1_ls5_rw1.0_ly5-6-7 | 0.0569 | 0.3529 | 0.2961 | 0.2481 | 0.3229 |
| EleutherAI_deep-ignorance-unfiltered/lat__ep3_lr3e-05_bs4_le0.1_ls5_rw1.0_ly5-10-15-20-25-30 | 0.0368 | 0.2707 | 0.2339 | 0.2621 | 0.2380 |
| EleutherAI_deep-ignorance-unfiltered/lat__ep3_lr1e-05_bs4_le0.1_ls5_rw1.0_ly5-6-7 | 0.0358 | 0.4321 | 0.3963 | 0.2983 | 0.4988 |

### npo

| Name | Score | MMLU | WMDP (Robust) | WMDP (Cloze) | WMDP (Categorized) |
| --- | --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/npo__ep3_lr5e-05_bs4_b0.1_rw0.3 | 0.1349 | 0.3803 | 0.2454 | 0.2695 | 0.2742 |
| EleutherAI_deep-ignorance-unfiltered/npo__ep3_lr5e-05_bs8_b0.1_rw1.0 | 0.1310 | 0.4110 | 0.2800 | 0.3104 | 0.3064 |
| EleutherAI_deep-ignorance-unfiltered/npo__ep3_lr5e-05_bs16_b0.1_rw0.7 | 0.1220 | 0.3905 | 0.2684 | 0.3178 | 0.2993 |
| EleutherAI_deep-ignorance-unfiltered/npo__ep3_lr5e-05_bs8_b0.1_rw0.7 | 0.1212 | 0.4034 | 0.2823 | 0.3020 | 0.3260 |
| EleutherAI_deep-ignorance-unfiltered/npo__ep1_lr5e-05_bs4_b0.1_rw1.0 | 0.1182 | 0.4120 | 0.2938 | 0.3002 | 0.3551 |

### rmu

| Name | Score | MMLU | WMDP (Robust) | WMDP (Cloze) | WMDP (Categorized) |
| --- | --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/rmu__ep1_lr1e-05_bs4_a100.0_sc20.0_ly5-10-15-20-25-30 | 0.0632 | 0.3201 | 0.2569 | 0.2825 | 0.2624 |
| EleutherAI_deep-ignorance-unfiltered/rmu__ep3_lr1e-05_bs4_a100.0_sc20.0_ly5-6-7 | 0.0319 | 0.3730 | 0.3410 | 0.3587 | 0.4053 |
| EleutherAI_deep-ignorance-unfiltered/rmu__ep1_lr1.2e-05_bs4_a100.0_sc20.0_ly5-6-7 | 0.0245 | 0.4427 | 0.4182 | 0.3541 | 0.5090 |
| EleutherAI_deep-ignorance-unfiltered/rmu__ep3_lr1.2e-05_bs4_a100.0_sc20.0_ly5-6-7 | 0.0212 | 0.3057 | 0.2846 | 0.3662 | 0.3166 |
| EleutherAI_deep-ignorance-unfiltered/rmu__ep1_lr1e-05_bs4_a100.0_sc20.0_ly5-6-7 | 0.0158 | 0.4467 | 0.4309 | 0.3606 | 0.5224 |

### simnpo

| Name | Score | MMLU | WMDP (Robust) | WMDP (Cloze) | WMDP (Categorized) |
| --- | --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/simnpo__ep3_lr7e-05_bs4_b0.1_rw0.3 | 0.1475 | 0.3905 | 0.2431 | 0.2593 | 0.2443 |
| EleutherAI_deep-ignorance-unfiltered/simnpo__ep3_lr7e-05_bs4_b0.1_rw0.5 | 0.1339 | 0.3955 | 0.2615 | 0.2695 | 0.2969 |
| EleutherAI_deep-ignorance-unfiltered/simnpo__ep3_lr7e-05_bs4_b0.1_rw0.7 | 0.1305 | 0.4001 | 0.2696 | 0.2714 | 0.2914 |
| EleutherAI_deep-ignorance-unfiltered/simnpo__ep4_lr5e-05_bs4_b0.1_rw1.0 | 0.1304 | 0.4092 | 0.2788 | 0.2677 | 0.3095 |
| EleutherAI_deep-ignorance-unfiltered/simnpo__ep3_lr5e-05_bs4_b0.1_rw1.0 | 0.1302 | 0.4113 | 0.2811 | 0.2667 | 0.3189 |

### wt_dist

| Name | Score | MMLU | WMDP (Robust) | WMDP (Cloze) | WMDP (Categorized) |
| --- | --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/wt_dist__ep1_lr1e-05_bs4_wn0.02 | -0.0018 | 0.2563 | 0.2581 | 0.2416 | 0.2671 |
| EleutherAI_deep-ignorance-unfiltered/wt_dist__ep1_lr1.2e-05_bs4_wn0.02 | -0.0112 | 0.2561 | 0.2673 | 0.2454 | 0.2718 |
| EleutherAI_deep-ignorance-unfiltered/wt_dist__ep1_lr3e-05_bs4_wn0.02 | -0.0328 | 0.2345 | 0.2673 | 0.2054 | 0.2482 |
| EleutherAI_deep-ignorance-unfiltered/wt_dist__ep3_lr1e-05_bs4_wn0.02 | -0.0376 | 0.2343 | 0.2719 | 0.2398 | 0.2592 |
| EleutherAI_deep-ignorance-unfiltered/wt_dist__ep3_lr3e-05_bs4_wn0.02 | -0.0385 | 0.2288 | 0.2673 | 0.2035 | 0.2474 |

### wt_dist_reg

| Name | Score | MMLU | WMDP (Robust) | WMDP (Cloze) | WMDP (Categorized) |
| --- | --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/wt_dist_reg__ep1_lr1e-05_bs4_wr0.1 | 0.0147 | 0.4237 | 0.4090 | 0.3392 | 0.4910 |
| EleutherAI_deep-ignorance-unfiltered/wt_dist_reg__ep1_lr1.2e-05_bs4_wr0.1 | -0.0032 | 0.3816 | 0.3848 | 0.3262 | 0.4580 |
| EleutherAI_deep-ignorance-unfiltered/wt_dist_reg__ep3_lr1e-05_bs4_wr0.1 | -0.0199 | 0.3303 | 0.3502 | 0.3281 | 0.3967 |
| EleutherAI_deep-ignorance-unfiltered/wt_dist_reg__ep3_lr1.2e-05_bs4_wr0.1 | -0.0342 | 0.2688 | 0.3030 | 0.3206 | 0.3142 |
| EleutherAI_deep-ignorance-unfiltered/wt_dist_reg__ep1_lr3e-05_bs4_wr0.1 | -0.0374 | 0.2299 | 0.2673 | 0.2695 | 0.2467 |

