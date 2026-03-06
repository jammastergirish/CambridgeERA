## Baselines

| Name | Score | MMLU | WMDP (Robust) | WMDP (Robust Rewritten) | WMDP (Cloze) | WMDP (Categorized) |
| --- | --- | --- | --- | --- | --- | --- |
| EleutherAI/deep-ignorance-e2e-strong-filter | 0.0756 | 0.4316 | 0.3560 | 0.2499 | 0.2426 | 0.4006 |
| EleutherAI/deep-ignorance-unfiltered | 0.0190 | 0.4499 | 0.4309 | 0.2573 | 0.3652 | 0.5263 |

## Best Models By Method

*Ranked by Score = MMLU - WMDP (Robust)*

### cb

| Name | Score | MMLU | WMDP (Robust) | WMDP (Robust Rewritten) | WMDP (Cloze) | WMDP (Categorized) |
| --- | --- | --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/cb__ep3_lr5e-05_bs32_a50.0_sc20.0_ly5-6-7_ml1024 | -0.0019 | 0.2446 | 0.2465 | 0.2647 | 0.2268 | 0.2459 |
| EleutherAI_deep-ignorance-unfiltered/cb__ep3_lr5e-05_bs32_a50.0_sc25.0_ly5-6-7_ml1024 | -0.0053 | 0.2447 | 0.2500 | 0.2663 | 0.2407 | 0.2514 |
| EleutherAI_deep-ignorance-unfiltered/cb__ep3_lr5e-05_bs32_a100.0_sc25.0_ly5-6-7_ml1024 | -0.0150 | 0.2465 | 0.2615 | 0.2717 | 0.2268 | 0.2608 |
| EleutherAI_deep-ignorance-unfiltered/cb__ep3_lr5e-05_bs32_a150.0_sc20.0_ly5-6-7_ml1024 | -0.0192 | 0.2423 | 0.2615 | 0.2573 | 0.2240 | 0.2506 |
| EleutherAI_deep-ignorance-unfiltered/cb__ep3_lr3e-05_bs32_a50.0_sc20.0_ly5-6-7_ml1024 | -0.0209 | 0.2349 | 0.2558 | 0.2684 | 0.2296 | 0.2490 |

### ga

| Name | Score | MMLU | WMDP (Robust) | WMDP (Robust Rewritten) | WMDP (Cloze) | WMDP (Categorized) |
| --- | --- | --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/ga__ep3_lr3e-05_bs32_rw5.0_ml2048 | 0.1173 | 0.3558 | 0.2385 | 0.2404 | 0.2472 | 0.2553 |
| EleutherAI_deep-ignorance-unfiltered/ga__ep3_lr3e-05_bs32_rw3.0_ml2048 | 0.0986 | 0.3393 | 0.2408 | 0.2248 | 0.2454 | 0.2537 |
| EleutherAI_deep-ignorance-unfiltered/ga__ep3_lr3e-05_bs32_rw2.0_ml2048 | 0.0528 | 0.2844 | 0.2316 | 0.2437 | 0.2546 | 0.2404 |
| EleutherAI_deep-ignorance-unfiltered/ga__ep4_lr3e-05_bs32_rw1.0_ml2048 | 0.0085 | 0.2596 | 0.2512 | 0.2520 | 0.2426 | 0.2419 |
| EleutherAI_deep-ignorance-unfiltered/ga__ep1_lr3e-05_bs32_rw1.0_ml2048 | -0.0113 | 0.2352 | 0.2465 | 0.2536 | 0.2416 | 0.2435 |

### npo

| Name | Score | MMLU | WMDP (Robust) | WMDP (Robust Rewritten) | WMDP (Cloze) | WMDP (Categorized) |
| --- | --- | --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/npo__ep3_lr4e-05_bs32_b0.01_rw1.0_ml2048 | 0.1440 | 0.3813 | 0.2373 | 0.2548 | 0.2509 | 0.2506 |
| EleutherAI_deep-ignorance-unfiltered/npo__ep3_lr3e-05_bs32_b0.02_rw1.0_ml2048 | 0.1098 | 0.3413 | 0.2316 | 0.2577 | 0.2612 | 0.2302 |
| EleutherAI_deep-ignorance-unfiltered/npo__ep3_lr3e-05_bs32_b0.01_rw1.0_ml4096 | 0.1036 | 0.3675 | 0.2638 | 0.2676 | 0.2472 | 0.2718 |
| EleutherAI_deep-ignorance-unfiltered/npo__ep3_lr3e-05_bs16_b0.01_rw1.0_ml2048 | 0.0729 | 0.3437 | 0.2707 | 0.2643 | 0.2584 | 0.2490 |
| EleutherAI_deep-ignorance-unfiltered/npo__ep3_lr3e-05_bs32_b0.01_rw1.0_ml2048 | 0.0728 | 0.3159 | 0.2431 | 0.2626 | 0.2584 | 0.2349 |

### rmu

| Name | Score | MMLU | WMDP (Robust) | WMDP (Robust Rewritten) | WMDP (Cloze) | WMDP (Categorized) |
| --- | --- | --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/rmu__ep2_lr1e-05_bs32_a100.0_sc20.0_ly11-12-13_ml2048 | 0.0386 | 0.4338 | 0.3952 | 0.2503 | 0.3662 | 0.4792 |
| EleutherAI_deep-ignorance-unfiltered/rmu__ep2_lr1e-05_bs32_a100.0_sc20.0_ly5-6-7-11-12-13_ml2048 | 0.0336 | 0.4345 | 0.4009 | 0.2524 | 0.3606 | 0.4847 |
| EleutherAI_deep-ignorance-unfiltered/rmu__ep1_lr1e-05_bs32_a100.0_sc20.0_ly5-6-7_ml2048 | 0.0242 | 0.4504 | 0.4263 | 0.2717 | 0.3634 | 0.5224 |
| EleutherAI_deep-ignorance-unfiltered/rmu__ep2_lr1e-05_bs32_a100.0_sc20.0_ly5-6-7_ml2048 | 0.0190 | 0.4511 | 0.4320 | 0.2713 | 0.3578 | 0.5247 |
| EleutherAI_deep-ignorance-unfiltered/rmu__ep2_lr1e-05_bs32_a100.0_sc40.0_ly5-6-7_ml2048 | 0.0170 | 0.4501 | 0.4332 | 0.2717 | 0.3578 | 0.5255 |

### simnpo

| Name | Score | MMLU | WMDP (Robust) | WMDP (Robust Rewritten) | WMDP (Cloze) | WMDP (Categorized) |
| --- | --- | --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/simnpo__ep3_lr3e-05_bs32_b0.01_rw1.0_ml2048 | 0.1780 | 0.4315 | 0.2535 | 0.2614 | 0.2565 | 0.2616 |
| EleutherAI_deep-ignorance-unfiltered/simnpo__ep4_lr3e-05_bs32_b0.01_rw1.0_ml2048 | 0.1650 | 0.4288 | 0.2638 | 0.2347 | 0.2602 | 0.2765 |
| EleutherAI_deep-ignorance-unfiltered/simnpo__ep3_lr3e-05_bs32_b0.01_rw1.0_ml1024 | 0.1278 | 0.3962 | 0.2684 | 0.2544 | 0.2574 | 0.2687 |
| EleutherAI_deep-ignorance-unfiltered/simnpo__ep4_lr3e-05_bs32_b0.01_rw1.0_ml1024 | 0.1117 | 0.4205 | 0.3088 | 0.2470 | 0.2909 | 0.3331 |
| girishgupta_simnpo__ep3_lr3e-05_bs4_b0.01_rw1.0_ml1024 | 0.1105 | 0.4216 | 0.3111 | 0.2680 | 0.2695 | 0.3401 |

### tar

| Name | Score | MMLU | WMDP (Robust) | WMDP (Robust Rewritten) | WMDP (Cloze) | WMDP (Categorized) |
| --- | --- | --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/tar__ta1.0_tlr1e-05_tep1_ml1024 | -0.0378 | 0.2295 | 0.2673 | 0.2651 | 0.2491 | 0.2467 |

### wt_dist

| Name | Score | MMLU | WMDP (Robust) | WMDP (Robust Rewritten) | WMDP (Cloze) | WMDP (Categorized) |
| --- | --- | --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/wt_dist__ep3_lr3e-05_bs32_wn0.01_ml2048 | 0.0615 | 0.3518 | 0.2903 | 0.2598 | 0.3550 | 0.3543 |
| EleutherAI_deep-ignorance-unfiltered/wt_dist__ep3_lr3e-05_bs32_wn0.005_ml2048 | 0.0270 | 0.4256 | 0.3986 | 0.2598 | 0.3931 | 0.4776 |
| EleutherAI_deep-ignorance-unfiltered/wt_dist__ep3_lr3e-05_bs32_wn0.1_ml2048 | -0.0066 | 0.2537 | 0.2604 | 0.2343 | 0.2388 | 0.2694 |
| EleutherAI_deep-ignorance-unfiltered/wt_dist__ep3_lr3e-05_bs32_wn0.05_ml2048 | -0.0326 | 0.2301 | 0.2627 | 0.2684 | 0.2435 | 0.2474 |
| EleutherAI_deep-ignorance-unfiltered/wt_dist__ep2_lr3e-05_bs32_wn0.02_ml2048 | -0.0359 | 0.2291 | 0.2650 | 0.2651 | 0.2110 | 0.2443 |

