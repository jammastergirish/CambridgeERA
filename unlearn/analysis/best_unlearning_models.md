## Baselines

| Name | Score | L2 Dist | MMLU | WMDP (Robust) | WMDP (Robust Rewritten) | WMDP (Cloze) | WMDP (Categorized) | Forget NLL | Retain NLL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EleutherAI/deep-ignorance-e2e-strong-filter | 0.0756 | N/A | 0.4316 | 0.3560 | 0.2499 | 0.2426 | 0.4006 | N/A | N/A |
| EleutherAI/deep-ignorance-unfiltered | 0.0190 | N/A | 0.4499 | 0.4309 | 0.2573 | 0.3652 | 0.5263 | N/A | N/A |

## Best Models By Method

*Ranked by Score = MMLU - WMDP (Robust)*

### cb

| Name | Score | L2 Dist | MMLU | WMDP (Robust) | WMDP (Robust Rewritten) | WMDP (Cloze) | WMDP (Categorized) | Forget NLL | Retain NLL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/cb__ep1_lr1.3e-05_bs16_a200.0_sc10.0_ly13-14-15_mle512_mli1024 | 0.0289 | N/A | 0.4425 | 0.4136 | 0.2594 | 0.3504 | 0.5004 | N/A | N/A |
| EleutherAI_deep-ignorance-unfiltered/cb__ep1_lr1.3e-05_bs16_a1000.0_sc5.0_ly13-14-15_mle512_mli1024 | 0.0274 | 2.6785 | 0.4421 | 0.4147 | 0.2565 | 0.3522 | 0.5020 | 2.1783 | 2.6848 |
| EleutherAI_deep-ignorance-unfiltered/cb__ep2_lr1e-05_bs32_a1000.0_sc5.0_ly11-12-13_ml2048 | 0.0165 | N/A | 0.4071 | 0.3906 | 0.2524 | 0.3606 | 0.4564 | 2.3332 | 4.3229 |
| EleutherAI_deep-ignorance-unfiltered/cb__ep3_lr1e-05_bs32_a500.0_sc10.0_ly11-12-13_ml2048 | 0.0152 | N/A | 0.3147 | 0.2995 | 0.2618 | 0.3336 | 0.3032 | N/A | N/A |
| EleutherAI_deep-ignorance-unfiltered/cb__ep3_lr1e-05_bs32_a500.0_sc10.0_ly11-12-13_ml2048 | 0.0136 | N/A | 0.3154 | 0.3018 | 0.2630 | 0.3309 | 0.3064 | 4.9704 | 7.0958 |
| EleutherAI_deep-ignorance-unfiltered/cb__ep3_lr1e-05_bs32_a800.0_sc8.0_ly11-12-13_ml2048 | 0.0104 | N/A | 0.3146 | 0.3041 | 0.2618 | 0.3336 | 0.3071 | 4.9594 | 7.0928 |
| EleutherAI_deep-ignorance-unfiltered/cb__ep3_lr5e-05_bs32_a50.0_sc20.0_ly5-6-7_ml1024 | -0.0019 | N/A | 0.2446 | 0.2465 | 0.2647 | 0.2268 | 0.2459 | 11.7820 | 10.4446 |
| EleutherAI_deep-ignorance-unfiltered/cb__ep3_lr5e-05_bs32_a50.0_sc25.0_ly5-6-7_ml1024 | -0.0053 | N/A | 0.2447 | 0.2500 | 0.2663 | 0.2407 | 0.2514 | N/A | N/A |
| EleutherAI_deep-ignorance-unfiltered/cb__ep3_lr5e-05_bs32_a100.0_sc25.0_ly5-6-7_ml1024 | -0.0150 | N/A | 0.2465 | 0.2615 | 0.2717 | 0.2268 | 0.2608 | 11.4395 | 10.4954 |
| EleutherAI_deep-ignorance-unfiltered/cb__ep3_lr5e-05_bs32_a150.0_sc20.0_ly5-6-7_ml1024 | -0.0192 | N/A | 0.2423 | 0.2615 | 0.2573 | 0.2240 | 0.2506 | 11.1575 | 10.4234 |
| EleutherAI_deep-ignorance-unfiltered/cb__ep3_lr3e-05_bs32_a50.0_sc20.0_ly5-6-7_ml1024 | -0.0209 | N/A | 0.2349 | 0.2558 | 0.2684 | 0.2296 | 0.2490 | 10.1736 | 9.5921 |
| EleutherAI_deep-ignorance-unfiltered/cb__ep3_lr3e-05_bs32_a50.0_sc25.0_ly5-6-7_ml1024 | -0.0225 | N/A | 0.2332 | 0.2558 | 0.2709 | 0.2379 | 0.2482 | 10.1675 | 9.5875 |
| EleutherAI_deep-ignorance-unfiltered/cb__ep3_lr3e-05_bs32_a50.0_sc15.0_ly5-6-7_ml1024 | -0.0252 | N/A | 0.2329 | 0.2581 | 0.2700 | 0.2351 | 0.2522 | 10.1761 | 9.6053 |
| EleutherAI_deep-ignorance-unfiltered/cb__ep3_lr5e-05_bs32_a100.0_sc20.0_ly5-6-7_ml1024 | -0.0284 | N/A | 0.2458 | 0.2742 | 0.2737 | 0.2221 | 0.2639 | 11.4968 | 10.4809 |
| EleutherAI_deep-ignorance-unfiltered/cb__ep3_lr3e-05_bs32_a150.0_sc20.0_ly5-6-7_ml1024 | -0.0308 | N/A | 0.2342 | 0.2650 | 0.2626 | 0.2454 | 0.2553 | 10.2871 | 9.4630 |
| EleutherAI_deep-ignorance-unfiltered/cb__ep3_lr5e-05_bs32_a150.0_sc25.0_ly5-6-7_ml1024 | -0.0312 | N/A | 0.2395 | 0.2707 | 0.2528 | 0.2268 | 0.2592 | 11.2043 | 10.4906 |
| EleutherAI_deep-ignorance-unfiltered/cb__ep3_lr3e-05_bs32_a150.0_sc15.0_ly5-6-7_ml1024 | -0.0330 | N/A | 0.2354 | 0.2684 | 0.2647 | 0.2481 | 0.2545 | 10.2448 | 9.4423 |
| EleutherAI_deep-ignorance-unfiltered/cb__ep3_lr3e-05_bs32_a150.0_sc25.0_ly5-6-7_ml1024 | -0.0335 | N/A | 0.2361 | 0.2696 | 0.2594 | 0.2481 | 0.2553 | 10.2625 | 9.4550 |
| EleutherAI_deep-ignorance-unfiltered/cb__ep3_lr3e-05_bs32_a100.0_sc15.0_ly5-6-7_ml1024 | -0.0346 | N/A | 0.2338 | 0.2684 | 0.2651 | 0.2444 | 0.2514 | 10.3023 | 9.4916 |
| EleutherAI_deep-ignorance-unfiltered/cb__ep3_lr3e-05_bs32_a100.0_sc25.0_ly5-6-7_ml1024 | -0.0363 | N/A | 0.2333 | 0.2696 | 0.2622 | 0.2491 | 0.2522 | 10.2902 | 9.4786 |
| EleutherAI_deep-ignorance-unfiltered/cb__ep3_lr3e-05_bs32_a100.0_sc20.0_ly5-6-7_ml1024 | -0.0398 | N/A | 0.2344 | 0.2742 | 0.2651 | 0.2435 | 0.2608 | 10.3004 | 9.4840 |
| EleutherAI_deep-ignorance-unfiltered/cb__ep3_lr1e-05_bs1_a800.0_sc8.0_ly11-12-13_ml6144 | -0.0471 | N/A | 0.2317 | 0.2788 | 0.2676 | 0.2398 | 0.2482 | 9.7559 | 8.7581 |

### cb_lat

| Name | Score | L2 Dist | MMLU | WMDP (Robust) | WMDP (Robust Rewritten) | WMDP (Cloze) | WMDP (Categorized) | Forget NLL | Retain NLL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/cb_lat__ep1_lr1.3e-05_bs16_a1000.0_sc5.0_le0.1_ls5_ly13-14-15_mle512_mli1024 | 0.0294 | 2.6796 | 0.4430 | 0.4136 | 0.2569 | 0.3485 | 0.5035 | 2.1783 | 2.6847 |

### dpo

| Name | Score | L2 Dist | MMLU | WMDP (Robust) | WMDP (Robust Rewritten) | WMDP (Cloze) | WMDP (Categorized) | Forget NLL | Retain NLL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/dpo__ep3_lr4e-05_bs32_b0.01_mle512_mli2048 | 0.0879 | N/A | 0.3551 | 0.2673 | 0.2511 | 0.2481 | 0.2757 | N/A | N/A |
| EleutherAI_deep-ignorance-unfiltered/dpo__ep3_lr3e-05_bs32_b0.01_mle512_mli2048 | 0.0473 | N/A | 0.3089 | 0.2615 | 0.2269 | 0.2221 | 0.2498 | 162.2404 | 3.4266 |
| EleutherAI_deep-ignorance-unfiltered/dpo__ep3_lr3e-05_bs32_b0.05_mle512_mli2048 | 0.0303 | N/A | 0.3310 | 0.3007 | 0.2651 | 0.2110 | 0.3134 | N/A | N/A |
| EleutherAI_deep-ignorance-unfiltered/dpo__ep3_lr3e-05_bs32_b0.1_mle512_mli2048 | 0.0238 | N/A | 0.3257 | 0.3018 | 0.2659 | 0.2500 | 0.3048 | N/A | N/A |

### ga

| Name | Score | L2 Dist | MMLU | WMDP (Robust) | WMDP (Robust Rewritten) | WMDP (Cloze) | WMDP (Categorized) | Forget NLL | Retain NLL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/ga__ep3_lr3e-05_bs32_rw5.0_ml2048 | 0.1173 | N/A | 0.3558 | 0.2385 | 0.2404 | 0.2472 | 0.2553 | 199.1775 | 2.8462 |
| EleutherAI_deep-ignorance-unfiltered/ga__ep3_lr3e-05_bs32_rw3.0_ml2048 | 0.0986 | N/A | 0.3393 | 0.2408 | 0.2248 | 0.2454 | 0.2537 | 198.9943 | 2.6624 |
| EleutherAI_deep-ignorance-unfiltered/ga__ep3_lr3e-05_bs32_rw2.0_ml2048 | 0.0528 | N/A | 0.2844 | 0.2316 | 0.2437 | 0.2546 | 0.2404 | 195.8933 | 2.7136 |
| EleutherAI_deep-ignorance-unfiltered/ga__ep1_lr2e-05_bs32_rw5.0_mle512_mli1024 | 0.0132 | 3.1391 | 0.4499 | 0.4366 | 0.2548 | 0.3550 | 0.5287 | 2.1596 | 2.2369 |
| EleutherAI_deep-ignorance-unfiltered/ga__ep4_lr3e-05_bs32_rw1.0_ml2048 | 0.0085 | N/A | 0.2596 | 0.2512 | 0.2520 | 0.2426 | 0.2419 | 203.0704 | 2.9996 |
| EleutherAI_deep-ignorance-unfiltered/ga__ep1_lr3e-05_bs32_rw1.0_ml2048 | -0.0113 | N/A | 0.2352 | 0.2465 | 0.2536 | 0.2416 | 0.2435 | 180.4826 | 3.3891 |
| EleutherAI_deep-ignorance-unfiltered/ga__ep3_lr3e-05_bs32_rw1.0_ml2048 | -0.0130 | N/A | 0.2416 | 0.2546 | 0.2396 | 0.2398 | 0.2529 | 201.5835 | 3.4513 |
| EleutherAI_deep-ignorance-unfiltered/ga__ep2_lr3e-05_bs32_rw1.0_ml2048 | -0.0168 | N/A | 0.2344 | 0.2512 | 0.2458 | 0.2398 | 0.2372 | 198.0561 | 2.8619 |
| EleutherAI_deep-ignorance-unfiltered/ga__ep3_lr1e-05_bs32_rw1.0_ml2048 | -0.0312 | N/A | 0.2326 | 0.2638 | 0.2647 | 0.2379 | 0.2632 | 212.4552 | 3.8803 |
| EleutherAI_deep-ignorance-unfiltered/ga__ep3_lr2e-05_bs32_rw1.0_ml2048 | -0.0352 | N/A | 0.2609 | 0.2961 | 0.2643 | 0.2444 | 0.2742 | 217.0435 | 3.1056 |

### ga_simple

| Name | Score | L2 Dist | MMLU | WMDP (Robust) | WMDP (Robust Rewritten) | WMDP (Cloze) | WMDP (Categorized) | Forget NLL | Retain NLL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/ga_simple__ep1_lr1e-05_bs32_ml1024 | 0.0231 | N/A | 0.4471 | 0.4240 | 0.2606 | 0.3727 | 0.5185 | 2.1441 | 2.1690 |
| EleutherAI_deep-ignorance-unfiltered/ga_simple__ep2_lr1.5e-05_bs32_ml1024 | 0.0224 | N/A | 0.3865 | 0.3641 | 0.2491 | 0.3281 | 0.3998 | 43.6780 | 30.5354 |
| EleutherAI_deep-ignorance-unfiltered/ga_simple__ep3_lr1e-05_bs32_ml1024 | 0.0216 | N/A | 0.4133 | 0.3917 | 0.2561 | 0.3467 | 0.4564 | 39.5764 | 23.9488 |
| EleutherAI_deep-ignorance-unfiltered/ga_simple__ep1_lr2e-05_bs32_ml1024 | 0.0209 | N/A | 0.3723 | 0.3514 | 0.2565 | 0.3058 | 0.3826 | N/A | N/A |
| EleutherAI_deep-ignorance-unfiltered/ga_simple__ep2_lr1e-05_bs32_ml1024 | 0.0194 | N/A | 0.4365 | 0.4171 | 0.2647 | 0.3773 | 0.4996 | 3.4951 | 2.5983 |
| EleutherAI_deep-ignorance-unfiltered/ga_simple__ep1_lr1.5e-05_bs32_ml1024 | 0.0153 | N/A | 0.4416 | 0.4263 | 0.2635 | 0.3727 | 0.5122 | 2.6094 | 2.3711 |
| EleutherAI_deep-ignorance-unfiltered/ga_simple__ep1_lr3e-05_bs32_ml1024 | 0.0134 | N/A | 0.2485 | 0.2350 | 0.2339 | 0.2407 | 0.2482 | 90.5616 | 78.9245 |

### grad_diff

| Name | Score | L2 Dist | MMLU | WMDP (Robust) | WMDP (Robust Rewritten) | WMDP (Cloze) | WMDP (Categorized) | Forget NLL | Retain NLL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/grad_diff__ep3_lr4e-05_bs32_fw1.0_mle512_mli2048 | 0.0885 | N/A | 0.3546 | 0.2661 | 0.2725 | 0.2463 | 0.2655 | 181.1359 | 2.7602 |
| EleutherAI_deep-ignorance-unfiltered/grad_diff__ep3_lr4e-05_bs32_fw0.75_mle512_mli2048 | 0.0766 | N/A | 0.3474 | 0.2707 | 0.2639 | 0.2602 | 0.2553 | 180.6188 | 2.6808 |
| EleutherAI_deep-ignorance-unfiltered/grad_diff__ep3_lr4e-05_bs32_fw1.5_mle512_mli2048 | 0.0566 | N/A | 0.3377 | 0.2811 | 0.2717 | 0.2444 | 0.2891 | 181.4866 | 3.0881 |
| EleutherAI_deep-ignorance-unfiltered/grad_diff__ep3_lr3e-05_bs32_fw1.0_ml1024 | 0.0468 | N/A | 0.3533 | 0.3065 | 0.2433 | 0.2519 | 0.3354 | 208.3995 | 3.9771 |
| EleutherAI_deep-ignorance-unfiltered/grad_diff__ep2_lr4e-05_bs32_fw1.5_mle512_mli2048 | 0.0375 | N/A | 0.3220 | 0.2846 | 0.2725 | 0.2435 | 0.2773 | 178.7582 | 3.7167 |
| EleutherAI_deep-ignorance-unfiltered/grad_diff__ep2_lr4e-05_bs32_fw1.0_mle512_mli2048 | 0.0332 | N/A | 0.3155 | 0.2823 | 0.2655 | 0.2314 | 0.2726 | 178.4407 | 2.9857 |
| EleutherAI_deep-ignorance-unfiltered/grad_diff__ep3_lr3e-05_bs32_fw0.75_mle512_mli2048 | 0.0245 | N/A | 0.3240 | 0.2995 | 0.2487 | 0.2509 | 0.3205 | 177.1177 | 2.7526 |
| EleutherAI_deep-ignorance-unfiltered/grad_diff__ep1_lr1e-05_bs32_fw1.0_mle512_mli1024 | 0.0168 | 1.2914 | 0.4488 | 0.4320 | 0.2606 | 0.3578 | 0.5271 | 2.1689 | 2.3237 |
| EleutherAI_deep-ignorance-unfiltered/grad_diff__ep3_lr3e-05_bs32_fw2.0_ml1024 | 0.0145 | N/A | 0.2807 | 0.2661 | 0.2302 | 0.2361 | 0.2702 | 167.6785 | 3.9282 |
| EleutherAI_deep-ignorance-unfiltered/grad_diff__ep3_lr3e-05_bs32_fw1.0_mle512_mli2048 | 0.0126 | N/A | 0.3145 | 0.3018 | 0.2598 | 0.2370 | 0.3032 | 177.4739 | 3.0196 |
| EleutherAI_deep-ignorance-unfiltered/grad_diff__ep1_lr4e-05_bs32_fw1.0_mle512_mli2048 | 0.0090 | N/A | 0.3223 | 0.3134 | 0.2635 | 0.2435 | 0.3307 | 172.7062 | 3.6520 |
| EleutherAI_deep-ignorance-unfiltered/grad_diff__ep2_lr3e-05_bs32_fw1.0_mle512_mli2048 | 0.0058 | N/A | 0.3134 | 0.3076 | 0.2520 | 0.2463 | 0.3119 | 175.1713 | 3.3911 |
| EleutherAI_deep-ignorance-unfiltered/grad_diff__ep3_lr3e-05_bs32_fw1.5_mle512_mli2048 | 0.0039 | N/A | 0.2965 | 0.2926 | 0.2577 | 0.2435 | 0.2852 | 177.8307 | 3.4684 |
| EleutherAI_deep-ignorance-unfiltered/grad_diff__ep3_lr3e-05_bs32_fw3.0_ml1024 | -0.0007 | N/A | 0.2712 | 0.2719 | 0.2330 | 0.2314 | 0.2726 | N/A | N/A |
| EleutherAI_deep-ignorance-unfiltered/grad_diff__ep3_lr3e-05_bs32_fw2.0_mle512_mli2048 | -0.0043 | N/A | 0.3114 | 0.3157 | 0.2528 | 0.2268 | 0.3181 | 178.0312 | 4.0228 |

### lat

| Name | Score | L2 Dist | MMLU | WMDP (Robust) | WMDP (Robust Rewritten) | WMDP (Cloze) | WMDP (Categorized) | Forget NLL | Retain NLL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/lat__ep3_lr3e-05_bs32_le0.1_ls5_rw1.0_ly5-6-7_mle512_mli2048 | 0.0156 | N/A | 0.2898 | 0.2742 | 0.2507 | 0.2704 | 0.2663 | 188.5414 | 2.6867 |

### npo

| Name | Score | L2 Dist | MMLU | WMDP (Robust) | WMDP (Robust Rewritten) | WMDP (Cloze) | WMDP (Categorized) | Forget NLL | Retain NLL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/npo__ep3_lr4e-05_bs32_b0.01_rw1.0_ml2048 | 0.1440 | N/A | 0.3813 | 0.2373 | 0.2548 | 0.2509 | 0.2506 | 205.0158 | 2.9583 |
| EleutherAI_deep-ignorance-unfiltered/npo__ep1_lr5e-05_bs32_b0.01_rw1.5_mle512_mli6144 | 0.1255 | 18.2407 | 0.3974 | 0.2719 | 0.2696 | 0.2481 | 0.2852 | 182.1695 | 2.6186 |
| EleutherAI_deep-ignorance-unfiltered/npo__ep1_lr5e-05_bs32_b0.01_rw1.0_mle512_mli6144 | 0.1121 | 18.1638 | 0.3495 | 0.2373 | 0.2577 | 0.2416 | 0.2537 | 182.4943 | 2.7752 |
| EleutherAI_deep-ignorance-unfiltered/npo__ep1_lr4e-05_bs32_b0.01_rw1.5_mle512_mli6144 | 0.1115 | 14.8298 | 0.3719 | 0.2604 | 0.2565 | 0.2584 | 0.2702 | 179.3783 | 2.6856 |
| EleutherAI_deep-ignorance-unfiltered/npo__ep3_lr3e-05_bs32_b0.02_rw1.0_ml2048 | 0.1098 | N/A | 0.3413 | 0.2316 | 0.2577 | 0.2612 | 0.2302 | 203.6757 | 2.7988 |
| EleutherAI_deep-ignorance-unfiltered/npo__ep3_lr3e-05_bs32_b0.01_rw1.0_ml4096 | 0.1036 | N/A | 0.3675 | 0.2638 | 0.2676 | 0.2472 | 0.2718 | 207.2005 | 3.1744 |
| EleutherAI_deep-ignorance-unfiltered/npo__ep3_lr4e-05_bs32_b0.01_rw2.0_ml2048 | 0.1025 | N/A | 0.4089 | 0.3065 | 0.2667 | 0.2528 | 0.3134 | 179.9415 | 2.8352 |
| EleutherAI_deep-ignorance-unfiltered/npo__ep3_lr4e-05_bs32_b0.02_rw1.0_ml2048 | 0.1023 | N/A | 0.4330 | 0.3306 | 0.2655 | 0.2602 | 0.3739 | N/A | N/A |
| EleutherAI_deep-ignorance-unfiltered/npo__ep3_lr4e-05_bs32_b0.01_rw1.5_ml2048 | 0.0998 | N/A | 0.3959 | 0.2961 | 0.2647 | 0.2704 | 0.2993 | 180.0890 | 2.5439 |
| EleutherAI_deep-ignorance-unfiltered/npo__ep1_lr5e-05_bs32_b0.015_rw1.25_mle512_mli6144 | 0.0989 | 18.4561 | 0.3950 | 0.2961 | 0.2626 | 0.2435 | 0.3386 | 181.6943 | 2.5063 |
| EleutherAI_deep-ignorance-unfiltered/npo__ep1_lr5e-05_bs32_b0.01_rw2.0_mle512_mli6144 | 0.0929 | 18.3625 | 0.3705 | 0.2776 | 0.2643 | 0.2481 | 0.2891 | 181.3814 | 2.5338 |
| EleutherAI_deep-ignorance-unfiltered/npo__ep3_lr4e-05_bs32_b0.01_rw3.0_ml2048 | 0.0857 | N/A | 0.4175 | 0.3318 | 0.2692 | 0.2454 | 0.3488 | 179.0658 | 2.4161 |
| EleutherAI_deep-ignorance-unfiltered/npo__ep3_lr3e-05_bs16_b0.01_rw1.0_ml2048 | 0.0729 | N/A | 0.3437 | 0.2707 | 0.2643 | 0.2584 | 0.2490 | 220.5241 | 2.8078 |
| EleutherAI_deep-ignorance-unfiltered/npo__ep3_lr3e-05_bs32_b0.01_rw1.0_ml2048 | 0.0728 | N/A | 0.3159 | 0.2431 | 0.2626 | 0.2584 | 0.2349 | 204.6300 | 2.8058 |
| EleutherAI_deep-ignorance-unfiltered/npo__ep4_lr3e-05_bs16_b0.01_rw1.0_ml2048 | 0.0680 | N/A | 0.3330 | 0.2650 | 0.2573 | 0.2760 | 0.2490 | 222.3897 | 2.6125 |
| EleutherAI_deep-ignorance-unfiltered/npo__ep3_lr3e-05_bs32_b0.01_rw1.0_ml1024 | 0.0679 | N/A | 0.3052 | 0.2373 | 0.2372 | 0.2491 | 0.2490 | 196.8414 | 5.7226 |
| EleutherAI_deep-ignorance-unfiltered/npo__ep4_lr3e-05_bs32_b0.01_rw1.0_ml2048 | 0.0650 | N/A | 0.3311 | 0.2661 | 0.2598 | 0.2677 | 0.2569 | 206.2143 | 2.4163 |
| EleutherAI_deep-ignorance-unfiltered/npo__ep3_lr3e-05_bs32_b0.01_rw2.0_ml2048 | 0.0608 | N/A | 0.3258 | 0.2650 | 0.2585 | 0.2342 | 0.2584 | 194.3022 | 2.8449 |
| EleutherAI_deep-ignorance-unfiltered/npo__ep3_lr2e-05_bs32_b0.01_rw1.0_ml2048 | 0.0551 | N/A | 0.3454 | 0.2903 | 0.2557 | 0.2602 | 0.2844 | 216.4733 | 2.4112 |
| EleutherAI_deep-ignorance-unfiltered/npo__ep3_lr3e-05_bs32_b0.01_rw0.75_ml2048 | 0.0255 | N/A | 0.2916 | 0.2661 | 0.2766 | 0.2491 | 0.2490 | 194.7834 | 2.8707 |
| EleutherAI_deep-ignorance-unfiltered/npo__ep3_lr3e-05_bs32_b0.01_rw0.5_ml2048 | 0.0237 | N/A | 0.2887 | 0.2650 | 0.2659 | 0.2574 | 0.2490 | 194.6357 | 2.4974 |
| EleutherAI_deep-ignorance-unfiltered/npo__ep2_lr3e-05_bs32_b0.01_rw5.0_ml2048 | 0.0173 | N/A | 0.4240 | 0.4067 | 0.2565 | 0.2556 | 0.4980 | 161.0679 | 2.4669 |
| EleutherAI_deep-ignorance-unfiltered/npo__ep3_lr3e-05_bs32_b0.005_rw1.0_ml2048 | 0.0133 | N/A | 0.2575 | 0.2442 | 0.2585 | 0.2416 | 0.2396 | 204.9156 | 2.8930 |
| EleutherAI_deep-ignorance-unfiltered/npo__ep2_lr2e-05_bs32_b0.01_rw3.0_ml2048 | 0.0037 | N/A | 0.4242 | 0.4205 | 0.2417 | 0.2677 | 0.5082 | 126.8750 | 2.7236 |

### rmu

| Name | Score | L2 Dist | MMLU | WMDP (Robust) | WMDP (Robust Rewritten) | WMDP (Cloze) | WMDP (Categorized) | Forget NLL | Retain NLL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/rmu__ep1_lr2e-05_bs32_a1000.0_sc20.0_ly11-12-13_ml2048 | 0.0562 | N/A | 0.4030 | 0.3468 | 0.2532 | 0.3615 | 0.4006 | 2.2329 | 3.0996 |
| EleutherAI_deep-ignorance-unfiltered/rmu__ep2_lr1e-05_bs32_a100.0_sc20.0_ly11-12-13_ml2048 | 0.0386 | N/A | 0.4338 | 0.3952 | 0.2503 | 0.3662 | 0.4792 | N/A | N/A |
| EleutherAI_deep-ignorance-unfiltered/rmu__ep2_lr1e-05_bs32_a100.0_sc20.0_ly5-6-7-11-12-13_ml2048 | 0.0336 | N/A | 0.4345 | 0.4009 | 0.2524 | 0.3606 | 0.4847 | N/A | N/A |
| EleutherAI_deep-ignorance-unfiltered/rmu__ep1_lr1e-05_bs32_a100.0_sc20.0_ly5-6-7_ml2048 | 0.0242 | N/A | 0.4504 | 0.4263 | 0.2717 | 0.3634 | 0.5224 | N/A | N/A |
| EleutherAI_deep-ignorance-unfiltered/rmu__ep2_lr1e-05_bs32_a100.0_sc20.0_ly5-6-7_ml2048 | 0.0190 | N/A | 0.4511 | 0.4320 | 0.2713 | 0.3578 | 0.5247 | N/A | N/A |
| EleutherAI_deep-ignorance-unfiltered/rmu__ep2_lr1e-05_bs32_a100.0_sc40.0_ly5-6-7_ml2048 | 0.0170 | N/A | 0.4501 | 0.4332 | 0.2717 | 0.3578 | 0.5255 | N/A | N/A |
| EleutherAI_deep-ignorance-unfiltered/rmu__ep2_lr2e-05_bs32_a200.0_sc20.0_ly11-12-13_ml2048 | -0.0008 | N/A | 0.2458 | 0.2465 | 0.2310 | 0.2472 | 0.2553 | 14.2894 | 18.6446 |
| EleutherAI_deep-ignorance-unfiltered/rmu__ep2_lr1.5e-05_bs32_a200.0_sc20.0_ly11-12-13_ml2048 | -0.0011 | N/A | 0.2835 | 0.2846 | 0.2491 | 0.3392 | 0.2773 | 3.3707 | 7.0862 |
| EleutherAI_deep-ignorance-unfiltered/rmu__ep3_lr3e-05_bs32_a100.0_sc20.0_ly5-6-7_ml2048 | -0.0378 | N/A | 0.2295 | 0.2673 | 0.2651 | 0.2704 | 0.2467 | 29.6285 | 29.6222 |
| EleutherAI_deep-ignorance-unfiltered/rmu__ep3_lr3e-05_bs32_a50.0_sc20.0_ly5-6-7_ml2048 | -0.0378 | N/A | 0.2295 | 0.2673 | 0.2651 | 0.2686 | 0.2467 | 29.6763 | 29.6108 |
| EleutherAI_deep-ignorance-unfiltered/rmu__ep1_lr3e-05_bs32_a300.0_sc20.0_ly11-12-13_ml2048 | -0.0407 | N/A | 0.2300 | 0.2707 | 0.2647 | 0.2667 | 0.2506 | 7.7566 | 15.1785 |
| EleutherAI_deep-ignorance-unfiltered/rmu__ep1_lr3e-05_bs32_a500.0_sc20.0_ly11-12-13_ml2048 | -0.0411 | N/A | 0.2297 | 0.2707 | 0.2647 | 0.2612 | 0.2506 | 7.8395 | 15.1944 |
| EleutherAI_deep-ignorance-unfiltered/rmu__ep2_lr1e-05_bs32_a100.0_sc20.0_ly5-6-7_ml2048 | N/A | N/A | 0.4511 | N/A | 0.2713 | 0.3578 | N/A | 2.1346 | 2.4992 |
| EleutherAI_deep-ignorance-unfiltered/rmu__ep1_lr1e-05_bs32_a100.0_sc20.0_ly5-6-7_ml2048 | N/A | N/A | 0.4504 | N/A | 0.2717 | 0.3634 | N/A | 2.1293 | 2.4806 |
| EleutherAI_deep-ignorance-unfiltered/rmu__ep2_lr1e-05_bs32_a100.0_sc40.0_ly5-6-7_ml2048 | N/A | N/A | 0.4501 | N/A | 0.2717 | 0.3578 | N/A | 2.1346 | 2.4993 |
| EleutherAI_deep-ignorance-unfiltered/rmu__ep2_lr1e-05_bs32_a100.0_sc20.0_ly5-6-7-11-12-13_ml2048 | N/A | N/A | 0.4345 | N/A | 0.2524 | 0.3606 | N/A | 2.1532 | 2.6268 |
| EleutherAI_deep-ignorance-unfiltered/rmu__ep2_lr1e-05_bs32_a100.0_sc20.0_ly11-12-13_ml2048 | N/A | N/A | 0.4338 | N/A | 0.2503 | 0.3662 | N/A | 2.1591 | 2.6727 |

### simnpo

| Name | Score | L2 Dist | MMLU | WMDP (Robust) | WMDP (Robust Rewritten) | WMDP (Cloze) | WMDP (Categorized) | Forget NLL | Retain NLL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/simnpo__ep3_lr3e-05_bs32_b0.01_rw1.0_ml2048 | 0.1780 | N/A | 0.4315 | 0.2535 | 0.2614 | 0.2565 | 0.2616 | 198.9024 | 2.4823 |
| EleutherAI_deep-ignorance-unfiltered/simnpo__ep4_lr3e-05_bs32_b0.01_rw1.0_ml2048 | 0.1650 | N/A | 0.4288 | 0.2638 | 0.2347 | 0.2602 | 0.2765 | 208.9739 | 2.9619 |
| EleutherAI_deep-ignorance-unfiltered/simnpo__ep3_lr3e-05_bs32_b0.01_rw1.0_ml1024 | 0.1278 | N/A | 0.3962 | 0.2684 | 0.2544 | 0.2574 | 0.2687 | 184.6168 | 9.5447 |
| EleutherAI_deep-ignorance-unfiltered/simnpo__ep4_lr3e-05_bs32_b0.01_rw1.0_ml1024 | 0.1117 | N/A | 0.4205 | 0.3088 | 0.2470 | 0.2909 | 0.3331 | 188.5893 | 7.6572 |
| girishgupta_simnpo__ep3_lr3e-05_bs4_b0.01_rw1.0_ml1024 | 0.1105 | N/A | 0.4216 | 0.3111 | 0.2680 | 0.2695 | 0.3401 | N/A | N/A |
| EleutherAI_deep-ignorance-unfiltered/simnpo__ep1_lr3e-05_bs32_b0.01_rw1.0_ml6144 | 0.0602 | N/A | 0.3597 | 0.2995 | 0.2520 | 0.2491 | 0.3315 | 176.0193 | 2.7876 |

### tar

| Name | Score | L2 Dist | MMLU | WMDP (Robust) | WMDP (Robust Rewritten) | WMDP (Cloze) | WMDP (Categorized) | Forget NLL | Retain NLL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/tar__ta5.0_tlr1e-05_tep1_mle512_mli1024 | 0.0315 | 11.1298 | 0.4313 | 0.3998 | 0.2540 | 0.3550 | 0.4894 | 2.6966 | 2.8671 |
| EleutherAI_deep-ignorance-unfiltered/tar__ta1.0_tlr1e-05_tep1_ml1024 | -0.0378 | N/A | 0.2295 | 0.2673 | 0.2651 | 0.2491 | 0.2467 | N/A | N/A |

### wt_dist

| Name | Score | L2 Dist | MMLU | WMDP (Robust) | WMDP (Robust Rewritten) | WMDP (Cloze) | WMDP (Categorized) | Forget NLL | Retain NLL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EleutherAI_deep-ignorance-unfiltered/wt_dist__ep3_lr3e-05_bs32_wn0.01_ml2048 | 0.0615 | N/A | 0.3518 | 0.2903 | 0.2598 | 0.3550 | 0.3543 | 2.4404 | 2.5658 |
| EleutherAI_deep-ignorance-unfiltered/wt_dist__ep3_lr3e-05_bs32_wn0.005_ml2048 | 0.0270 | N/A | 0.4256 | 0.3986 | 0.2598 | 0.3931 | 0.4776 | 2.3608 | 2.4598 |
| EleutherAI_deep-ignorance-unfiltered/wt_dist__ep1_lr2e-05_bs32_wn0.0001_mle512_mli1024 | 0.0152 | 9.2383 | 0.4495 | 0.4343 | 0.2532 | 0.3522 | 0.5263 | 2.1373 | 2.2353 |
| EleutherAI_deep-ignorance-unfiltered/wt_dist__ep3_lr3e-05_bs32_wn0.1_ml2048 | -0.0066 | N/A | 0.2537 | 0.2604 | 0.2343 | 0.2388 | 0.2694 | 6.9821 | 6.8524 |
| EleutherAI_deep-ignorance-unfiltered/wt_dist__ep3_lr3e-05_bs32_wn0.05_ml2048 | -0.0326 | N/A | 0.2301 | 0.2627 | 0.2684 | 0.2435 | 0.2474 | 6.5516 | 6.4134 |
| EleutherAI_deep-ignorance-unfiltered/wt_dist__ep2_lr3e-05_bs32_wn0.02_ml2048 | -0.0359 | N/A | 0.2291 | 0.2650 | 0.2651 | 0.2110 | 0.2443 | 5.5441 | 5.5322 |
| EleutherAI_deep-ignorance-unfiltered/wt_dist__ep3_lr3e-05_bs32_wn0.02_ml2048 | -0.0378 | N/A | 0.2295 | 0.2673 | 0.2651 | 0.2193 | 0.2459 | 5.3104 | 5.3388 |
| EleutherAI_deep-ignorance-unfiltered/wt_dist__ep1_lr3e-05_bs32_wn0.02_ml2048 | -0.0452 | N/A | 0.2290 | 0.2742 | 0.2639 | 0.2361 | 0.2553 | 5.9272 | 6.0225 |

## Cross-Method Comparison — Best Config Per Method

*Best run per method ranked by Score = MMLU − WMDP (Robust)*

| Method | Best Config | L2 Dist | MMLU | WMDP | MMLU-WMDP | Forget NLL | Retain NLL |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cb | ep1_lr1.3e-05_bs16_a200.0_sc10.0_ly13-14-15_mle512_mli1024 | N/A | 0.4425 | 0.4136 | 0.0289 | N/A | N/A |
| cb_lat | ep1_lr1.3e-05_bs16_a1000.0_sc5.0_le0.1_ls5_ly13-14-15_mle512_mli1024 | 2.68 | 0.4430 | 0.4136 | 0.0294 | 2.178 | 2.685 |
| dpo | ep3_lr4e-05_bs32_b0.01_mle512_mli2048 | N/A | 0.3551 | 0.2673 | 0.0879 | N/A | N/A |
| ga | ep3_lr3e-05_bs32_rw5.0_ml2048 | N/A | 0.3558 | 0.2385 | 0.1173 | 199.178 | 2.846 |
| ga_simple | ep1_lr1e-05_bs32_ml1024 | N/A | 0.4471 | 0.4240 | 0.0231 | 2.144 | 2.169 |
| grad_diff | ep3_lr4e-05_bs32_fw1.0_mle512_mli2048 | N/A | 0.3546 | 0.2661 | 0.0885 | 181.136 | 2.760 |
| lat | ep3_lr3e-05_bs32_le0.1_ls5_rw1.0_ly5-6-7_mle512_mli2048 | N/A | 0.2898 | 0.2742 | 0.0156 | 188.541 | 2.687 |
| npo | ep3_lr4e-05_bs32_b0.01_rw1.0_ml2048 | N/A | 0.3813 | 0.2373 | 0.1440 | 205.016 | 2.958 |
| rmu | ep1_lr2e-05_bs32_a1000.0_sc20.0_ly11-12-13_ml2048 | N/A | 0.4030 | 0.3468 | 0.0562 | 2.233 | 3.100 |
| simnpo | ep3_lr3e-05_bs32_b0.01_rw1.0_ml2048 | N/A | 0.4315 | 0.2535 | 0.1780 | 198.902 | 2.482 |
| tar | ta5.0_tlr1e-05_tep1_mle512_mli1024 | 11.13 | 0.4313 | 0.3998 | 0.0315 | 2.697 | 2.867 |
| wt_dist | ep3_lr3e-05_bs32_wn0.01_ml2048 | N/A | 0.3518 | 0.2903 | 0.0615 | 2.440 | 2.566 |
