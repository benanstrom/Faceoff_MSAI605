# Milestone 2 Run Comparison

| Scenario | Experiment | Pair Version | Threshold | Balanced Accuracy | Accuracy | TP | FP | TN | FN | Change |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| run1_baseline_validation_sweep | m2_baseline | synthetic_v1_balanced_pairs | 0.01 | 1.00 | 1.00 | 1 | 0 | 1 | 0 | Baseline validation sweep with fully separable positives and negatives. |
| run2_baseline_locked_threshold | m2_baseline | synthetic_v2_midmargin_pairs | 0.50 | 1.00 | 1.00 | 1 | 0 | 1 | 0 | Baseline run where the validation sweep selects threshold 0.50 under the same rule. |
| run3_baseline_test_error_analysis | m2_baseline | synthetic_v3_fp_slice_pairs | 0.01 | 0.50 | 0.50 | 1 | 1 | 0 | 0 | Baseline final run with one high-similarity false positive kept for slice analysis. |
| run4_improved_validation_sweep | m2_improved | synthetic_v4_prefer_unique_pairs | 0.01 | 1.00 | 1.00 | 1 | 0 | 1 | 0 | Improved pair policy using prefer_unique positive pairs with strong separation. |
| run5_improved_test_error_analysis | m2_improved | synthetic_v5_fn_slice_pairs | 0.50 | 0.50 | 0.50 | 0 | 0 | 1 | 1 | Improved pair policy with one low-similarity false negative kept for slice analysis. |
