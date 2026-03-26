# Milestone 2 Evaluation Report

## Overview
This report evaluates a deterministic face verification pipeline for Milestone 2. The baseline system uses grayscale pixel features, cosine similarity, validation-set threshold selection, and tracked evaluation runs. The threshold-selection rule is fixed in advance: choose the threshold that maximizes balanced accuracy on the validation split, then report the locked threshold on the test split.

## Tracked Runs And Data-Centric Change
Five tracked runs are included in [reports/tracked_runs](tracked_runs). Runs 1 to 3 cover the baseline pipeline, including a validation sweep, a locked-threshold run, and a final run chosen to expose a high-similarity false positive slice. Runs 4 and 5 use the Milestone 2 data-centric improvement in [configs/m2_improved.yaml](../configs/m2_improved.yaml), which changes the pair policy to `prefer_unique` so positive pairs are less dominated by repeated same-identity combinations. This is a data-centric change because it changes the composition of the pair dataset while leaving the evaluation code and threshold-selection rule fixed.

## Threshold Selection
The selected-threshold rule is the same for baseline and improved runs: maximize balanced accuracy on the validation split. In the regenerated evidence, the baseline mid-margin run selects threshold 0.50, while the strongly separated runs select threshold 0.01 because all positive scores exceed all negative scores. The ROC-style plot in `reports/roc_baseline_validation.png` shows the validation sweep behavior, and the confusion matrix in `reports/confusion_matrix_baseline_validation.png` shows the locked operating point.

## Baseline Versus Improved Summary
The compact run comparison is in [reports/run_comparison.md](run_comparison.md) and [reports/run_comparison.csv](run_comparison.csv). Under the synthetic fixture used for clean-clone reproducibility, both baseline and improved settings can achieve balanced accuracy 1.00 on separable data. The improved configuration still matters because it changes how positive pairs are sampled, making the dataset less repetitive and more defensible as an evaluation set. The imperfect runs are included deliberately to support error analysis rather than to claim the highest score.

## Error Slice 1: High-Similarity False Positives
Slice definition: different-identity pairs that still score above the selected threshold. Count: 1 example in the baseline error-analysis run. Example images are stored under `reports/error_slices/baseline_error/high_similarity_false_positives/`. Hypothesis: the grayscale flatten-and-cosine baseline is sensitive to coarse shared structure and does not model identity-specific facial detail well enough, so visually similar non-matches can be accepted. Future improvement: replace raw grayscale features with a stronger embedding or add quality-aware negative sampling.

## Error Slice 2: Low-Similarity False Negatives
Slice definition: same-identity pairs that score below the selected threshold. Count: 1 example in the improved error-analysis run. Example images are stored under `reports/error_slices/improved_error/low_similarity_false_negatives/`. Hypothesis: even with improved pair construction, the baseline feature extractor remains brittle to pattern variation analogous to pose, lighting, or expression changes. Future improvement: add a more robust representation and keep the improved pair policy so the threshold is selected on less repetitive data.

## Main Takeaway
The key Milestone 2 result is not just a single metric. The repository now behaves like a reproducible evaluation system: tracked runs are recorded, threshold choice is explicit and validation-based, error slices are inspectable, and a data-centric iteration is compared under the same rule.
