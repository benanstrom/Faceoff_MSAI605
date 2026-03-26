# Milestone 2 Report Artifacts

This directory contains the generated deliverables for Milestone 2:

- `milestone2_report.pdf`: submission-ready 2-page report artifact
- `milestone2_report.md`: source text for the report
- `run_comparison.csv` and `run_comparison.md`: comparison table across five tracked runs
- `tracked_runs_summary.json`: compact summary with run purpose, thresholds, metrics, commit hash, and pair fingerprints
- `tracked_runs/`: lightweight copied run evidence (`run.json`, `metrics.json`, `threshold_sweep.json`, `scores.json`) for the five reported runs
- `roc_*.png`: representative ROC plots from baseline and improved runs
- `confusion_matrix_*.png`: representative confusion matrices from baseline and improved runs
- `error_slices/`: copied example images and `slices.json` summaries for imperfect runs
- `fixtures/`: deterministic synthetic datasets used to generate the tracked runs bundled in this repo
- `report_manifest.json`: index of the generated runs and report assets

Tracked run directories referenced by the report live under `../artifacts/runs/`.
