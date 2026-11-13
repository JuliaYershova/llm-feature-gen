# Benchmark for the JMLR MLOSS paper

`benchmark_mloss.ipynb` produced the validation results in the paper, at library commit 21b6408 (0.1.14). The model is set in cell 1 (`MODEL`). The saved outputs in the notebook are from the qwen3.8:27b run.

`results/` holds the outputs of both runs:

| Folder | Model |
|---|---|
| `benchmark_seed46_21b6408/` | qwen3-vl:32b-instruct |
| `benchmark_seed46_21b6408_qwen3.8-27b/` | qwen3.8:27b |

Each folder contains `report.json` (all scores, per-fold values and p-values), `table.csv` (the summary table), `pip_freeze.txt`, and one folder per modality and run with the discovered schema, the generated features, `run.json` and the GPU power logs.
