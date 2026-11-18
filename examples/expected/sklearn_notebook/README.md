# Recorded run of `examples/sklearn_pipeline.ipynb`

The output of the notebook run reported in the paper's worked example (model `qwen3-vl:32b-instruct`, 90 training and 30 test documents).

- `result.json`: model, split sizes, the ten feature names, time for discovery and generation, and the accuracy of the generated features and of the TF-IDF baseline on the 30 test documents.
- `discovered_features.json`: the feature schema discovered on the 90 training documents.
- `feature_table.csv`: one row per document (120), with the ten generated features, `split`, `label` and the post text.

These files are a record, not test fixtures: no test replays them. Feature discovery is not deterministic, so running the notebook again overwrites them and can give different features and numbers. For an offline run, use `python3 examples/sklearn_pipeline.py --provider replay --check`.
