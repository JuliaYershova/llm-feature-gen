# Text-to-Tabular Pipeline

One complete run of the library, from raw text files to a trained classifier — small enough to read in ten minutes, checked in CI so it cannot rot.

- Script: [`examples/text_to_tabular_pipeline.py`](https://github.com/JuliaYershova/LLM-feature-gen/blob/main/examples/text_to_tabular_pipeline.py)
- Data: a tiny support-ticket corpus in `examples/data/text_to_tabular/`, split into `routine` and `urgent` tickets
- Expected artifacts: `examples/expected/text_to_tabular_pipeline/`

## Run it

From a repository checkout, with a provider configured:

```bash
python examples/text_to_tabular_pipeline.py --provider auto
```

`--provider` accepts `auto`, `openai`, `local`, or `replay`. The `replay` mode substitutes checked-in provider responses, so this command works offline, with no credentials, and always produces identical artifacts:

```bash
python examples/text_to_tabular_pipeline.py --provider replay --check
```

`--check` compares every generated file against the expected artifacts — the same comparison the test suite runs in CI.

## What it does

1. Reads the discovery samples and proposes an interpretable feature schema.
2. Generates one feature-value CSV per ticket class (`routine`, `urgent`).
3. Merges them into a single tabular dataset.
4. Trains and evaluates a leave-one-out nearest-centroid classifier on the feature columns.
5. Writes predictions and an accuracy report.

Outputs land in `outputs/text_to_tabular_pipeline/`:

```text
discovered_text_features.json    # the schema the LLM proposed
routine_feature_values.csv       # per-class feature values
urgent_feature_values.csv
all_feature_values.csv           # merged dataset
classifier_predictions.csv       # per-ticket predictions
classifier_report.json           # accuracy summary
```

The classifier is deliberately simple — the point of the example is that the feature table it trains on is readable: you can open `all_feature_values.csv` and see *why* a ticket looks urgent.

## Tutorial notebook

For an interactive walk through all four modalities — text, images, tabular, video — and provider switching, open [`tutorial.ipynb`](https://github.com/JuliaYershova/LLM-feature-gen/blob/main/tutorial.ipynb) from a repository checkout. It uses the bundled sample folders, which are not part of the pip package.
