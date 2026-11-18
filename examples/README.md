# Examples

## Canonical Text-to-Tabular Pipeline

The repository now includes one publishable end-to-end example:

- Script: `examples/text_to_tabular_pipeline.py`
- Raw inputs: `examples/data/text_to_tabular/`
- Checked-in expected artifacts: `examples/expected/text_to_tabular_pipeline/`

Run it from the repository root with a real provider:

```bash
python3 examples/text_to_tabular_pipeline.py --provider auto
```

If you want the fully offline reproducibility path used by tests, run:

```bash
python3 examples/text_to_tabular_pipeline.py --provider replay --check
```

What it does:

1. Reads a tiny support-ticket text corpus.
2. Discovers an interpretable schema JSON.
3. Generates one CSV per class folder.
4. Merges those CSVs into a single tabular dataset.
5. Runs a simple downstream leave-one-out nearest-centroid classifier.

The canonical path uses the actual provider stack selected from your environment. The `replay` mode is only there to make the same example verifiable offline in tests and for artifact checking.

## scikit-learn Example

`LLMFeatureTransformer` used as an ordinary scikit-learn transformer, on the
classic hard pair from 20 Newsgroups: `comp.sys.mac.hardware` against
`comp.sys.ibm.pc.hardware`.

- Script: `examples/sklearn_pipeline.py`
- Notebook: `examples/sklearn_pipeline.ipynb`
- Raw inputs: `examples/data/newsgroups_hardware/posts.csv`, built by
  `examples/tools/build_newsgroups_snapshot.py` (headers, footers and quoted replies
  removed; posts with email addresses or telephone numbers left out; repeated posts
  kept once)
- Checked-in expected artifacts: `examples/expected/sklearn_pipeline/`
- Recorded run of the notebook, reported in the paper: `examples/expected/sklearn_notebook/`

Run the script from the repository root with a real provider:

```bash
python3 examples/sklearn_pipeline.py --provider auto
```

Or reproduce the checked-in numbers offline, with no API key:

```bash
python3 examples/sklearn_pipeline.py --provider replay --check
```

What the script does:

1. Reuses the pinned feature schema, or discovers one with `--rediscover`.
2. Fits the pipeline on a training split and predicts the held-out one.
3. Scores the same pipeline by 5-fold cross-validation against a TF-IDF baseline.
4. Writes the feature values every row was classified from to `feature_table.csv`.

Offline runs replay recorded model answers from
`examples/expected/sklearn_pipeline/responses.json`, keyed by post id. The model that
produced them is stored in `result.json`, and the test suite runs this replay. `--record`
overwrites that file from a live run. Keying by post rather than by a cache key
derived from the prompt is deliberate: editing a prompt must not invalidate the
recordings a reviewer needs to reproduce the result.

The notebook is the short version, without the replay machinery: split the posts
into training and test sets, `fit_transform` the training posts (features are
discovered from the training posts only), `transform` the test posts, then one-hot
encode and train a logistic regression, and compare it with a TF-IDF logistic
regression on the same split. It uses `LocalProvider`, a local OpenAI-compatible
endpoint configured through the `LOCAL_*` variables, and needs a running model; it
does not replay, and feature discovery is not deterministic, so a new run can give
different features and numbers. The numbers reported in the paper come from the
notebook's recorded run in `examples/expected/sklearn_notebook/`, which holds its
result, the discovered schema and the per-document feature table. Running the
notebook again overwrites those files.
