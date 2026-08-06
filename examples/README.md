# Examples

## Text-to-Tabular Pipeline

One complete run of the library, from raw text files to a trained classifier:

- Script: `examples/text_to_tabular_pipeline.py`
- Raw inputs: `examples/data/text_to_tabular/`
- Checked-in expected artifacts: `examples/expected/text_to_tabular_pipeline/`

Run it from the repository root with a real provider:

```bash
python3 examples/text_to_tabular_pipeline.py --provider auto
```

Or fully offline, replaying checked-in provider responses — the same path the test suite verifies in CI:

```bash
python3 examples/text_to_tabular_pipeline.py --provider replay --check
```

What it does:

1. Reads a tiny support-ticket text corpus.
2. Discovers an interpretable schema JSON.
3. Generates one CSV per class folder.
4. Merges those CSVs into a single tabular dataset.
5. Trains and evaluates a leave-one-out nearest-centroid classifier on the feature columns.

The `replay` mode exists so the example runs without credentials and always produces identical artifacts; use `auto`, `openai`, or `local` to run it against your configured provider stack. A longer walkthrough is on the docs site under [Examples](https://juliayershova.github.io/llm-feature-gen/examples/).
