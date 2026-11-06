# Examples

## Exhaustive Map-Reduce Discovery

`map_reduce_discovery.py` processes all five example inputs in batches of two.
Its offline toy provider demonstrates exact deduplication, retention of a
feature found only in the last batch, reports, and checkpoint reuse:

```bash
python examples/map_reduce_discovery.py --provider offline --output-dir /tmp/map_reduce_demo
```

Run with `--provider local` or `--provider openai` to use a configured LLM instead.
Offline proposals are illustrative; they are not a feature-quality benchmark.

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
