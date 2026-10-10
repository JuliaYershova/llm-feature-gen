# Outputs and Schema Reference

The library writes two main artifact types:

- Discovery JSON in `outputs/`
- Generation CSV files in `outputs/`

## Discovery JSON

The discovery helpers always write the raw provider result list to disk, even when the Python return value is simplified to a single dictionary in joint mode.

Typical path names:

- `outputs/discovered_image_features.json`
- `outputs/discovered_text_features.json`
- `outputs/discovered_tabular_features.json`
- `outputs/discovered_video_features.json`

Typical joint-discovery structure:

```json
[
  {
    "proposed_features": [
      {
        "feature": "spice level",
        "type": "categorical",
        "description": "How spicy the dish appears or is described to be"
      },
      {
        "feature": "presentation style",
        "type": "categorical"
      }
    ]
  }
]
```

Notes:

- The package expects a `proposed_features` collection when loading a schema for generation.
- Each feature entry is provider-defined. Common keys are `feature`, `name`, `type`, and `description`.
- Per-item discovery writes one list entry per input item instead of a single shared schema.
- For video discovery with `as_set=False`, the package pools frames across all input videos and writes one result per extracted frame, not one result per video.
- Folder-based video discovery samples at most `max_videos_to_sample` videos before extraction; pass `random_seed` to make that subset reproducible.

## Generation CSV

Generation creates one CSV per class folder, named `<class_name>_feature_values.csv`.

Column layout:

| Column | Meaning |
| --- | --- |
| `File` | Source file name, or `filename__row_<n>` for tabular row-level outputs |
| `Class` | Class folder name, or row-level label override when `label_column` is provided |
| `<feature columns>` | One column per discovered feature |
| `raw_llm_output` | Raw JSON payload returned by the provider for traceability |

Example:

```csv
File,Class,spice level,presentation style,raw_llm_output
review1.txt,positive,high,refined,"{""features"": {""spice level"": ""high"", ""presentation style"": ""refined""}}"
```

If `merge_to_single_csv=True`, the package also writes `outputs/all_feature_values.csv` unless you override `merged_csv_name`.

## Schema loading rules

[`load_discovered_features`](api/generate.md) normalizes these cases into one dictionary shape:

- a dictionary that already contains `proposed_features`
- a single-item list containing that dictionary
- a list of feature entries without the outer dictionary, which is wrapped automatically

This means generation code can rely on a single in-memory schema form even when provider outputs vary slightly.

## Map-reduce discovery reports and checkpoints

`strategy="map_reduce"` writes the same single-item list schema used by generation,
plus `<schema_stem>_discovery_report.json` alongside it. The report includes:

- Eligible input counts, processed IDs, and empty/null/unsupported exclusions.
- Map candidate and final feature counts, reduction rounds, and call/cache counts.
- Each final feature's original name, source candidate IDs, unique evidence item
  IDs, and support count.
- Provider/configuration fingerprints, name conflicts, skipped invalid merges,
  and any shortfall against the corpus-level feature goal.
- Cumulative provider usage when the provider implements `usage_summary()`.

Input IDs preserve text-list positions, document/chunk paths, table file/row
positions, and image/video paths. Duplicate source content remains separately
identified. Evidence and candidate lineage are kept in the report, so generation
requires no changes to its feature schema.
Model requests use neutral hashed item IDs, so class-named folders and source
filenames do not expose item labels. Reports translate evidence back to the
original source identities.

Only a successfully completed run replaces the schema. Map responses must
acknowledge every input exactly once; candidates need known source evidence.
Reduction merges explicit equivalents with matching value vocabularies and keeps
every other candidate. Two bounded sweeps limit reduction cost; they do not
guarantee that every pair of semantic duplicates is compared. Conflicting names
receive deterministic numeric suffixes.

An optional `checkpoint_dir` contains run fingerprints and validated map/reduce
responses. Inputs, prompts, extraction settings, provider identity, and batch
settings determine reuse; cached responses are revalidated. Custom providers
must expose a JSON-serializable `cache_identity` attribute or zero-argument
method covering their response settings. Otherwise checkpoint reuse is disabled
and reported. Interrupted runs retain completed checkpoints and any previous
successful schema.
