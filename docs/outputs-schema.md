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
  IDs, support count, and `value_mappings` from each original candidate's values
  to the final canonical vocabulary.
- Provider/configuration fingerprints, repaired evidence IDs, discarded unsupported
  proposals, name conflicts, skipped invalid merges,
  and any shortfall against the corpus-level feature goal.
- Cumulative provider usage when the provider implements `usage_summary()`.

Input IDs preserve text-list positions, document/chunk paths, table file/row
positions, and image/video paths. Duplicate source content remains separately
identified. Evidence and candidate lineage are kept in the report, so generation
requires no changes to its feature schema.
Model requests use neutral hashed item IDs, so class-named folders and source
filenames do not expose item labels. Reports translate evidence back to the
original source identities.
`processed_item_ids` follows map processing order, which is shuffled by default;
the original IDs remain stable. The report's configuration records `shuffle_seed`
(`0` by default, or `None` to preserve collection order).

Only a successfully completed run replaces the schema. Map responses must
acknowledge every input exactly once; candidates need known source evidence.
Reduction merges explicit equivalents with validated value reconciliation and
keeps every other evidenced candidate. Unknown and duplicate evidence references
are filtered with warnings. Proposals left without evidence are retried once,
then discarded explicitly if correction fails. Input coverage remains strict.
A merge supplies a canonical description and `value_mappings`: objects containing
`candidate_id`, `source_value`, and `canonical_value`. Every source value of every
merged candidate, including the survivor, must appear exactly once. The canonical
`possible_values` list is derived from nonempty targets in first-occurrence order,
so each canonical value has source support. Legacy redundant vocabulary lists
are accepted, with mappings taking precedence.
Synonyms and partial vocabularies can be reconciled; the model must leave different
properties and ambiguous meanings separate. Invalid groups are retried once and
then skipped with warnings, preserving their candidates. Accepted merges and
proposals survive retries. Known singleton groups leave candidates unchanged
without triggering a retry.
Accepted merges get a semantic-review warning when a pair of differently named
candidates has no shared source values, ignoring case and surrounding whitespace.
The warning names both candidates with their IDs and the survivor; it does not
reject the merge or request a retry. Disjoint synonyms can be legitimate, and
shared values do not guarantee that two features describe the same property.
Provider-raised response/parsing errors get one retry with validation feedback;
a second failure aborts without replacing the previous schema.

The report stores composed mappings by original map candidate ID, for example:

```json
"value_mappings": {
  "map_0000:feature_0000": {"straight": "straight", "curved": "curved"},
  "map_0001:feature_0000": {"linear": "straight", "curved": "curved"}
}
```

Exact duplicates retain identity mappings. Further merges compose earlier
translations, so each original candidate maps directly to the final vocabulary.
The survivor retains its feature name while its description and values become
the reconciled definition; the generation schema keeps its existing format.

Same-name families are packed separately and consolidated in bounded batches
until no merges occur. A single bounded comparison pass across names is followed
by further same-name consolidation when needed. Unchanged batches are not
repeated. This does not guarantee every pair of semantic duplicates is compared.
Conflicting names receive deterministic numeric suffixes.

An optional `checkpoint_dir` contains run fingerprints and validated map/reduce
responses, including combined accepted results and diagnostics from retries.
Cached results are revalidated and their warnings restored. Discovery checkpoint
version 4 invalidates responses saved under the previous reduction contract.
Inputs, prompts, extraction settings, provider identity, and batch
settings, including the shuffle seed, determine reuse; cached responses are
revalidated. Changing the seed invalidates reuse even if the resulting item
order happens to match. Custom providers
must expose a JSON-serializable `cache_identity` attribute or zero-argument
method covering their response settings. Otherwise checkpoint reuse is disabled
and reported. A `cache_identity()` method returning `None` explicitly disables
checkpoint reuse, matching generation-cache behavior. Interrupted runs retain completed checkpoints and any previous
successful schema.
