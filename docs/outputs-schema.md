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
