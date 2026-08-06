# Multiclass and Batch Workflows

Two extensions of the basic text workflow: discovery and generation for more than two classes, and batched generation with an on-disk cache for larger datasets.

## More than two classes

The default discovery prompt assumes two hidden categories. When your dataset has more, build a class-aware prompt by passing the class list:

```python
from llm_feature_gen import discover_features_multiclass, generate_features_multiclass

classes = ["billing", "technical", "account"]

discover_features_multiclass(
    texts_or_file="discover_texts",
    classes=classes,
)

generate_features_multiclass(
    root_folder="texts",
    discovered_features="outputs/discovered_text_features.json",
    classes=classes,
)
```

`discover_features_multiclass` asks the provider for features that separate exactly your classes (at least `3 × len(classes)` features by default). `generate_features_multiclass` verifies that every class folder exists before spending any provider calls, and otherwise behaves like [generate_features][llm_feature_gen.generate.generate_features].

### Train/test splits in one call

`run_multiclass_pipeline` chains three stages on one shared schema, so the train and test CSVs have identical columns:

```python
from llm_feature_gen import run_multiclass_pipeline
from llm_feature_gen.providers import OpenAIProvider

results = run_multiclass_pipeline(
    discover_folder="discover_texts",
    train_folder="texts/train",
    test_folder="texts/test",
    classes=["billing", "technical", "account"],
    provider=OpenAIProvider(),
)
```

Output layout under `output_dir` (default `outputs/`):

```text
outputs/
  discovered_text_features.json      # shared schema
  train_generated/
    <class>_feature_values.csv       # one per class
    train_feature_values.csv         # merged
  test_generated/
    <class>_feature_values.csv
    test_feature_values.csv
```

The return value mirrors this: `results["discovered_features"]`, `results["train_csv_paths"]`, and `results["test_csv_paths"]`, where both path mappings include a `"__merged__"` entry.

## Batched generation with caching

Scoring one file per provider call is simple but slow and, on hosted APIs, expensive to repeat. `generate_features_from_texts_cached` batches texts into grouped calls and caches every response on disk:

```python
from llm_feature_gen import generate_features_from_texts_cached

csv_paths = generate_features_from_texts_cached(
    root_folder="texts",
    discovered_features_path="outputs/discovered_text_features.json",
    batch_size=8,
)
```

The cache (default `outputs/feature_cache.json`) is keyed by the text content and a hash of the feature schema. Re-running after an interruption, or adding new files to a class folder, only pays for texts that were not scored before. Changing the schema invalidates the cache naturally, because the hash no longer matches.

For in-memory data — when your texts are already in a list or a DataFrame column — skip the folder layout entirely:

```python
from llm_feature_gen import BatchTextCache, generate_features_batch

df = generate_features_batch(
    texts=["I am sad", "I am scared"],
    labels=["sadness", "fear"],
    discovered_features="outputs/discovered_text_features.json",
    cache=BatchTextCache(),
)
```

This returns a DataFrame with the same columns as the CSV outputs (`File`, `Class`, one column per feature, `raw_llm_output`). A failed batch is retried once and then skipped, leaving `"not given by LLM"` in the affected rows — check for that value before training.

Full signatures: [multiclass API](api/multiclass.md) and [batch API](api/batch.md).
