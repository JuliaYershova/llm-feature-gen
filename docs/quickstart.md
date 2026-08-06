# Quickstart

Goal: your first feature table — a discovered schema plus generated values — from about a dozen lines of Python.

!!! note "Before you start"
    You need an LLM provider. Either put an OpenAI key into a `.env` file in your working directory:

    ```env
    OPENAI_API_KEY=your_api_key
    OPENAI_MODEL=gpt-4.1-mini
    ```

    or run against a local server with no key at all — see [provider configuration](provider-configuration.md). Every discovery and generation call sends data to the configured provider and, for hosted APIs, consumes paid tokens.

## 1. Discover a feature schema

Pass raw texts — a list, a single file, or a folder of documents. The provider proposes features that vary across them:

```python
from llm_feature_gen import discover_features_from_texts

discovered = discover_features_from_texts([
    "The dish was rich, spicy, and served in a deep bowl.",
    "The dessert was light, creamy, and topped with fresh fruit.",
])
```

`discovered` is also saved to `outputs/discovered_text_features.json`:

```json
{
  "proposed_features": [
    {
      "feature": "richness",
      "description": "Whether the food is described as heavy and rich or light",
      "possible_values": ["rich", "light"]
    },
    {
      "feature": "spice level",
      "description": "How spicy the dish appears to be",
      "possible_values": ["spicy", "mild", "sweet"]
    }
  ]
}
```

The exact features vary between runs — the schema is proposed by the model. Once you are happy with a schema, keep the JSON file and reuse it for every later generation run.

## 2. Generate feature values

Generation is folder-based: one subfolder per class. Create a tiny dataset and score it against the discovered schema:

```python
from pathlib import Path

from llm_feature_gen import generate_features_from_texts

samples = {
    "demo_texts/positive/review1.txt": "The meal was vibrant, aromatic, and beautifully plated.",
    "demo_texts/negative/review1.txt": "The service was slow and the food arrived cold.",
}
for name, text in samples.items():
    Path(name).parent.mkdir(parents=True, exist_ok=True)
    Path(name).write_text(text, encoding="utf-8")

csv_paths = generate_features_from_texts("demo_texts", merge_to_single_csv=True)
```

`outputs/all_feature_values.csv` now holds one row per file:

| File | Class | richness | spice level | raw_llm_output |
| --- | --- | --- | --- | --- |
| review1.txt | positive | rich | spicy | `{...}` |
| review1.txt | negative | light | mild | `{...}` |

The `raw_llm_output` column keeps the unparsed provider response for auditing. Drop it — together with `File` and `Class` — before training a model on the feature columns.

## Other modalities

The same two calls exist for images, videos, and tabular files, and any of them accepts an explicit provider:

```python
from llm_feature_gen import discover_features_from_images
from llm_feature_gen.providers import LocalProvider

result = discover_features_from_images("discover_images", provider=LocalProvider())
```

For tabular datasets, name the text-bearing column in both steps:

```python
from llm_feature_gen import discover_features_from_tabular, generate_features_from_tabular

discover_features_from_tabular("discover_tabular/test.csv", text_column="text")
generate_features_from_tabular(
    root_folder="tabular",
    text_column="text",
    label_column="label",
)
```

!!! warning "Sample folders live in the repository, not the package"
    Folder names like `discover_images/` and `tabular/` in these snippets refer to sample data shipped in the [GitHub repository](https://github.com/JuliaYershova/LLM-feature-gen) — a `pip install` does not include them. Point the calls at your own folders, or clone the repository to run the snippets as written.

Next steps: [multiclass and batch workflows](advanced-workflows.md) for more than two classes and cheaper repeated runs, or the [end-to-end example](examples.md) that finishes in a downstream classifier.
