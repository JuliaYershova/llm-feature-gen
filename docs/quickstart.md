# Quickstart

This example creates a tiny text dataset, discovers a shared schema, and then generates per-class feature values.

```python
from pathlib import Path

from llm_feature_gen import (
    discover_features_from_texts,
    generate_features_from_texts,
)

discover_samples = {
    "demo_discover_texts/sample1.txt": "The dish was rich, spicy, and served in a deep bowl.",
    "demo_discover_texts/sample2.txt": "The dessert was light, creamy, and topped with fresh fruit.",
}

generation_samples = {
    "demo_texts/positive/review1.txt": "The meal was vibrant, aromatic, and beautifully plated.",
    "demo_texts/negative/review1.txt": "The service was slow and the food arrived cold.",
}

for relative_path, text in {**discover_samples, **generation_samples}.items():
    path = Path(relative_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

discovered = discover_features_from_texts("demo_discover_texts")
csv_paths = generate_features_from_texts(
    root_folder="demo_texts",
    merge_to_single_csv=True,
)

print(discovered)
print(csv_paths)
```

Expected outputs:

- `outputs/discovered_text_features.json`
- `outputs/positive_feature_values.csv`
- `outputs/negative_feature_values.csv`
- `outputs/all_feature_values.csv`

## Common patterns

Use a provider explicitly when you want to switch backends:

```python
from llm_feature_gen import discover_features_from_images
from llm_feature_gen.providers import LocalProvider

provider = LocalProvider()
result = discover_features_from_images(
    image_paths_or_folder="discover_images",
    provider=provider,
)
```

For tabular datasets, pass the text-bearing column during both discovery and generation:

```python
from llm_feature_gen import (
    discover_features_from_tabular,
    generate_features_from_tabular,
)

discover_features_from_tabular("discover_tabular/test.csv", text_column="review")
generate_features_from_tabular(
    root_folder="tabular",
    text_column="review",
    label_column="label",
)
```
