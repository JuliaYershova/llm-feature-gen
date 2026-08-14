# Quickstart

Turn raw text into a trained, interpretable classifier in about ten lines — then keep reading for the feature table underneath it.

## Install

```bash
pip install llm-feature-gen
```

The pipeline below also composes scikit-learn estimators. If you do not already have scikit-learn, pull it in with the extra:

```bash
pip install "llm-feature-gen[sklearn]"
```

!!! note "You need an LLM provider"
    Put an OpenAI key into a `.env` file in your working directory:

    ```env
    OPENAI_API_KEY=your_api_key
    OPENAI_MODEL=gpt-4.1-mini
    ```

    Or run against a local server with no key at all — see [provider configuration](provider-configuration.md). Every call sends data to the configured provider and, for hosted APIs, consumes paid tokens.

## Your first pipeline

`LLMFeatureTransformer` is a standard scikit-learn transformer. Drop it in front of any estimator and the rest of your pipeline never has to know an LLM was involved:

```python
from llm_feature_gen import LLMFeatureTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

texts = [
    "I was charged twice for a single order.",
    "The invoice total does not match my receipt.",
    "I cannot reset my password.",
    "Login fails with a two-factor error.",
]
labels = ["billing", "billing", "access", "access"]

model = make_pipeline(
    LLMFeatureTransformer(),
    OneHotEncoder(handle_unknown="ignore"),
    LogisticRegression(),
).fit(texts, labels)

print(model.predict(["My card was billed the wrong amount."]))
# ['billing']
```

That is the whole integration. `fit` asks the LLM which features distinguish your texts, then fills those features in for every sample; `OneHotEncoder` turns the resulting categories into numbers; `LogisticRegression` does the rest.

## See what the model actually saw

The transformer works on its own — on a plain install, with no scikit-learn required — and this is where the package earns its keep. The intermediate representation is a table you can read:

```python
features = LLMFeatureTransformer().fit_transform(texts)
print(features)
```

| issue category | urgency | sentiment |
| --- | --- | --- |
| billing | medium | frustrated |
| billing | low | neutral |
| access | high | frustrated |
| access | medium | neutral |

Every column has a name a domain expert can question, an auditor can check, and a decision tree can split on. Compare that with an embedding vector, where column 47 means nothing to anyone.

!!! warning "Pin the schema before cross-validation"
    By default each `fit()` discovers a fresh schema, which costs an LLM call and can propose different features between runs. For `cross_val_score`, `GridSearchCV`, or anything else that fits repeatedly, discover once and pass the saved schema:

    ```python
    LLMFeatureTransformer(discovered_features="outputs/discovered_text_features.json")
    ```

    See [reproducibility](reproducibility.md) for why this matters beyond cost.

### Useful transformer options

| Argument | Default | What it does |
| --- | --- | --- |
| `discovered_features` | `None` | Schema dict or path to a JSON file. `None` discovers a new schema on every `fit`. |
| `text_column` | `None` | Which column holds the text when `X` is a multi-column DataFrame. |
| `use_batch` | `True` | Group samples into one request per batch instead of one call per row. |
| `batch_size` | `8` | Samples per provider call when batching. |
| `cache` | `None` | A `BatchTextCache` so repeated runs over the same texts are free. |
| `provider` | `None` | Any provider instance; defaults to `OpenAIProvider()`. |
| `output_dir` | `"outputs"` | Where `fit` writes the schema it discovers. |

The transformer accepts a list of strings, a pandas `Series`, or a DataFrame (name the column with `text_column`). It returns a DataFrame of string categories, so keep an encoder between it and any numeric estimator. Inputs the model declines to score come back as `"not given by LLM"` rather than raising.

Whenever `fit` discovers a schema it also saves it to `outputs/discovered_text_features.json`, replacing any previous file. That saved file is what you hand back as `discovered_features` to pin the pipeline.

## Working with the feature table directly

Not every project ends in a scikit-learn pipeline. The two underlying steps are plain functions, and they write CSV and JSON files you can inspect, edit, and commit.

### 1. Discover a feature schema

Pass raw texts — a list, a single file, or a folder of documents. The provider proposes features that vary across them:

```python
from llm_feature_gen import discover_features_from_texts

discovered = discover_features_from_texts([
    "I was charged twice for a single order.",
    "Login fails with a two-factor error.",
])
```

`discovered` is also saved to `outputs/discovered_text_features.json`:

```json
{
  "proposed_features": [
    {
      "feature": "issue category",
      "description": "The subsystem the ticket concerns",
      "possible_values": ["billing", "access", "delivery"]
    },
    {
      "feature": "urgency",
      "description": "How quickly the ticket appears to need a response",
      "possible_values": ["high", "medium", "low"]
    }
  ]
}
```

The exact features vary between runs — the schema is proposed by the model. Once you are happy with a schema, keep the JSON file and reuse it for every later generation run.

### 2. Generate feature values

Generation is folder-based: one subfolder per class. Create a tiny dataset and score it against the discovered schema:

```python
from pathlib import Path

from llm_feature_gen import generate_features_from_texts

samples = {
    "demo_tickets/billing/ticket1.txt": "The invoice total does not match my receipt.",
    "demo_tickets/access/ticket1.txt": "I cannot reset my password.",
}
for name, text in samples.items():
    Path(name).parent.mkdir(parents=True, exist_ok=True)
    Path(name).write_text(text, encoding="utf-8")

csv_paths = generate_features_from_texts("demo_tickets", merge_to_single_csv=True)
```

Each subfolder name becomes both the `Class` value and the per-class CSV filename, so this run writes `outputs/billing_feature_values.csv`, `outputs/access_feature_values.csv`, and the merged `outputs/all_feature_values.csv`:

| File | Class | issue category | urgency | raw_llm_output |
| --- | --- | --- | --- | --- |
| ticket1.txt | billing | billing | medium | `{...}` |
| ticket1.txt | access | access | high | `{...}` |

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

`LLMFeatureTransformer` covers text only. For the other modalities, generate the CSV and load it into your pipeline from there.

!!! warning "Sample folders live in the repository, not the package"
    Folder names like `discover_images/` and `tabular/` in these snippets refer to sample data shipped in the [GitHub repository](https://github.com/JuliaYershova/LLM-feature-gen) — a `pip install` does not include them. Point the calls at your own folders, or clone the repository to run the snippets as written.

Next steps: [multiclass and batch workflows](advanced-workflows.md) for more than two classes and cheaper repeated runs, or the [end-to-end example](examples.md) that finishes in a downstream classifier.
