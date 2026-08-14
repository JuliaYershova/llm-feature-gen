# LLM Feature Gen

[![PyPI version](https://img.shields.io/pypi/v/llm-feature-gen)](https://pypi.org/project/llm-feature-gen/)
[![Tests](https://github.com/JuliaYershova/LLM-feature-gen/actions/workflows/tests.yml/badge.svg)](https://github.com/JuliaYershova/LLM-feature-gen/actions/workflows/tests.yml)
[![Docs](https://github.com/JuliaYershova/LLM-feature-gen/actions/workflows/docs.yml/badge.svg)](https://juliayershova.github.io/llm-feature-gen/)
[![Codecov](https://codecov.io/gh/JuliaYershova/LLM-feature-gen/graph/badge.svg?token=BHLNPPOZUH)](https://codecov.io/gh/JuliaYershova/LLM-feature-gen)
[![License](https://img.shields.io/github/license/JuliaYershova/LLM-feature-gen)](LICENSE)

`llm-feature-gen` converts unstructured text, images, tabular files, and video into interpretable tabular features using large language models.

A language model first proposes a feature schema from a sample of the corpus, then assigns values to every input. Each input becomes one row and each feature one named column, so the result is an ordinary design matrix that any tabular estimator can consume:

| input | issue category | urgency | sentiment |
| --- | --- | --- | --- |
| "I was charged twice for a single order." | billing | medium | frustrated |
| "Login fails with a two-factor error." | access | high | frustrated |
| "Could you confirm my invoice was received?" | billing | low | neutral |

Unlike a dense embedding, every column carries a name a domain expert can interpret and audit. Unlike zero-shot classification, the output is a reusable dataset rather than a single label: the schema is discovered once, saved as JSON, and can be pinned for all later runs.

Full documentation: https://juliayershova.github.io/llm-feature-gen/

## Quickstart

```bash
pip install llm-feature-gen
```

The default provider is OpenAI — put a key into a `.env` file in your working directory (see [Providers](#providers) for Azure and local, key-free options):

```env
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-4.1-mini
```

For text, `LLMFeatureTransformer` is a standard scikit-learn transformer: it takes raw texts and returns a table of named features, so it can be placed in a `Pipeline` ahead of any estimator. Composing it with scikit-learn's own estimators requires scikit-learn:

```bash
pip install "llm-feature-gen[sklearn]"
```

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

`fit` asks the LLM which features distinguish your texts, then fills them in for every sample. Nothing downstream needs to know an LLM was involved — and unlike an embedding, the intermediate representation is a table you can read. The transformer works standalone on a plain install, with no scikit-learn required:

```python
print(LLMFeatureTransformer().fit_transform(texts))
```

| issue category | urgency | sentiment |
| --- | --- | --- |
| billing | medium | frustrated |
| billing | low | neutral |
| access | high | frustrated |
| access | medium | neutral |

The transformer returns string categories, so keep an encoder between it and any numeric estimator.

> **Pin the schema before cross-validation.** Each `fit()` discovers a fresh schema by default, which costs an LLM call and can propose different features between runs. For `cross_val_score` or `GridSearchCV`, discover once and reuse the file: `LLMFeatureTransformer(discovered_features="outputs/discovered_text_features.json")`.

`LLMFeatureTransformer` covers text. For images, video, and tabular files — or when you want CSV files rather than an in-memory pipeline — use the two-step workflow below.

## The Two-Step Workflow

Underneath the transformer are two plain function calls that you can use directly:

1. Discover a feature schema from example inputs.
2. Generate feature values for class-organized data and save CSV files.

```python
from pathlib import Path

from llm_feature_gen import discover_features_from_texts, generate_features_from_texts

samples = {
    "demo_discover/sample1.txt": "I was charged twice for a single order.",
    "demo_discover/sample2.txt": "Login fails with a two-factor error.",
    "demo_tickets/billing/ticket1.txt": "The invoice total does not match my receipt.",
    "demo_tickets/access/ticket1.txt": "I cannot reset my password.",
}

for file_name, text in samples.items():
    path = Path(file_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

discovered = discover_features_from_texts("demo_discover")
csv_paths = generate_features_from_texts(
    root_folder="demo_tickets",
    merge_to_single_csv=True,
)

print(discovered)
print(csv_paths)
```

Expected outputs:

- `outputs/discovered_text_features.json`
- `outputs/billing_feature_values.csv`
- `outputs/access_feature_values.csv`
- `outputs/all_feature_values.csv`

Generation expects one subfolder per class, and each subfolder name becomes both the `Class` column and the CSV filename:

```text
demo_tickets/
  billing/
    ticket1.txt
  access/
    ticket1.txt
```

Every generated CSV includes `File`, `Class`, one column per discovered feature, and `raw_llm_output` for auditing the original provider response.

## Capabilities

| Modality | File formats | Discover | Generate | Batched + cached | Multiclass helpers | sklearn transformer |
| --- | --- | :-: | :-: | :-: | :-: | :-: |
| Text | `.txt`, `.md`, `.pdf`, `.docx`, `.html` | ✓ | ✓ | ✓ | ✓ | ✓ |
| Images | `.jpg`, `.jpeg`, `.png` | ✓ | ✓ | — | — | — |
| Tabular | `.csv`, `.xlsx`, `.xls`, `.parquet`, `.json` | ✓ | ✓ | — | — | — |
| Video | `.mp4`, `.mov`, `.avi`, `.mkv` | ✓ | ✓ | — | — | — |

## Installation

Everything installs from PyPI; each row below is needed only for the capability next to it.

| Command | Needed for |
| --- | --- |
| `pip install llm-feature-gen` | The package, all four modalities, and `LLMFeatureTransformer` on its own |
| `pip install "llm-feature-gen[sklearn]"` | Composing the transformer with scikit-learn's estimators in a `Pipeline` |
| `pip install pypdf python-docx beautifulsoup4 openpyxl xlrd pyarrow` | PDF, DOCX, HTML, XLSX, XLS, and Parquet inputs |
| `pip install faster-whisper` | Local audio transcription through `LocalProvider` |
| `ffmpeg` system binary | Extracting audio from video files |

Quote the `[sklearn]` argument: square brackets are glob characters in zsh, the default macOS shell, so the unquoted form fails before pip runs.

Supported targets are CPython 3.9, 3.11, and 3.13 on Linux, macOS, and Windows. See [SUPPORT.md](SUPPORT.md) for the current support matrix.

## Providers

Every discovery and generation call goes through a provider, configured with a `.env` file in the directory where you run your script. Use `OpenAIProvider` for a hosted backend, or `LocalProvider` for open-source-first and offline-friendly experiments.

### OpenAI and Azure OpenAI

`OpenAIProvider` is the default when no provider is passed. It auto-detects Azure mode when `AZURE_OPENAI_ENDPOINT` is set; otherwise it uses the standard OpenAI API.

For OpenAI, extending the two variables from the [Quickstart](#quickstart) with an audio model:

```env
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-4.1-mini
OPENAI_AUDIO_MODEL=whisper-1
```

For Azure OpenAI:

```env
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_API_VERSION=your_api_version
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_GPT41_DEPLOYMENT_NAME=your_chat_deployment
AZURE_OPENAI_WHISPER_DEPLOYMENT=your_audio_deployment
```

### Local Provider Setup

`LocalProvider` talks to any server that exposes an OpenAI-compatible chat-completions API, including Ollama, vLLM, LM Studio, and llama.cpp-compatible gateways.

The provider has separate model settings for text and vision:

| Variable | Default | Purpose |
| --- | --- | --- |
| `LOCAL_OPENAI_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible local API endpoint |
| `LOCAL_OPENAI_API_KEY` | `ollama` | Placeholder key required by the OpenAI SDK; many local servers ignore it |
| `LOCAL_MODEL_TEXT` | `llama3` | Model used for text discovery and generation |
| `LOCAL_MODEL_VISION` | `llava` | Model used for image and video-frame workflows |
| `LOCAL_WHISPER_MODEL_SIZE` | `base` | Faster-Whisper model size for local audio transcription |
| `LOCAL_WHISPER_DEVICE` | `cpu` | `cpu`, `cuda`, or `auto` for Faster-Whisper |

#### Ollama Example

Install and start Ollama, then pull one text model and one vision model:

```bash
ollama pull llama3
ollama pull llava
ollama serve
```

Use this `.env`:

```env
LOCAL_OPENAI_BASE_URL=http://localhost:11434/v1
LOCAL_OPENAI_API_KEY=ollama
LOCAL_MODEL_TEXT=llama3
LOCAL_MODEL_VISION=llava
LOCAL_WHISPER_MODEL_SIZE=base
LOCAL_WHISPER_DEVICE=cpu
```

Then pass the provider explicitly. Every entry point accepts one, and discovery and generation must both receive it:

```python
from llm_feature_gen import LLMFeatureTransformer, discover_features_from_texts, generate_features_from_texts
from llm_feature_gen.providers import LocalProvider

provider = LocalProvider()

LLMFeatureTransformer(provider=provider)
discover_features_from_texts("discover_texts", provider=provider)
generate_features_from_texts("texts", provider=provider)
```

#### LM Studio or vLLM

Point `LOCAL_OPENAI_BASE_URL` at the server's OpenAI-compatible `/v1` endpoint and set the model names to the identifiers exposed by that server:

```env
LOCAL_OPENAI_BASE_URL=http://localhost:8000/v1
LOCAL_OPENAI_API_KEY=local
LOCAL_MODEL_TEXT=your-text-model
LOCAL_MODEL_VISION=your-vision-model
```

#### Video and Audio Locally

Video workflows extract frames and use `LOCAL_MODEL_VISION` for visual analysis. With `use_audio=True`, `LocalProvider` transcribes through `faster-whisper`, which is the one optional install this path adds.

#### Reproducibility Notes

For papers and benchmarks, report the local backend, model names, model versions or hashes, quantization, hardware, and decoding settings. The package defaults to `temperature=0.0` in both built-in providers, but local servers can still differ in tokenizer, context length, JSON-mode support, and multimodal formatting.

Local models vary more than hosted APIs in their ability to return strict JSON. For best results, use instruction-tuned models with reliable JSON output and keep discovery batches small enough for the model context window.

## Core API

Import the common helpers directly from `llm_feature_gen`:

```python
from llm_feature_gen import (
    LLMFeatureTransformer,
    discover_features_from_images,
    discover_features_from_tabular,
    discover_features_from_texts,
    discover_features_from_videos,
    generate_features_from_images,
    generate_features_from_tabular,
    generate_features_from_texts,
    generate_features_from_videos,
)
```

Discovery helpers write JSON schemas to `outputs/` by default:

| Helper | Default output |
| --- | --- |
| `discover_features_from_texts` | `outputs/discovered_text_features.json` |
| `discover_features_from_images` | `outputs/discovered_image_features.json` |
| `discover_features_from_tabular` | `outputs/discovered_tabular_features.json` |
| `discover_features_from_videos` | `outputs/discovered_video_features.json` |

Generation helpers read the matching schema by default and write one CSV per class. Set `merge_to_single_csv=True` to also create `outputs/all_feature_values.csv`.

### scikit-learn Transformer

`LLMFeatureTransformer` is a text-only `TransformerMixin` / `BaseEstimator` implementation, so beyond the `Pipeline` use shown in the Quickstart it also works with `ColumnTransformer`, `cross_val_score`, and `GridSearchCV`. It accepts a list of strings, a pandas `Series`, or a DataFrame, and exposes `get_feature_names_out()`. Inputs the model declines to score come back as `"not given by LLM"`.

| Argument | Default | Purpose |
| --- | --- | --- |
| `provider` | `None` | Provider instance; defaults to `OpenAIProvider()` |
| `discovered_features` | `None` | Schema dict or JSON path; `None` discovers on every `fit` |
| `text_column` | `None` | Column holding the text when `X` is a multi-column DataFrame |
| `use_batch` | `True` | One request per batch instead of one call per row |
| `batch_size` | `8` | Samples per provider call when batching |
| `cache` | `None` | A `BatchTextCache` so repeated runs over the same texts are free |
| `min_features` | `10` | Minimum number of features to propose during discovery |
| `output_dir` | `"outputs"` | Directory for the schema written during discovery |
| `output_filename` | `"discovered_text_features.json"` | Name of that schema file |

When it discovers a schema, `fit` also writes it to `output_dir/output_filename`, overwriting any previous file. That written schema is exactly what you pass back as `discovered_features` to pin the pipeline.

### Text

Shown in [The Two-Step Workflow](#the-two-step-workflow) above. `min_features` is accepted here too, as it is by every discovery helper.

### Images

```python
from llm_feature_gen import discover_features_from_images, generate_features_from_images

discover_features_from_images("discover_images")

generate_features_from_images(
    root_folder="images",
    merge_to_single_csv=True,
)
```

### Tabular Data

```python
from llm_feature_gen import discover_features_from_tabular, generate_features_from_tabular

discover_features_from_tabular(
    file_or_folder="discover_tabular",
    text_column="text",
)

generate_features_from_tabular(
    root_folder="tabular",
    text_column="text",
    label_column="label",
    merge_to_single_csv=True,
)
```

### Video

```python
from llm_feature_gen import discover_features_from_videos, generate_features_from_videos

discover_features_from_videos(
    videos_or_folder="discover_videos",
    num_frames=5,
    use_audio=True,
)

generate_features_from_videos(
    root_folder="videos",
    use_audio=True,
    merge_to_single_csv=True,
)
```

Set `use_audio=False` if you only want visual frames or do not have audio transcription configured.

## Batch and Multi-Class Text Workflows

For larger text datasets, use batched generation with an on-disk cache:

```python
from llm_feature_gen import generate_features_from_texts_cached

generate_features_from_texts_cached(
    root_folder="texts",
    discovered_features_path="outputs/discovered_text_features.json",
    batch_size=8,
)
```

For three or more classes, use the multi-class helpers:

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

## End-to-End Example

The repository includes a complete text-to-tabular example with checked-in expected outputs:

```bash
python examples/text_to_tabular_pipeline.py --provider auto
```

For the offline replay path used by tests:

```bash
python examples/text_to_tabular_pipeline.py --provider replay --check
```

See [examples/README.md](examples/README.md) for details. For an interactive walkthrough of all four modalities, open [tutorial.ipynb](tutorial.ipynb) from a repository checkout.

## Development

From a repository checkout:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

Build the documentation locally with:

```bash
pip install -e ".[docs]"
mkdocs serve
```

Useful project links:

- Documentation: https://juliayershova.github.io/llm-feature-gen/
- Changelog: [CHANGELOG.md](CHANGELOG.md)
- Contributing guide: [CONTRIBUTING.md](CONTRIBUTING.md)
- Issues: https://github.com/JuliaYershova/LLM-feature-gen/issues

## Citing

If you use `llm-feature-gen` in academic work, cite the software (a machine-readable [`CITATION.cff`](CITATION.cff) is also included, so GitHub's "Cite this repository" button works):

```bibtex
@software{jersova_llm_feature_gen,
  author  = {Jer{\v s}ova, Julija},
  title   = {llm-feature-gen: interpretable feature discovery and
             generation from multimodal data with LLMs},
  year    = {2026},
  url     = {https://github.com/JuliaYershova/LLM-feature-gen},
  version = {0.1.13},
}
```

## License

MIT. See [LICENSE](LICENSE).
