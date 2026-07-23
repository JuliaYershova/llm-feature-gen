# LLM Feature Gen

[![PyPI version](https://img.shields.io/pypi/v/llm-feature-gen)](https://pypi.org/project/llm-feature-gen/)
[![Tests](https://github.com/JuliaYershova/LLM-feature-gen/actions/workflows/tests.yml/badge.svg)](https://github.com/JuliaYershova/LLM-feature-gen/actions/workflows/tests.yml)
[![Docs](https://github.com/JuliaYershova/LLM-feature-gen/actions/workflows/docs.yml/badge.svg)](https://juliayershova.github.io/llm-feature-gen/)
[![Codecov](https://codecov.io/gh/JuliaYershova/LLM-feature-gen/graph/badge.svg?token=BHLNPPOZUH)](https://codecov.io/gh/JuliaYershova/LLM-feature-gen)

`llm-feature-gen` turns text, images, tabular files, and videos into interpretable tabular features with LLMs.

Use it when you have raw multimodal data and want a CSV dataset with explicit, human-readable feature columns for analysis, modeling, or inspection.

## What It Does

The package follows a two-step workflow:

1. Discover a feature schema from example inputs.
2. Generate feature values for class-organized data and save CSV files.

Supported workflows include:

- Text: `.txt`, `.md`, `.pdf`, `.docx`, `.html`
- Images: `.jpg`, `.jpeg`, `.png`
- Tabular data: `.csv`, `.xlsx`, `.xls`, `.parquet`, `.json`
- Video: `.mp4`, `.mov`, `.avi`, `.mkv`

The default provider is OpenAI or Azure OpenAI. A local OpenAI-compatible provider is also available for Ollama, vLLM, LM Studio, and similar servers.

## Installation

```bash
pip install llm-feature-gen
```

Supported targets are CPython 3.9, 3.11, and 3.13 on Linux, macOS, and Windows. See [SUPPORT.md](SUPPORT.md) for the current support matrix.

Some file formats need optional runtime dependencies:

```bash
pip install pypdf python-docx beautifulsoup4 openpyxl xlrd pyarrow
```

Video audio extraction also requires the `ffmpeg` system binary.

## Configure a Provider

Create a `.env` file in the directory where you run your script. For open-source-first and offline-friendly experiments, start with `LocalProvider`.

## Local Provider Setup

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

### Ollama Example

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

Then pass the provider explicitly to both discovery and generation:

```python
from llm_feature_gen import discover_features_from_texts, generate_features_from_texts
from llm_feature_gen.providers import LocalProvider

provider = LocalProvider()

discovered = discover_features_from_texts(
    texts_or_file="discover_texts",
    provider=provider,
)

csv_paths = generate_features_from_texts(
    root_folder="texts",
    provider=provider,
    merge_to_single_csv=True,
)

print(discovered)
print(csv_paths)
```

### LM Studio or vLLM

Point `LOCAL_OPENAI_BASE_URL` at the server's OpenAI-compatible `/v1` endpoint and set the model names to the identifiers exposed by that server:

```env
LOCAL_OPENAI_BASE_URL=http://localhost:8000/v1
LOCAL_OPENAI_API_KEY=local
LOCAL_MODEL_TEXT=your-text-model
LOCAL_MODEL_VISION=your-vision-model
```

### Video and Audio Locally

Video workflows extract frames and use `LOCAL_MODEL_VISION` for visual analysis. If `use_audio=True`, `LocalProvider` uses `faster-whisper` for transcription.

Install local transcription support only if you need audio:

```bash
pip install faster-whisper
```

If you do not need audio, or want to avoid the extra dependency, call video helpers with `use_audio=False`.

### Reproducibility Notes

For papers and benchmarks, report the local backend, model names, model versions or hashes, quantization, hardware, and decoding settings. The package defaults to `temperature=0.0` in both built-in providers, but local servers can still differ in tokenizer, context length, JSON-mode support, and multimodal formatting.

Local models vary more than hosted APIs in their ability to return strict JSON. For best results, use instruction-tuned models with reliable JSON output and keep discovery batches small enough for the model context window.

## OpenAI and Azure OpenAI Setup

Use `OpenAIProvider` when you want a hosted backend. It auto-detects Azure mode when `AZURE_OPENAI_ENDPOINT` is set. Otherwise it uses the standard OpenAI API.

For OpenAI:

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

## Quickstart

This example creates a tiny text dataset, discovers a shared schema, and generates feature-value CSVs.

```python
from pathlib import Path

from llm_feature_gen import discover_features_from_texts, generate_features_from_texts

samples = {
    "demo_discover_texts/sample1.txt": "The dish was rich, spicy, and served in a deep bowl.",
    "demo_discover_texts/sample2.txt": "The dessert was light, creamy, and topped with fresh fruit.",
    "demo_texts/positive/review1.txt": "The meal was vibrant, aromatic, and beautifully plated.",
    "demo_texts/negative/review1.txt": "The service was slow and the food arrived cold.",
}

for file_name, text in samples.items():
    path = Path(file_name)
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

Generation expects one subfolder per class:

```text
demo_texts/
  positive/
    review1.txt
  negative/
    review1.txt
```

## Core API

Import the common helpers directly from `llm_feature_gen`:

```python
from llm_feature_gen import (
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

### Text

```python
from llm_feature_gen import discover_features_from_texts, generate_features_from_texts

discover_features_from_texts("discover_texts", min_features=10)

generate_features_from_texts(
    root_folder="texts",
    merge_to_single_csv=True,
)
```

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

See [examples/README.md](examples/README.md) for details.

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

## License

MIT. See [LICENSE](LICENSE).
