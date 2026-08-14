# Install Guide

## PyPI install

Install the published package:

```bash
pip install llm-feature-gen
```

### scikit-learn integration

`LLMFeatureTransformer` itself runs on the plain install — scikit-learn is not a required dependency, and the transformer falls back to its own `BaseEstimator` when scikit-learn is absent. You only need the extra to compose it with scikit-learn's own estimators, as the [quickstart](quickstart.md) pipeline does:

```bash
pip install "llm-feature-gen[sklearn]"
```

Quote the argument. Square brackets are glob characters in zsh — the default shell on macOS — so the unquoted form fails before pip ever runs:

```
zsh: no matches found: llm-feature-gen[sklearn]
```

`'llm-feature-gen[sklearn]'` and `llm-feature-gen\[sklearn\]` work equally well.

### Jupyter notebooks

When installing from inside a notebook, run the install command in its own cell
and restart the kernel before importing `llm_feature_gen`. Some hosted notebook
services keep the old kernel environment active after `%pip install`, which can
raise `ModuleNotFoundError` even after pip reports a successful install.

If `%pip` installs into the wrong environment, target the current kernel
explicitly:

```python
import sys

!{sys.executable} -m pip install -U llm-feature-gen
```

Supported targets are summarized in the [platform support matrix](support.md).

## Development install

If you are working from a repository checkout:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

To build the documentation locally:

```bash
pip install -e ".[docs]"
```

## Optional runtime dependencies

Some file formats depend on extra libraries at runtime:

- PDF: `pypdf`
- DOCX: `python-docx`
- HTML: `beautifulsoup4`
- XLSX: `openpyxl`
- XLS: `xlrd`
- Parquet: `pyarrow` or `fastparquet`

Install them as needed:

```bash
pip install pypdf python-docx beautifulsoup4 openpyxl xlrd pyarrow
```

Video audio extraction also requires the `ffmpeg` system binary to be available on your machine.

## Environment variables

The default provider reads credentials from a `.env` file in your working directory.

OpenAI:

```env
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-4.1-mini
OPENAI_AUDIO_MODEL=whisper-1
```

Azure OpenAI:

```env
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_API_VERSION=your_api_version
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_GPT41_DEPLOYMENT_NAME=your_chat_deployment
AZURE_OPENAI_WHISPER_DEPLOYMENT=your_audio_deployment
```

For local OpenAI-compatible servers, see the [provider configuration reference](provider-configuration.md).
