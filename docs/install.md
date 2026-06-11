# Install Guide

## PyPI install

Install the published package:

```bash
pip install llm-feature-gen
```

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
