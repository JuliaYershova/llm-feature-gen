# Provider Configuration

The package ships with two provider implementations:

- `OpenAIProvider` for OpenAI and Azure OpenAI.
- `LocalProvider` for OpenAI-compatible local servers such as Ollama, vLLM, and LM Studio.

Both expose the same high-level methods used by the discovery and generation helpers:

- `image_features(...)`
- `text_features(...)`
- `transcribe_audio(...)`

## OpenAIProvider

`OpenAIProvider` auto-detects Azure mode when `AZURE_OPENAI_ENDPOINT` is set. Otherwise it uses the standard OpenAI API.

### OpenAI environment variables

| Variable | Required | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | Yes | API key for the OpenAI client |
| `OPENAI_MODEL` | Yes | Default chat model used for text and image flows |
| `OPENAI_AUDIO_MODEL` | No | Audio transcription model, defaults to `whisper-1` |

### Azure OpenAI environment variables

| Variable | Required | Purpose |
| --- | --- | --- |
| `AZURE_OPENAI_API_KEY` | Yes | Azure OpenAI API key |
| `AZURE_OPENAI_API_VERSION` | Yes | API version for the Azure client |
| `AZURE_OPENAI_ENDPOINT` | Yes | Azure resource endpoint |
| `AZURE_OPENAI_GPT41_DEPLOYMENT_NAME` | Yes | Default deployment name for chat completions |
| `AZURE_OPENAI_WHISPER_DEPLOYMENT` | Yes | Deployment used for audio transcription |

## LocalProvider

`LocalProvider` targets OpenAI-compatible local endpoints and uses `faster-whisper` for optional local transcription when installed.

### Local environment variables

| Variable | Required | Purpose |
| --- | --- | --- |
| `LOCAL_OPENAI_BASE_URL` | No | Base URL for the local OpenAI-compatible server |
| `LOCAL_OPENAI_API_KEY` | No | Placeholder key expected by the SDK, defaults to `ollama` |
| `LOCAL_MODEL_TEXT` | No | Default text model |
| `LOCAL_MODEL_VISION` | No | Default vision model |
| `LOCAL_WHISPER_MODEL_SIZE` | No | Faster-Whisper model size, defaults to `base` |
| `LOCAL_WHISPER_DEVICE` | No | `cpu`, `cuda`, or `auto` for local transcription |

Example `.env`:

```env
LOCAL_OPENAI_BASE_URL=http://localhost:11434/v1
LOCAL_OPENAI_API_KEY=ollama
LOCAL_MODEL_TEXT=llama3
LOCAL_MODEL_VISION=llava
LOCAL_WHISPER_MODEL_SIZE=base
LOCAL_WHISPER_DEVICE=cpu
```

## Passing providers explicitly

You can construct a provider and pass it into any discovery or generation helper:

```python
from llm_feature_gen import generate_features_from_videos
from llm_feature_gen.providers import OpenAIProvider

provider = OpenAIProvider(max_tokens=4096, temperature=0.0)
csv_paths = generate_features_from_videos(
    root_folder="videos",
    provider=provider,
    merge_to_single_csv=True,
)
```

If you build a custom provider, keep the same method signatures as the built-in providers so it can drop into the helper functions cleanly.
