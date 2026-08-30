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
| `AZURE_OPENAI_WHISPER_DEPLOYMENT` | Only for audio | Deployment used for audio transcription |

### Completion and structured-output options

Use `max_completion_tokens` to set the completion limit. `max_tokens` remains a backwards-compatible alias, but the two options cannot be passed together.

`reasoning_effort` defaults to `"none"`. Users can select `"minimal"`, `"low"`, `"medium"`, `"high"`, `"xhigh"`, or `"max"`, subject to the levels supported by their chosen model. Passing `None` omits the parameter entirely. If an older OpenAI or Azure deployment rejects the parameter, the provider retries without it and remembers that result for later requests to the same deployment.

`OpenAIProvider` prefers JSON Schema responses and automatically retries in JSON-object mode when the selected OpenAI or Azure deployment does not support schemas. `LocalProvider` uses the JSON response shape embedded in the prompts instead.

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

You can construct a provider and pass it into any discovery or generation helper.
Both OpenAI and local providers accept `prompt` for task instructions and
`system_prompt` for the model's role and behavior:

```python
from llm_feature_gen import generate_features_from_videos
from llm_feature_gen.providers import OpenAIProvider

provider = OpenAIProvider(
    max_completion_tokens=4096,
    reasoning_effort="low",
)
csv_paths = generate_features_from_videos(
    root_folder="videos",
    provider=provider,
    prompt="Infer each feature from visible or transcribed evidence.",
    system_prompt="Act as a careful dataset annotator and return JSON only.",
    merge_to_single_csv=True,
)
```

All discovery and generation helpers also accept an optional `system_prompt` for custom model instructions. If you build a custom provider, keep the same method signatures as the built-in providers so it can drop into the helper functions cleanly.
