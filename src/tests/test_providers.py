from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from llm_feature_gen.providers.base_provider import Usage, BaseProvider
from llm_feature_gen.providers import local_provider as local_mod
from llm_feature_gen.providers import openai_provider as openai_mod
from llm_feature_gen.contracts import (
    ProviderResponseError,
    explain_empty_reply,
    instruct_variant_of,
)


class DummyRateLimitError(Exception):
    pass


class DummyBadRequestError(Exception):
    pass


class FakeCreate:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=item))])


def make_chat_client(responses):
    create = FakeCreate(responses)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return client, create


def test_openai_provider_init_paths(monkeypatch: pytest.MonkeyPatch):
    with pytest.raises(ValueError, match="only one"):
        openai_mod.OpenAIProvider(max_completion_tokens=100, max_tokens=50)
    with pytest.raises(ValueError, match="reasoning_effort"):
        openai_mod.OpenAIProvider(reasoning_effort="turbo")

    fake_azure_client = object()
    monkeypatch.setattr(openai_mod.openai, "AzureOpenAI", lambda **kwargs: fake_azure_client)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "v")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example")
    monkeypatch.setenv("AZURE_OPENAI_GPT41_DEPLOYMENT_NAME", "gpt")
    monkeypatch.setenv("AZURE_OPENAI_WHISPER_DEPLOYMENT", "whisper")
    provider = openai_mod.OpenAIProvider()
    assert provider.is_azure is True
    assert provider.client is fake_azure_client
    assert provider.max_completion_tokens == 2048
    assert provider.max_tokens == 2048
    assert provider.reasoning_effort == "none"
    assert provider._reasoning_effort_support == {}

    assert openai_mod.OpenAIProvider(max_tokens=4096).max_completion_tokens == 4096
    assert openai_mod.OpenAIProvider(max_completion_tokens=1024).max_completion_tokens == 1024
    assert openai_mod.OpenAIProvider(reasoning_effort="HIGH").reasoning_effort == "high"
    assert openai_mod.OpenAIProvider(reasoning_effort=None).reasoning_effort is None

    monkeypatch.delenv("AZURE_OPENAI_WHISPER_DEPLOYMENT")
    provider = openai_mod.OpenAIProvider()
    assert provider.is_azure is True
    assert provider.audio_model is None

    monkeypatch.delenv("AZURE_OPENAI_API_KEY")
    with pytest.raises(EnvironmentError):
        openai_mod.OpenAIProvider()

    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_MODEL", "gpt")
    monkeypatch.setattr(openai_mod, "OpenAI", lambda api_key: "personal-client")
    provider = openai_mod.OpenAIProvider()
    assert provider.is_azure is False
    assert provider.client == "personal-client"
    assert provider.audio_model == "whisper-1"

    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(EnvironmentError):
        openai_mod.OpenAIProvider()

    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.delenv("OPENAI_MODEL")
    with pytest.raises(EnvironmentError):
        openai_mod.OpenAIProvider()


def test_openai_provider_chat_and_public_methods(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    assert openai_mod.OpenAIProvider.supports_response_schema is True

    provider = object.__new__(openai_mod.OpenAIProvider)
    provider.max_retries = 2
    provider.temperature = 0.1
    provider.max_completion_tokens = 50
    provider.reasoning_effort = "low"
    provider.default_model = "model"
    provider.audio_model = "audio-model"

    client, create = make_chat_client(['{"ok": 1}'])
    provider.client = client
    assert provider._chat_json("m", "system", [{"type": "text", "text": "u"}], json_mode=True) == {"ok": 1}
    assert create.calls[0]["response_format"] == {"type": "json_object"}
    assert create.calls[0]["max_completion_tokens"] == 50
    assert create.calls[0]["reasoning_effort"] == "low"
    assert "temperature" not in create.calls[0]

    client, create = make_chat_client(['{"proposed_features": []}'])
    provider.client = client
    assert provider._chat_json(
        "m",
        "system",
        [{"type": "text", "text": "u"}],
        json_mode=True,
        response_schema=openai_mod.FEATURE_DISCOVERY_SCHEMA,
    ) == {"proposed_features": []}
    response_format = create.calls[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"]["required"] == ["proposed_features"]

    client, _ = make_chat_client(["not-json"])
    provider.client = client
    assert provider._chat_json("m", "system", [{"type": "text", "text": "u"}]) == {"features": "not-json"}

    monkeypatch.setattr(openai_mod.openai, "RateLimitError", DummyRateLimitError)
    sleeps = []
    monkeypatch.setattr(openai_mod.time, "sleep", lambda seconds: sleeps.append(seconds))
    client, _ = make_chat_client([DummyRateLimitError(), '{"retry": true}'])
    provider.client = client
    assert provider._chat_json("m", "system", [{"type": "text", "text": "u"}]) == {"retry": True}
    assert sleeps == [2]

    client, _ = make_chat_client([DummyRateLimitError(), DummyRateLimitError()])
    provider.client = client
    with pytest.raises(ProviderResponseError, match="Rate limit"):
        provider._chat_json("m", "system", [{"type": "text", "text": "u"}])

    client, _ = make_chat_client([RuntimeError("boom")])
    provider.client = client
    with pytest.raises(ProviderResponseError, match="boom"):
        provider._chat_json("m", "system", [{"type": "text", "text": "u"}])

    monkeypatch.setattr(openai_mod.openai, "BadRequestError", DummyBadRequestError)
    client, create = make_chat_client(
        [
            DummyBadRequestError("response_format json_schema is unsupported"),
            '{"proposed_features": []}',
        ]
    )
    provider.client = client
    assert provider._chat_json(
        "m",
        "system",
        [{"type": "text", "text": "u"}],
        json_mode=True,
        response_schema=openai_mod.FEATURE_DISCOVERY_SCHEMA,
    ) == {"proposed_features": []}
    assert create.calls[0]["response_format"]["type"] == "json_schema"
    assert create.calls[1]["response_format"] == {"type": "json_object"}
    assert provider._response_schema_support["m"] is False
    assert provider._should_fallback_to_json_mode(RuntimeError("response_format unsupported")) is False

    client, create = make_chat_client(['{"proposed_features": []}'])
    provider.client = client
    assert provider._chat_json(
        "m",
        "system",
        [{"type": "text", "text": "u"}],
        json_mode=True,
        response_schema=dict(openai_mod.FEATURE_DISCOVERY_SCHEMA),
    ) == {"proposed_features": []}
    assert create.calls[0]["response_format"] == {"type": "json_object"}

    client, create = make_chat_client(
        [DummyBadRequestError("response_format unsupported"), '{"ok": true}']
    )
    provider.client = client
    assert provider._chat_json(
        "no-json-mode",
        "system",
        [{"type": "text", "text": "u"}],
        response_schema={"type": "object"},
    ) == {"ok": True}
    assert "response_format" not in create.calls[1]

    client, _ = make_chat_client(
        [
            json.dumps(
                {
                    "proposed_features": [
                        {
                            "name": "task_type",
                            "feature": "task_type",
                            "description": "desc",
                            "possible_values": [],
                        }
                    ]
                }
            )
        ]
    )
    provider.client = client
    with pytest.raises(ProviderResponseError, match="unexpected keys"):
        provider._chat_json(
            "m",
            "system",
            [{"type": "text", "text": "u"}],
            json_mode=True,
            response_schema=dict(openai_mod.FEATURE_DISCOVERY_SCHEMA),
        )

    client, _ = make_chat_client(["not-json"])
    provider.client = client
    with pytest.raises(ProviderResponseError, match="Invalid JSON"):
        provider._chat_json(
            "m",
            "system",
            [{"type": "text", "text": "u"}],
            json_mode=True,
            response_schema=openai_mod.FEATURE_DISCOVERY_SCHEMA,
        )

    provider.reasoning_effort = "none"
    client, create = make_chat_client(['{"ok": true}'])
    provider.client = client
    assert provider._chat_json("m", "system", [{"type": "text", "text": "u"}]) == {"ok": True}
    assert create.calls[0]["reasoning_effort"] == "none"
    assert "temperature" not in create.calls[0]

    provider.max_retries = 0
    provider.client = make_chat_client(['{"unused": true}'])[0]
    with pytest.raises(ProviderResponseError, match="Unknown failure"):
        provider._chat_json("m", "system", [{"type": "text", "text": "u"}], json_mode=True)

    captured = []
    provider._chat_json = lambda deployment, system_prompt, user_content, json_mode=False, response_schema=None: captured.append(
        {
            "deployment": deployment,
            "system_prompt": system_prompt,
            "user_content": user_content,
            "json_mode": json_mode,
            "response_schema": response_schema,
        }
    ) or {"features": "x"}
    assert provider.image_features(["a", "b"], feature_gen=True) == [{"features": "x"}, {"features": "x"}]
    assert "strict JSON" in captured[0]["system_prompt"]
    assert captured[0]["user_content"][0]["type"] == "image_url"

    captured.clear()
    assert provider.image_features(["a", "b"], as_set=True, extra_context="ctx") == [{"features": "x"}]
    assert "ADDITIONAL CONTEXT" in captured[0]["user_content"][-1]["text"]

    captured.clear()
    assert provider.text_features(["hello"], prompt="prompt", feature_gen=True) == [{"features": "x"}]
    assert "structured JSON" in captured[0]["system_prompt"]
    captured.clear()
    assert provider.text_features(["hello"], feature_gen=True) == [{"features": "x"}]
    assert "tabular dataset construction" not in captured[0]["system_prompt"]
    captured.clear()
    assert provider.text_features(["hello"], prompt="plain", feature_gen=False) == [{"features": "x"}]
    assert captured[0]["system_prompt"] == "plain"
    assert captured[0]["response_schema"] is None

    captured.clear()
    assert provider.text_features(["hello"], prompt='{"proposed_features": []}') == [{"features": "x"}]
    assert captured[0]["response_schema"] is None

    captured.clear()
    assert provider.text_features(
        ["hello"],
        prompt='{"proposed_features": []}',
        response_schema=openai_mod.FEATURE_DISCOVERY_SCHEMA,
    ) == [{"features": "x"}]
    assert captured[0]["response_schema"] == openai_mod.FEATURE_DISCOVERY_SCHEMA

    captured.clear()
    assert provider.image_features(["a"], prompt="image task", system_prompt="custom vision system") == [{"features": "x"}]
    assert captured[0]["system_prompt"] == "custom vision system"
    assert captured[0]["user_content"][-1]["text"] == "image task"

    captured.clear()
    assert provider.text_features(["hello"], prompt="text task", system_prompt="custom text system") == [{"features": "x"}]
    assert captured[0]["system_prompt"] == "custom text system"
    assert captured[0]["user_content"][0]["text"] == "text task\n\nTEXT:\nhello"

    with pytest.raises(FileNotFoundError, match="not found"):
        provider.transcribe_audio(str(tmp_path / "missing.wav"))

    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")

    provider.is_azure = True
    provider.audio_model = None
    with pytest.raises(EnvironmentError, match="WHISPER"):
        provider.transcribe_audio(str(audio_path))

    provider.is_azure = False
    provider.audio_model = "audio-model"
    provider.client = SimpleNamespace(
        audio=SimpleNamespace(
            transcriptions=SimpleNamespace(
                create=lambda model, file: SimpleNamespace(text="transcribed")
            )
        )
    )
    assert provider.transcribe_audio(str(audio_path)) == "transcribed"

    provider.client = SimpleNamespace(
        audio=SimpleNamespace(
            transcriptions=SimpleNamespace(
                create=lambda model, file: (_ for _ in ()).throw(DummyRateLimitError())
            )
        )
    )
    with pytest.raises(DummyRateLimitError):
        provider.transcribe_audio(str(audio_path))

    provider.client = SimpleNamespace(
        audio=SimpleNamespace(
            transcriptions=SimpleNamespace(
                create=lambda model, file: (_ for _ in ()).throw(RuntimeError("bad"))
            )
        )
    )
    with pytest.raises(RuntimeError, match="bad"):
        provider.transcribe_audio(str(audio_path))


def test_openai_provider_retries_completion_token_parameter(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(openai_mod.openai, "BadRequestError", DummyBadRequestError)

    provider = object.__new__(openai_mod.OpenAIProvider)
    provider.max_retries = 1
    provider.temperature = 0.0
    provider.max_completion_tokens = 50

    assert provider._token_limit_fallback(
        "max_completion_tokens",
        RuntimeError("not a bad request"),
    ) is None
    assert provider._token_limit_fallback(
        "max_completion_tokens",
        DummyBadRequestError("different bad request"),
    ) is None
    assert provider._token_limit_fallback(
        "other",
        DummyBadRequestError("max_tokens max_completion_tokens"),
    ) is None

    client, create = make_chat_client(
        [
            DummyBadRequestError(
                "Unsupported parameter: 'max_completion_tokens'. Use 'max_tokens' instead."
            ),
            '{"ok": true}',
        ]
    )
    provider.client = client

    assert provider._chat_json("deployment-alias", "system", [{"type": "text", "text": "u"}]) == {"ok": True}
    assert "max_completion_tokens" in create.calls[0]
    assert "max_tokens" not in create.calls[0]
    assert "max_completion_tokens" not in create.calls[1]
    assert create.calls[1]["max_tokens"] == 50
    assert provider._completion_token_parameter == "max_tokens"

    client, create = make_chat_client(['{"remembered": true}'])
    provider.client = client
    assert provider._chat_json("another-alias", "system", [{"type": "text", "text": "u"}]) == {"remembered": True}
    assert "max_completion_tokens" not in create.calls[0]
    assert create.calls[0]["max_tokens"] == 50

    provider._completion_token_parameter = "max_tokens"
    client, create = make_chat_client(
        [
            DummyBadRequestError(
                "Unsupported parameter: 'max_tokens'. Use 'max_completion_tokens' instead."
            ),
            '{"modern": true}',
        ]
    )
    provider.client = client
    assert provider._chat_json("modern-alias", "system", [{"type": "text", "text": "u"}]) == {"modern": True}
    assert create.calls[1]["max_completion_tokens"] == 50
    assert provider._completion_token_parameter == "max_completion_tokens"


def test_openai_provider_falls_back_when_reasoning_effort_is_unsupported(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(openai_mod.openai, "BadRequestError", DummyBadRequestError)

    provider = object.__new__(openai_mod.OpenAIProvider)
    provider.max_retries = 1
    provider.temperature = 0.25
    provider.max_completion_tokens = 50
    provider.reasoning_effort = "none"

    assert provider._should_fallback_without_reasoning_effort(
        RuntimeError("reasoning_effort unsupported")
    ) is False
    assert provider._should_fallback_without_reasoning_effort(
        DummyBadRequestError("another bad request")
    ) is False

    client, create = make_chat_client(
        [
            DummyBadRequestError("Unsupported parameter: reasoning_effort"),
            '{"ok": true}',
        ]
    )
    provider.client = client

    assert provider._chat_json(
        "legacy-model", "system", [{"type": "text", "text": "u"}]
    ) == {"ok": True}
    assert create.calls[0]["reasoning_effort"] == "none"
    assert "temperature" not in create.calls[0]
    assert "reasoning_effort" not in create.calls[1]
    assert create.calls[1]["temperature"] == 0.25
    assert provider._reasoning_effort_support["legacy-model"] is False

    client, create = make_chat_client(['{"remembered": true}'])
    provider.client = client
    assert provider._chat_json(
        "legacy-model", "system", [{"type": "text", "text": "u"}]
    ) == {"remembered": True}
    assert "reasoning_effort" not in create.calls[0]
    assert create.calls[0]["temperature"] == 0.25


def test_openai_provider_validates_discovery_schema_payload():
    provider = object.__new__(openai_mod.OpenAIProvider)

    provider._validate_feature_discovery_payload(
        {
            "proposed_features": [
                {
                    "feature": "task_type",
                    "description": "desc",
                    "possible_values": ["a", "b"],
                }
            ]
        }
    )

    invalid_payloads = [
        [],
        {"unexpected": []},
        {"proposed_features": "not a list"},
        {"proposed_features": ["not an object"]},
        {"proposed_features": [{"feature": "x", "description": "desc", "possible_values": [], "type": "extra"}]},
        {"proposed_features": [{"feature": "x", "description": "desc"}]},
        {"proposed_features": [{"feature": 1, "description": "desc", "possible_values": []}]},
        {"proposed_features": [{"feature": "x", "description": 1, "possible_values": []}]},
        {"proposed_features": [{"feature": "x", "description": "desc", "possible_values": [1]}]},
    ]

    for payload in invalid_payloads:
        with pytest.raises(ProviderResponseError):
            provider._validate_feature_discovery_payload(payload)


def test_local_provider_extract_json_and_chat(monkeypatch: pytest.MonkeyPatch):
    assert local_mod.LocalProvider.supports_response_schema is False
    monkeypatch.setattr(local_mod, "OpenAI", lambda **kwargs: "client")
    provider = local_mod.LocalProvider()

    assert provider._extract_json('{"a": 1}') == {"a": 1}
    assert provider._extract_json("```json\n{\"a\": 1}\n```") == {"a": 1}
    assert provider._extract_json("prefix [1, 2] suffix") == [1, 2]
    assert provider._extract_json("```json\n{bad}\n```") is None
    assert provider._extract_json("prefix {bad} suffix") is None
    assert provider._extract_json("nothing here") is None

    monkeypatch.setattr(local_mod, "BadRequestError", DummyBadRequestError)
    monkeypatch.setattr(local_mod.openai, "RateLimitError", DummyRateLimitError)
    monkeypatch.setattr(local_mod.time, "sleep", lambda seconds: None)

    provider.max_retries = 2
    provider.temperature = 0.0
    provider.max_tokens = 100

    client, _ = make_chat_client(['{"ok": 1}'])
    provider.client = client
    assert provider._chat_json("m", "system", [{"type": "text", "text": "u"}], json_mode=True) == {"ok": 1}

    client, _ = make_chat_client(["before ```json\n[1,2]\n``` after"])
    provider.client = client
    assert provider._chat_json("m", "system", [{"type": "text", "text": "u"}]) == {"features": [1, 2]}

    client, _ = make_chat_client([DummyBadRequestError("json_object unsupported"), '{"fallback": true}'])
    provider.client = client
    assert provider._chat_json("m", "system", [{"type": "text", "text": "u"}], json_mode=True) == {"fallback": True}

    client, _ = make_chat_client([DummyBadRequestError("other")])
    provider.client = client
    with pytest.raises(DummyBadRequestError, match="other"):
        provider._chat_json("m", "system", [{"type": "text", "text": "u"}], json_mode=True)

    client, _ = make_chat_client([DummyRateLimitError(), DummyRateLimitError()])
    provider.client = client
    with pytest.raises(DummyRateLimitError):
        provider._chat_json("m", "system", [{"type": "text", "text": "u"}])

    client, _ = make_chat_client([RuntimeError("boom")])
    provider.client = client
    with pytest.raises(RuntimeError, match="boom"):
        provider._chat_json("m", "system", [{"type": "text", "text": "u"}])

    client, _ = make_chat_client(["plain words"])
    provider.client = client
    assert provider._chat_json("m", "system", [{"type": "text", "text": "u"}]) == {"features": "plain words"}

    client, _ = make_chat_client(["plain words"])
    provider.client = client
    with pytest.raises(ValueError, match="Invalid JSON response"):
        provider._chat_json("m", "system", [{"type": "text", "text": "u"}], json_mode=True)


    client, _ = make_chat_client(["prefix {\"a\": 1} suffix"])
    provider.client = client
    assert provider._chat_json("m", "system", [{"type": "text", "text": "u"}]) == {"a": 1}

    provider.max_retries = 0
    with pytest.raises(RuntimeError, match="Unknown failure"):
        provider._chat_json("m", "system", [{"type": "text", "text": "u"}])

def test_local_provider_public_methods_and_transcription(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(local_mod, "OpenAI", lambda **kwargs: "client")
    provider = local_mod.LocalProvider(default_text_model="txt-model", default_vision_model="vision-model")

    captured = []
    provider._chat_json = lambda deployment, system_prompt, user_content, json_mode=False: captured.append(
        {
            "deployment": deployment,
            "system_prompt": system_prompt,
            "user_content": user_content,
            "json_mode": json_mode,
        }
    ) or {"features": "x"}

    assert provider.image_features(["a"], feature_gen=True) == [{"features": "x"}]
    assert captured[-1]["deployment"] == "vision-model"
    assert provider.image_features(["a", "b"], as_set=True, extra_context="ctx") == [{"features": "x"}]
    assert "ADDITIONAL CONTEXT" in captured[-1]["user_content"][-1]["text"]

    assert provider.text_features(["hello"], prompt="prompt", feature_gen=True) == [{"features": "x"}]
    assert captured[-1]["deployment"] == "txt-model"
    assert provider.text_features(["hello"], feature_gen=True) == [{"features": "x"}]
    assert provider.text_features(["hello"], prompt="plain", feature_gen=False) == [{"features": "x"}]
    assert provider.text_features(["hello"], prompt="task", system_prompt="local custom") == [{"features": "x"}]
    assert captured[-1]["system_prompt"] == "local custom"
    assert captured[-1]["user_content"][0]["text"] == "task\n\nTEXT:\nhello"

    monkeypatch.setattr(local_mod, "HAS_LOCAL_WHISPER", False)
    with pytest.raises(ImportError, match="not installed"):
        provider.transcribe_audio("audio.wav")

    monkeypatch.setattr(local_mod, "HAS_LOCAL_WHISPER", True)

    class BrokenWhisper:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("init failed")

    monkeypatch.setattr(local_mod, "WhisperModel", BrokenWhisper, raising=False)
    provider._whisper_model = None
    with pytest.raises(RuntimeError, match="init failed"):
        provider.transcribe_audio("audio.wav")

    class Segment:
        def __init__(self, text):
            self.text = text

    class WorkingWhisper:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, audio_path, beam_size=5):
            return [Segment("hello"), Segment("world")], None

    monkeypatch.setattr(local_mod, "WhisperModel", WorkingWhisper, raising=False)
    provider._whisper_model = None
    assert provider.transcribe_audio("audio.wav") == "hello world"

    class FailingWhisper(WorkingWhisper):
        def transcribe(self, audio_path, beam_size=5):
            raise RuntimeError("oops")

    provider._whisper_model = FailingWhisper()
    with pytest.raises(RuntimeError, match="oops"):
        provider.transcribe_audio("audio.wav")


def test_local_provider_module_can_import_with_fake_faster_whisper(monkeypatch: pytest.MonkeyPatch):
    fake_fw = ModuleType("faster_whisper")
    fake_fw.WhisperModel = object
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw)
    reloaded = importlib.reload(local_mod)
    assert reloaded.HAS_LOCAL_WHISPER is True
    monkeypatch.delitem(sys.modules, "faster_whisper", raising=False)
    importlib.reload(local_mod)


class _Usage:
    def __init__(self, prompt_tokens=0, completion_tokens=0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _Response:
    def __init__(self, usage=None):
        self.usage = usage


class _Tracker(BaseProvider):
    """Bare mixin user — deliberately no __init__, to prove the lazy counter."""


def test_usage_dataclass_reports_totals():
    usage = Usage(calls=2, prompt_tokens=100, completion_tokens=25)
    assert usage.total_tokens == 125
    assert usage.as_dict() == {
        "calls": 2,
        "prompt_tokens": 100,
        "completion_tokens": 25,
        "total_tokens": 125,
    }


def test_record_usage_accumulates_across_calls():
    tracker = _Tracker()
    assert tracker.usage_summary()["calls"] == 0

    tracker._record_usage(_Response(_Usage(100, 25)))
    tracker._record_usage(_Response(_Usage(40, 10)))

    assert tracker.usage_summary() == {
        "calls": 2,
        "prompt_tokens": 140,
        "completion_tokens": 35,
        "total_tokens": 175,
    }


def test_record_usage_counts_calls_without_token_payload():
    tracker = _Tracker()
    tracker._record_usage(_Response(usage=None))
    tracker._record_usage(_Response(_Usage(None, None)))

    summary = tracker.usage_summary()
    assert summary["calls"] == 2
    assert summary["total_tokens"] == 0


def test_reset_usage_clears_counters():
    tracker = _Tracker()
    tracker._record_usage(_Response(_Usage(10, 5)))
    tracker.reset_usage()
    assert tracker.usage_summary() == {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

def test_usage_counter_can_be_replaced():
    tracker = _Tracker()
    tracker._record_usage(_Response(_Usage(10, 5)))

    tracker._usage = Usage(calls=7, prompt_tokens=1, completion_tokens=2)

    assert tracker.usage_summary() == {
        "calls": 7,
        "prompt_tokens": 1,
        "completion_tokens": 2,
        "total_tokens": 3,
    }

def test_local_provider_raises_a_useful_error_on_an_empty_reply():
    """An empty reply must name the cause, not surface as invalid JSON."""
    provider = object.__new__(local_mod.LocalProvider)
    provider.max_retries = 1
    provider.temperature = 0.0
    provider.max_tokens = 2048

    client, _ = make_chat_client([""])
    provider.client = client

    with pytest.raises(ProviderResponseError, match="empty reply"):
        provider._chat_json("m", "system", [{"type": "text", "text": "u"}], json_mode=True)


def test_explain_empty_reply_names_the_cause_and_the_fix():
    class Message:
        def __init__(self, reasoning=None):
            self.reasoning = reasoning

    class Usage:
        def __init__(self, completion_tokens):
            self.completion_tokens = completion_tokens

    class Response:
        def __init__(self, completion_tokens):
            self.usage = Usage(completion_tokens)

    explain = explain_empty_reply

    # reasoning came back instead of an answer -> point at the instruct tag
    thinking = explain(Response(2048), Message("reasoning..."), "qwen3-vl:32b", 2048)
    assert "reasoning but no answer" in thinking
    assert "qwen3-vl:32b-instruct" in thinking

    # no reasoning field, but the budget ran out -> same conclusion
    exhausted = explain(Response(2048), Message(), "qwen3-vl:32b", 2048)
    assert "all 2048 tokens" in exhausted
    assert "qwen3-vl:32b-instruct" in exhausted

    # already an instruct model -> suggest more room, not another tag
    instruct = explain(Response(2048), Message(), "qwen3-vl:32b-instruct", 2048)
    assert "Raise max_tokens" in instruct
    assert "instruct variant" not in instruct

    # empty for some other reason -> say so rather than guess
    unknown = explain(Response(7), Message(), "qwen3-vl:32b", 2048)
    assert "empty reply after 7 tokens" in unknown


def test_instruct_variant_of_skips_models_that_already_are_one():
    assert instruct_variant_of("qwen3-vl:32b") == "qwen3-vl:32b-instruct"
    assert instruct_variant_of("qwen3-vl:32b-instruct") == ""

def test_openai_provider_raises_a_useful_error_on_an_empty_reply():
    """The OpenAI path needs the same explanation as the local one."""
    provider = object.__new__(openai_mod.OpenAIProvider)
    provider.max_retries = 1
    provider.temperature = 0.0
    provider.max_tokens = 2048

    client, _ = make_chat_client([""])
    provider.client = client

    with pytest.raises(ProviderResponseError, match="empty reply"):
        provider._chat_json("m", "system", [{"type": "text", "text": "u"}], json_mode=True)


def test_openai_provider_surfaces_structured_output_refusals():
    provider = object.__new__(openai_mod.OpenAIProvider)
    provider.max_retries = 1
    provider.temperature = 0.0
    provider.max_completion_tokens = 128
    provider.reasoning_effort = None

    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None, refusal="unsafe request"))]
    )
    provider.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: response)
        )
    )

    with pytest.raises(ProviderResponseError, match="Model refused.*unsafe request"):
        provider._chat_json(
            "m",
            "system",
            [{"type": "text", "text": "u"}],
            json_mode=True,
            response_schema={"type": "object", "properties": {}, "additionalProperties": False},
        )
