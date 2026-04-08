from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from llm_feature_gen.providers import local_provider as local_mod
from llm_feature_gen.providers import openai_provider as openai_mod


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

    monkeypatch.delenv("AZURE_OPENAI_WHISPER_DEPLOYMENT")
    with pytest.raises(EnvironmentError):
        openai_mod.OpenAIProvider()

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
    provider = object.__new__(openai_mod.OpenAIProvider)
    provider.max_retries = 2
    provider.temperature = 0.1
    provider.max_tokens = 50
    provider.default_model = "model"
    provider.audio_model = "audio-model"

    client, create = make_chat_client(['{"ok": 1}'])
    provider.client = client
    assert provider._chat_json("m", "system", [{"type": "text", "text": "u"}], json_mode=True) == {"ok": 1}
    assert create.calls[0]["response_format"] == {"type": "json_object"}

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
    assert provider._chat_json("m", "system", [{"type": "text", "text": "u"}])["error"].startswith("Rate limit")

    client, _ = make_chat_client([RuntimeError("boom")])
    provider.client = client
    assert provider._chat_json("m", "system", [{"type": "text", "text": "u"}]) == {"error": "boom"}

    provider.max_retries = 0
    provider.client = make_chat_client(['{"unused": true}'])[0]
    assert provider._chat_json("m", "system", [{"type": "text", "text": "u"}], json_mode=True) is None

    captured = []
    provider._chat_json = lambda deployment, system_prompt, user_content, json_mode=False: captured.append(
        {
            "deployment": deployment,
            "system_prompt": system_prompt,
            "user_content": user_content,
            "json_mode": json_mode,
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

    missing = provider.transcribe_audio(str(tmp_path / "missing.wav"))
    assert "not found" in missing

    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")
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
    assert "Rate limit" in provider.transcribe_audio(str(audio_path))

    provider.client = SimpleNamespace(
        audio=SimpleNamespace(
            transcriptions=SimpleNamespace(
                create=lambda model, file: (_ for _ in ()).throw(RuntimeError("bad"))
            )
        )
    )
    assert "bad" in provider.transcribe_audio(str(audio_path))


def test_local_provider_extract_json_and_chat(monkeypatch: pytest.MonkeyPatch):
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
    assert provider._chat_json("m", "system", [{"type": "text", "text": "u"}], json_mode=True) == {"error": "other"}

    client, _ = make_chat_client([DummyRateLimitError(), DummyRateLimitError()])
    provider.client = client
    assert provider._chat_json("m", "system", [{"type": "text", "text": "u"}]) == {"error": "Rate limit exceeded."}

    client, _ = make_chat_client([RuntimeError("boom")])
    provider.client = client
    assert provider._chat_json("m", "system", [{"type": "text", "text": "u"}]) == {"error": "boom"}

    client, _ = make_chat_client(["plain words"])
    provider.client = client
    assert provider._chat_json("m", "system", [{"type": "text", "text": "u"}]) == {"features": "plain words"}

    client, _ = make_chat_client(["prefix {\"a\": 1} suffix"])
    provider.client = client
    assert provider._chat_json("m", "system", [{"type": "text", "text": "u"}]) == {"a": 1}

    provider.max_retries = 0
    assert provider._chat_json("m", "system", [{"type": "text", "text": "u"}]) == {"error": "Unknown failure"}


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

    monkeypatch.setattr(local_mod, "HAS_LOCAL_WHISPER", False)
    assert "not installed" in provider.transcribe_audio("audio.wav")

    monkeypatch.setattr(local_mod, "HAS_LOCAL_WHISPER", True)

    class BrokenWhisper:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("init failed")

    monkeypatch.setattr(local_mod, "WhisperModel", BrokenWhisper, raising=False)
    provider._whisper_model = None
    assert "init failed" in provider.transcribe_audio("audio.wav")

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
    assert "oops" in provider.transcribe_audio("audio.wav")


def test_local_provider_module_can_import_with_fake_faster_whisper(monkeypatch: pytest.MonkeyPatch):
    fake_fw = ModuleType("faster_whisper")
    fake_fw.WhisperModel = object
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw)
    reloaded = importlib.reload(local_mod)
    assert reloaded.HAS_LOCAL_WHISPER is True
    monkeypatch.delitem(sys.modules, "faster_whisper", raising=False)
    importlib.reload(local_mod)
