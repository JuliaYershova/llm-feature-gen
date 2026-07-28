from __future__ import annotations

import builtins
import csv
import json
from pathlib import Path

import pytest

from llm_feature_gen.discover import discover_features_from_texts
from llm_feature_gen.generate import generate_features_from_texts
from llm_feature_gen.providers import local_provider as local_mod
from llm_feature_gen.providers.local_provider import LocalProvider
from llm_feature_gen.utils.text import extract_text_from_file


class SmokeProvider:
    def __init__(self) -> None:
        self.discovery_calls = []
        self.generation_calls = []

    def text_features(self, text_list, prompt=None, feature_gen=False):
        texts = list(text_list)
        if prompt and "DISCOVERED_FEATURES_SPEC" in prompt:
            self.generation_calls.append({"texts": texts, "prompt": prompt})
            return [
                {
                    "features": {
                        "tone": "positive" if "excellent" in text.lower() else "negative",
                        "length_bucket": "short" if len(text.split()) <= 4 else "long",
                    }
                }
                for text in texts
            ]

        self.discovery_calls.append({"texts": texts, "prompt": prompt})
        return [
            {
                "proposed_features": [
                    {"feature": "tone", "type": "categorical"},
                    {"feature": "length_bucket", "type": "categorical"},
                ]
            }
        ]


class FakeChatCompletions:
    def __init__(self) -> None:
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = json.dumps(
            {
                "proposed_features": [
                    {"feature": "sentiment", "type": "categorical"},
                    {"feature": "register", "type": "categorical"},
                ]
            }
        )
        return type(
            "Response",
            (),
            {"choices": [type("Choice", (), {"message": type("Message", (), {"content": content})()})()]},
        )()


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_text_discovery_smoke_writes_expected_json(tmp_path: Path):
    source_dir = tmp_path / "discover_texts"
    source_dir.mkdir()
    (source_dir / "a.txt").write_text("excellent soup", encoding="utf-8")
    (source_dir / "b.txt").write_text("bad service", encoding="utf-8")

    provider = SmokeProvider()
    output_dir = tmp_path / "outputs"

    result = discover_features_from_texts(source_dir, provider=provider, output_dir=output_dir)

    assert result["proposed_features"][0]["feature"] == "tone"
    assert provider.discovery_calls and len(provider.discovery_calls[0]["texts"]) == 1

    output_path = output_dir / "discovered_text_features.json"
    assert output_path.exists()
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved == [result]


def test_text_generation_smoke_writes_expected_csv_columns(tmp_path: Path):
    root = tmp_path / "texts"
    positive = root / "positive"
    negative = root / "negative"
    positive.mkdir(parents=True)
    negative.mkdir(parents=True)
    (positive / "review1.txt").write_text("excellent broth", encoding="utf-8")
    (negative / "review2.txt").write_text("cold food", encoding="utf-8")

    discovered_path = tmp_path / "discovered_text_features.json"
    discovered_path.write_text(
        json.dumps(
            [
                {
                    "proposed_features": [
                        {"feature": "tone", "type": "categorical"},
                        {"feature": "length_bucket", "type": "categorical"},
                    ]
                }
            ]
        ),
        encoding="utf-8",
    )

    provider = SmokeProvider()
    output_dir = tmp_path / "outputs"

    result = generate_features_from_texts(
        root_folder=root,
        discovered_features_path=discovered_path,
        provider=provider,
        output_dir=output_dir,
        merge_to_single_csv=True,
    )

    merged_path = Path(result["__merged__"])
    assert merged_path.exists()

    rows = _read_csv_rows(merged_path)
    assert len(rows) == 2
    assert list(rows[0].keys()) == ["File", "Class", "tone", "length_bucket", "raw_llm_output"]
    assert {row["Class"] for row in rows} == {"positive", "negative"}
    assert {row["tone"] for row in rows} == {"positive", "negative"}
    assert provider.generation_calls


def test_local_provider_smoke_uses_openai_compatible_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    fake_create = FakeChatCompletions()
    fake_client = type(
        "FakeClient",
        (),
        {"chat": type("Chat", (), {"completions": type("Completions", (), {"create": fake_create.create})()})()},
    )()

    monkeypatch.setattr(local_mod, "OpenAI", lambda **kwargs: fake_client)

    provider = LocalProvider(
        base_url="http://127.0.0.1:9999/v1",
        api_key="test-key",
        default_text_model="fake-text-model",
    )

    output_dir = tmp_path / "outputs"
    result = discover_features_from_texts(
        ["warm, detailed review text"],
        provider=provider,
        output_dir=output_dir,
    )

    assert result["proposed_features"][0]["feature"] == "sentiment"
    assert fake_create.calls
    request = fake_create.calls[0]
    assert request["model"] == "fake-text-model"
    assert request["messages"][0]["role"] == "system"
    assert request["messages"][1]["role"] == "user"

    saved = json.loads((output_dir / "discovered_text_features.json").read_text(encoding="utf-8"))
    assert saved[0]["proposed_features"][1]["feature"] == "register"


@pytest.mark.parametrize(
    ("suffix", "import_name", "package_name"),
    [
        (".pdf", "pypdf", "pypdf"),
        (".docx", "docx", "python-docx"),
        (".html", "bs4", "beautifulsoup4"),
    ],
)
def test_optional_parser_missing_dependency_message_is_clear(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    suffix: str,
    import_name: str,
    package_name: str,
):
    path = tmp_path / f"sample{suffix}"
    path.write_text("placeholder", encoding="utf-8")

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == import_name:
            raise ImportError(f"No module named '{import_name}'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match=package_name):
        extract_text_from_file(path)
