from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

import llm_feature_gen.sklearn as sklearn_mod
from llm_feature_gen.providers.openai_provider import OpenAIProvider
from llm_feature_gen.sklearn import LLMFeatureTransformer


class FakeTextProvider:
    def __init__(self) -> None:
        self.calls = []

    def text_features(self, text_list, prompt=None, system_prompt=None, response_schema=None):
        self.calls.append(
            {
                "texts": list(text_list),
                "prompt": prompt,
                "system_prompt": system_prompt,
                "response_schema": response_schema,
            }
        )
        if prompt and "DISCOVERED_FEATURES_SPEC" in prompt:
            include_length = '"feature": "length"' in prompt
            responses = []
            for text in text_list:
                features = {"topic": "billing" if "invoice" in text else "access"}
                if include_length:
                    features["length"] = str(len(text))
                responses.append({"features": features})
            return responses
        return [{"proposed_features": [{"feature": "topic"}, {"feature": "length"}]}]


def test_llm_feature_transformer_discovers_and_transforms_texts(tmp_path):
    provider = FakeTextProvider()
    transformer = LLMFeatureTransformer(provider=provider, output_dir=tmp_path, min_features=2)

    result = transformer.fit_transform(["need invoice", "cannot log in"])

    assert list(result.columns) == ["topic", "length"]
    assert result.shape == (2, 2)
    assert result["topic"].tolist() == ["billing", "access"]
    assert transformer.get_feature_names_out().tolist() == ["topic", "length"]
    assert (tmp_path / "discovered_text_features.json").exists()


def test_llm_feature_transformer_discovers_with_multiclass_labels(tmp_path, monkeypatch):
    captured = {}

    def fake_discover_features_multiclass(**kwargs):
        captured.update(kwargs)
        return {"proposed_features": [{"feature": "topic"}]}

    monkeypatch.setattr(
        sklearn_mod,
        "discover_features_multiclass",
        fake_discover_features_multiclass,
    )

    transformer = LLMFeatureTransformer(
        provider=FakeTextProvider(),
        output_dir=tmp_path,
        min_features=3,
    )

    transformer.fit(
        ["refund needed", "cannot login", "invoice issue"],
        y=["billing", "access", "billing"],
    )

    assert captured["texts_or_file"] == ["refund needed", "cannot login", "invoice issue"]
    assert captured["classes"] == ["billing", "access"]
    assert captured["output_dir"] == tmp_path
    assert captured["output_filename"] == "discovered_text_features.json"
    assert captured["min_features"] == 3

    transformer = LLMFeatureTransformer(
        provider=FakeTextProvider(),
        classes=["spam", "ham"],
    )

    transformer.fit(["limited offer", "hello"], y=["ignored", "ignored"])

    assert captured["classes"] == ["spam", "ham"]


def test_llm_feature_transformer_supports_openai_structured_outputs(tmp_path):
    discovery = {
        "proposed_features": [
            {
                "feature": "topic",
                "description": "Primary request topic",
                "possible_values": ["billing", "access"],
            }
        ]
    }
    responses = [discovery, {"topic": "billing"}, {"topic": "access"}]
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        content = json.dumps(responses.pop(0))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content, refusal=None))]
        )

    provider = object.__new__(OpenAIProvider)
    provider.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    provider.default_model = "gpt-test"
    provider.max_retries = 1
    provider.temperature = 0.0
    provider.max_completion_tokens = 256
    provider.max_tokens = 256
    provider.reasoning_effort = None
    provider._completion_token_parameter = "max_completion_tokens"
    provider._response_schema_support = {}

    transformer = LLMFeatureTransformer(
        provider=provider,
        output_dir=tmp_path,
        min_features=1,
        batch_size=2,
    )
    result = transformer.fit_transform(["need invoice", "cannot log in"])

    assert result.to_dict(orient="records") == [{"topic": "billing"}, {"topic": "access"}]
    assert calls[0]["response_format"]["json_schema"]["schema"]["required"] == ["proposed_features"]
    assert calls[1]["response_format"]["json_schema"]["schema"]["required"] == ["topic"]
    assert calls[2]["response_format"]["json_schema"]["schema"]["required"] == ["topic"]
    assert provider.usage_summary()["calls"] == 3


def test_llm_feature_transformer_forwards_openai_prompts(tmp_path):
    provider = FakeTextProvider()
    transformer = LLMFeatureTransformer(
        provider=provider,
        output_dir=tmp_path,
        min_features=2,
        discovery_prompt="Discover {min_features} features for {n_classes} groups:\n{class_list}",
        discovery_system_prompt="discovery instructions",
        generation_prompt="custom generation task",
        generation_system_prompt="generation instructions",
    )

    transformer.fit_transform(["need invoice"])

    assert provider.calls[0]["system_prompt"] == "discovery instructions"
    assert provider.calls[0]["prompt"].startswith("Discover 2 features for 2 groups")
    assert provider.calls[1]["system_prompt"] == "generation instructions"
    assert provider.calls[1]["prompt"].startswith("custom generation task")
    assert "DISCOVERED_FEATURES_SPEC" in provider.calls[1]["prompt"]
    assert transformer.get_params()["generation_prompt"] == "custom generation task"


def test_llm_feature_transformer_uses_supplied_schema_and_dataframe_column():
    provider = FakeTextProvider()
    transformer = LLMFeatureTransformer(
        provider=provider,
        discovered_features={"proposed_features": [{"feature": "topic"}]},
        text_column="body",
    )

    result = transformer.fit_transform(pd.DataFrame({"body": ["need invoice"], "id": [1]}))

    assert result.to_dict(orient="records") == [{"topic": "billing"}]
    assert provider.calls[0]["texts"] == ["need invoice"]


def test_llm_feature_transformer_can_transform_without_batch_mode():
    provider = FakeTextProvider()
    transformer = LLMFeatureTransformer(
        provider=provider,
        discovered_features={"proposed_features": [{"feature": "topic"}]},
        use_batch=False,
    )

    result = transformer.fit_transform(["need invoice", "cannot log in"])

    assert result.to_dict(orient="records") == [{"topic": "billing"}, {"topic": "access"}]
    assert [call["texts"] for call in provider.calls] == [["need invoice"], ["cannot log in"]]


def test_llm_feature_transformer_without_batch_mode_handles_empty_responses():
    class EmptyResponseProvider:
        def text_features(self, text_list, prompt=None):
            return []

    result = LLMFeatureTransformer(
        provider=EmptyResponseProvider(),
        discovered_features={"proposed_features": [{"feature": "topic"}]},
        use_batch=False,
    ).fit_transform(["need invoice"])

    assert result.to_dict(orient="records") == [{"topic": "not given by LLM"}]


def test_llm_feature_transformer_loads_schema_path_and_uses_default_provider(tmp_path, monkeypatch):
    provider = FakeTextProvider()
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(
        json.dumps({"proposed_features": [{"feature": "topic"}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(sklearn_mod, "OpenAIProvider", lambda: provider)

    result = LLMFeatureTransformer(discovered_features=schema_path).fit_transform(["need invoice"])

    assert result.to_dict(orient="records") == [{"topic": "billing"}]
    assert provider.calls[0]["texts"] == ["need invoice"]


def test_llm_feature_transformer_accepts_common_single_column_inputs():
    schema = {"proposed_features": [{"feature": "topic"}]}

    assert LLMFeatureTransformer(
        provider=FakeTextProvider(),
        discovered_features=schema,
    ).fit_transform(pd.Series(["need invoice"])).to_dict(orient="records") == [{"topic": "billing"}]

    assert LLMFeatureTransformer(
        provider=FakeTextProvider(),
        discovered_features=schema,
    ).fit_transform(pd.DataFrame({"body": ["cannot log in"]})).to_dict(orient="records") == [{"topic": "access"}]

    assert LLMFeatureTransformer(
        provider=FakeTextProvider(),
        discovered_features=schema,
    ).fit_transform([["need invoice"]]).to_dict(orient="records") == [{"topic": "billing"}]


def test_llm_feature_transformer_validates_input_shape():
    transformer = LLMFeatureTransformer(
        provider=FakeTextProvider(),
        discovered_features={"proposed_features": [{"feature": "topic"}]},
    )

    with pytest.raises(ValueError, match="Column 'missing'"):
        LLMFeatureTransformer(
            provider=FakeTextProvider(),
            discovered_features={"proposed_features": [{"feature": "topic"}]},
            text_column="missing",
        ).fit(pd.DataFrame({"body": ["one"]}))

    with pytest.raises(ValueError, match="text_column"):
        transformer.fit(pd.DataFrame({"a": ["one"], "b": ["two"]}))

    with pytest.raises(ValueError, match="one or more text samples"):
        transformer.fit("only one scalar string")

    with pytest.raises(ValueError, match="1D sequence"):
        transformer.fit([["one", "two"]])

    with pytest.raises(ValueError, match="not fitted"):
        transformer.transform(["one"])

    with pytest.raises(ValueError, match="not fitted"):
        transformer.get_feature_names_out()

    with pytest.raises(ValueError, match="at least one feature name"):
        LLMFeatureTransformer(
            provider=FakeTextProvider(),
            discovered_features={"proposed_features": []},
        ).fit(["one"])


def test_llm_feature_transformer_works_in_sklearn_pipeline():
    pytest.importorskip("sklearn")
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    pipe = Pipeline(
        [
            (
                "llm",
                LLMFeatureTransformer(
                    provider=FakeTextProvider(),
                    discovered_features={"proposed_features": [{"feature": "topic"}]},
                ),
            ),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    result = pipe.fit_transform(["need invoice", "cannot log in"])

    assert result.shape == (2, 2)
