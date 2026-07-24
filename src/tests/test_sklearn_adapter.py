from __future__ import annotations

import json

import pandas as pd
import pytest

import llm_feature_gen.sklearn as sklearn_mod
from llm_feature_gen.sklearn import LLMFeatureTransformer


class FakeTextProvider:
    def __init__(self) -> None:
        self.calls = []

    def text_features(self, text_list, prompt=None):
        self.calls.append({"texts": list(text_list), "prompt": prompt})
        if prompt and "DISOVERED_FEATURES_SPEC" in prompt:
            return [
                {
                    "features": {
                        "topic": "billing" if "invoice" in text else "access",
                        "length": str(len(text)),
                    }
                }
                for text in text_list
            ]
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
