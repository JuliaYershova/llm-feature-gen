import pandas as pd
from llm_feature_gen import LLMFeatureTransformer


class _FakeProvider:
    def text_features(self, texts, **kwargs):
        return [{"features": {"sentiment": "neutral"}} for _ in texts]


def test_transform_empty_input_returns_empty_dataframe_with_columns():
    schema = {
        "proposed_features": [
            {
                "feature": "sentiment",
                "description": "Overall tone",
                "possible_values": ["positive", "neutral", "negative"],
            }
        ]
    }
    transformer = LLMFeatureTransformer(
        provider=_FakeProvider(),
        discovered_features=schema,
        show_progress=False,
    )
    transformer.fit(["Sample text"])

    out_list = transformer.transform([])
    assert isinstance(out_list, pd.DataFrame)
    assert out_list.empty
    assert list(out_list.columns) == ["sentiment"]

    empty_series = pd.Series([], dtype=str, index=pd.Index([], name="row_id"))
    out_series = transformer.transform(empty_series)
    assert out_series.empty
    assert list(out_series.columns) == ["sentiment"]
    assert out_series.index.name == "row_id"
