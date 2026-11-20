# SPDX-License-Identifier: MIT
# Licensed under the MIT License. See the LICENSE file in the repository root.
"""Index and mixed-column integration regressions for issue #82."""

import numpy as np
import pandas as pd
import pytest

from llm_feature_gen import LLMFeatureTransformer


class EchoFeatureProvider:
    """Return input-dependent values without an API key or network access."""

    def text_features(self, texts, **kwargs):
        return [{"features": {"f": text}} for text in texts]


SCHEMA = {
    "proposed_features": [{
        "feature": "f",
        "description": "Echo the input text for an offline alignment check.",
        "possible_values": ["a", "b"],
    }]
}


@pytest.mark.parametrize("use_batch", [True, False])
@pytest.mark.parametrize("input_kind", ["series", "dataframe"])
@pytest.mark.parametrize(
    "index",
    [
        pytest.param(pd.Index([205, 101], name="client_id"), id="named-integer"),
        pytest.param(
            pd.MultiIndex.from_tuples(
                [("b", 2), ("a", 1)], names=["group", "client_id"]
            ),
            id="named-multiindex",
        ),
        pytest.param(
            pd.date_range(
                "2026-01-01", periods=2, tz="Europe/Prague", name="observed_at"
            )[::-1],
            id="timezone-datetime",
        ),
    ],
)
def test_pandas_index_and_label_alignment(use_batch, input_kind, index):
    X = pd.Series(["a", "b"], index=index, name="body")
    if input_kind == "dataframe":
        X = X.to_frame()
    transformer = LLMFeatureTransformer(
        provider=EchoFeatureProvider(),
        discovered_features=SCHEMA,
        use_batch=use_batch,
        show_progress=False,
    )

    result = transformer.fit_transform(X)

    pd.testing.assert_index_equal(result.index, X.index)
    assert result["f"].tolist() == ["a", "b"]
    result["target"] = pd.Series([1, 0], index=X.index)
    assert result["target"].tolist() == [1, 0]


@pytest.mark.parametrize("use_batch", [True, False])
@pytest.mark.parametrize(
    "X",
    [
        pytest.param(["a", "b"], id="list"),
        pytest.param(np.array(["a", "b"]), id="array-1d"),
        pytest.param(np.array([["a"], ["b"]]), id="array-2d"),
    ],
)
def test_non_pandas_inputs_keep_range_index(use_batch, X):
    result = LLMFeatureTransformer(
        provider=EchoFeatureProvider(),
        discovered_features=SCHEMA,
        use_batch=use_batch,
        show_progress=False,
    ).fit_transform(X)

    pd.testing.assert_index_equal(result.index, pd.RangeIndex(2))
    assert result["f"].tolist() == ["a", "b"]


@pytest.mark.parametrize("use_batch", [True, False])
def test_column_transformer_combines_text_and_numeric_pandas_outputs(use_batch):
    pytest.importorskip("sklearn")
    from sklearn.compose import ColumnTransformer

    X = pd.DataFrame(
        {"body": ["a", "b"], "age": [28, 50]},
        index=pd.Index([205, 101], name="client_id"),
    )
    transformer = ColumnTransformer([
        (
            "llm",
            LLMFeatureTransformer(
                provider=EchoFeatureProvider(),
                discovered_features=SCHEMA,
                use_batch=use_batch,
                show_progress=False,
            ),
            "body",
        ),
        ("numeric", "passthrough", ["age"]),
    ]).set_output(transform="pandas")

    result = transformer.fit_transform(X)

    expected = pd.DataFrame(
        {"llm__f": ["a", "b"], "numeric__age": [28, 50]}, index=X.index
    )
    pd.testing.assert_frame_equal(result, expected)
