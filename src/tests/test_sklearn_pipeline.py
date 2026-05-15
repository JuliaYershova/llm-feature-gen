import pytest
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from llm_feature_gen.sklearn import LLMFeatureExtractor


def test_sklearn_pipeline_integration():
    """Testuje, zda LLMFeatureExtractor funguje uvnitř sklearn Pipeline."""

    class MockProvider:
        def text_features(self, texts, prompt, **kwargs):
            if "discover" in prompt.lower():
                # Změněno na příznaky, které zní jako Yes/No otázky
                return [{"proposed_features": ["is_urgent", "is_negative"]}]

            # Změněno: Vracíme čistá čísla (1 nebo 0), aby s tím scikit-learn mohl počítat
            return [{"features": {"is_urgent": 1, "is_negative": 1}}]

    X = ["I lost my card!", "How do I order a new one?"]
    y = [1, 0]

    provider = MockProvider()
    pipe = Pipeline([
        ('llm_extractor', LLMFeatureExtractor(provider=provider, discover_kwargs={"prompt": "discover"})),
        ('classifier', DecisionTreeClassifier(random_state=42))
    ])

    pipe.fit(X, y)
    predictions = pipe.predict(X)

    assert len(predictions) == 2

    extractor = pipe.named_steps['llm_extractor']
    assert extractor.feature_names_ == ["is_urgent", "is_negative"]