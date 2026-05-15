import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .discover import discover_features_from_texts
from .generate import _build_prompt_for_generation, parse_json_from_markdown
from .providers.openai_provider import OpenAIProvider
from .prompts import text_generation_prompt


class LLMFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn kompatibilní transformer, který využívá LLM
    k objevení a extrakci sémantických příznaků z textu.
    """

    def __init__(self, provider=None, discover_kwargs=None):
        self.provider = provider
        self.discover_kwargs = discover_kwargs or {}
        self.discovered_features_ = None
        self.feature_names_ = []

    def fit(self, X, y=None):
        """
        Fáze 'fit' vezme trénovací texty (X) a pomocí LLM
        z nich objeví abstraktní sémantické příznaky.
        """
        # Pokud uživatel nepředá vlastního providera (např. lokální Ollama), použije se výchozí
        provider = self.provider or OpenAIProvider()

        # Objevení příznaků přímo ze seznamu textů
        self.discovered_features_ = discover_features_from_texts(
            texts_or_file=list(X),
            provider=provider,
            as_set=True,
            **self.discover_kwargs
        )

        # Extrakce názvů příznaků pro budoucí použití
        features_list = self.discovered_features_
        if isinstance(features_list, dict):
            features_list = features_list.get("proposed_features", [])

        self.feature_names_ = []
        for feat in features_list:
            if isinstance(feat, dict) and "feature" in feat:
                self.feature_names_.append(feat["feature"])
            elif isinstance(feat, str):
                self.feature_names_.append(feat)

        return self

    def transform(self, X, y=None):
        """
        Fáze 'transform' vezme libovolné texty a pomocí LLM a objevených
        příznaků je převede do tabulkové podoby (pandas DataFrame).
        """
        if not self.discovered_features_:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")

        provider = self.provider or OpenAIProvider()
        full_prompt = _build_prompt_for_generation(text_generation_prompt, self.discovered_features_)

        results = []
        for text in X:
            try:
                llm_resp = provider.text_features([str(text)], prompt=full_prompt)
                parsed = llm_resp[0]

                # Očištění případného Markdown formátování
                if isinstance(parsed, dict) and "features" in parsed and isinstance(parsed["features"], str):
                    parsed = {"features": parse_json_from_markdown(parsed["features"])}

                inner = parsed.get("features", parsed) if isinstance(parsed, dict) else {}

                # Zajištění, že máme sloupce přesně odpovídající objeveným příznakům
                row = {}
                for feat in self.feature_names_:
                    row[feat] = inner.get(feat, None)
                results.append(row)

            except Exception as e:
                print(f"Warning: Failed to process text. Error: {e}")
                results.append({feat: None for feat in self.feature_names_})

        # Vrátíme tabulková data připravená pro klasický ML model
        return pd.DataFrame(results)