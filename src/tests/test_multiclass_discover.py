import pytest
from llm_feature_gen.discover import discover_features_from_texts


def test_multiclass_parameter_accepted():
    # Zkoušíme, zda funkce spadne, když jí předáme parametr classes
    # (mockujeme poskytovatele, aby nevolal reálné API)
    class MockProvider:
        def text_features(self, texts, prompt):
            # Uložíme si prompt, abychom v testu zkontrolovali, že se správně upravil
            self.last_prompt = prompt
            return [{"proposed_features": ["test_feature"]}]

    provider = MockProvider()
    classes = ["apple", "banana", "orange"]

    # Volání s naším novým parametrem
    discover_features_from_texts(
        texts_or_file=["text1", "text2"],
        provider=provider,
        classes=classes
    )

    # Ověření, že se Oxford comma pro 3 třídy správně vložila do promptu
    assert "'apple', 'banana', and 'orange'" in provider.last_prompt