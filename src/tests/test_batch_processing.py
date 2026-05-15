import pytest
import time
from llm_feature_gen.generate import assign_feature_values_from_folder


class MockSlowProvider:
    """Umělý provider, který simuluje pomalé API (odpověď trvá půl sekundy)."""

    def text_features(self, texts, prompt):
        time.sleep(0.5)
        return [{"features": {"test_feature": 1}}]


def test_batch_processing_speed(tmp_path):
    # 1. Příprava umělých složek a souborů nanečisto
    dataset_dir = tmp_path / "dataset"
    class_dir = dataset_dir / "test_class"
    class_dir.mkdir(parents=True)
    output_dir = tmp_path / "outputs"

    # Vytvoříme 4 fiktivní textové soubory (txt)
    for i in range(4):
        file_path = class_dir / f"doc_{i}.txt"
        file_path.write_text(f"Dummy content {i}", encoding="utf-8")

    provider = MockSlowProvider()
    discovered_features = {"proposed_features": ["test_feature"]}

    # 2. Měření času spuštění
    start_time = time.time()

    assign_feature_values_from_folder(
        folder_path=dataset_dir,
        class_name="test_class",
        discovered_features=discovered_features,
        provider=provider,
        output_dir=output_dir
    )

    duration = time.time() - start_time

    # 3. Vyhodnocení (Ověření, že paralelizace funguje)
    # Sériově: 4 * 0.5s = >2.0s
    # Paralelně: ~0.5s (Testujeme s bezpečnou rezervou pod 1.5s)
    assert duration < 1.5, f"Zpracování trvalo {duration:.2f} s. Paralelizace pravděpodobně nefunguje!"

    # Ověření, že se výstupní CSV opravdu vygenerovalo
    csv_file = output_dir / "test_class_feature_values.csv"
    assert csv_file.exists()