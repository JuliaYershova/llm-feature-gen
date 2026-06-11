from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import llm_feature_gen.batch as batch_mod


class FakeBatchProvider:
    def __init__(self, fail_first: bool = False) -> None:
        self.calls = []
        self.fail_first = fail_first

    def text_features(self, text_list, prompt=None):
        self.calls.append({"texts": list(text_list), "prompt": prompt})
        if self.fail_first:
            self.fail_first = False
            raise RuntimeError("temporary failure")

        return [
            {"features": {"topic": f"value-{index}", "length": str(len(text))}}
            for index, text in enumerate(text_list)
        ]


class AlwaysFailProvider:
    def __init__(self) -> None:
        self.calls = []

    def text_features(self, text_list, prompt=None):
        self.calls.append({"texts": list(text_list), "prompt": prompt})
        raise RuntimeError("permanent failure")


class CountingCache(batch_mod.BatchTextCache):
    def __init__(self, cache_file: Path) -> None:
        self.save_calls = 0
        super().__init__(cache_file)

    def save(self) -> None:
        self.save_calls += 1
        super().save()


def discovered_features():
    return {"proposed_features": [{"feature": "topic"}, {"feature": "length"}]}


def test_batch_cache_round_trip_and_clear(tmp_path: Path):
    cache_file = tmp_path / "cache.json"
    cache = batch_mod.BatchTextCache(cache_file)
    cache.set("hello", "schema", {"topic": "greeting"})

    reloaded = batch_mod.BatchTextCache(cache_file)
    assert reloaded.get("hello", "schema") == {"topic": "greeting"}
    assert reloaded.get("other", "schema") is None
    assert len(reloaded) == 1

    reloaded.clear()
    assert len(reloaded) == 0
    assert not cache_file.exists()


def test_batch_cache_handles_invalid_json_and_deferred_persistence(tmp_path: Path):
    cache_file = tmp_path / "cache.json"
    cache_file.write_text("{not-json", encoding="utf-8")

    cache = batch_mod.BatchTextCache(cache_file)
    assert len(cache) == 0

    cache.set("hello", "schema", {"topic": "deferred"}, persist=False)
    assert cache_file.read_text(encoding="utf-8") == "{not-json"

    cache.save()
    assert json.loads(cache_file.read_text(encoding="utf-8"))

    cache.clear()
    cache.clear()
    assert len(cache) == 0


def test_normalise_provider_response_variants():
    assert batch_mod._normalise_provider_response({"features": '{"topic": "parsed"}'}) == {"topic": "parsed"}
    assert batch_mod._normalise_provider_response({"features": "not json"}) == {}
    assert batch_mod._normalise_provider_response({"features": ["not", "dict"]}) == {}
    assert batch_mod._normalise_provider_response(["not", "dict"]) == {}
    assert batch_mod._normalise_provider_response({"topic": "flat"}) == {"topic": "flat"}


def test_generate_features_batch_batches_provider_calls_and_saves_cache_once_per_batch(tmp_path: Path):
    provider = FakeBatchProvider()
    cache = CountingCache(tmp_path / "cache.json")
    output_csv = tmp_path / "features.csv"

    df = batch_mod.generate_features_batch(
        texts=["alpha", "beta", "gamma"],
        labels=["A", "B", "C"],
        discovered_features=discovered_features(),
        provider=provider,
        batch_size=2,
        output_csv=output_csv,
        cache=cache,
    )

    assert [call["texts"] for call in provider.calls] == [["alpha", "beta"], ["gamma"]]
    assert cache.save_calls == 2
    assert output_csv.exists()
    assert list(df["File"]) == ["text_0", "text_1", "text_2"]
    assert list(df["Class"]) == ["A", "B", "C"]
    assert list(df["topic"]) == ["value-0", "value-1", "value-0"]
    assert pd.read_csv(output_csv).shape[0] == 3


def test_generate_features_batch_loads_schema_from_path_and_uses_default_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    discovered_path = tmp_path / "discovered.json"
    discovered_path.write_text(json.dumps(discovered_features()), encoding="utf-8")
    provider = FakeBatchProvider()
    monkeypatch.setattr(batch_mod, "OpenAIProvider", lambda: provider)
    monkeypatch.setattr(batch_mod, "_tqdm", None)

    df = batch_mod.generate_features_batch(
        texts=["alpha"],
        labels=["A"],
        discovered_features=discovered_path,
    )

    assert provider.calls[0]["texts"] == ["alpha"]
    assert list(df["topic"]) == ["value-0"]


def test_generate_features_batch_reuses_cached_results_and_skips_provider(tmp_path: Path):
    cache = batch_mod.BatchTextCache(tmp_path / "cache.json")
    schema = discovered_features()
    features_hash = batch_mod.BatchTextCache._hash(json.dumps(schema, sort_keys=True))
    cache.set("alpha", features_hash, {"topic": "cached", "length": "5"})

    provider = FakeBatchProvider()
    df = batch_mod.generate_features_batch(
        texts=["alpha"],
        labels=["A"],
        discovered_features=schema,
        provider=provider,
        batch_size=10,
        cache=cache,
    )

    assert provider.calls == []
    assert list(df["topic"]) == ["cached"]
    assert list(df["length"]) == ["5"]


def test_generate_features_batch_validates_inputs_and_retries_once(tmp_path: Path):
    with pytest.raises(ValueError, match="same length"):
        batch_mod.generate_features_batch(
            texts=["alpha"],
            labels=[],
            discovered_features=discovered_features(),
            provider=FakeBatchProvider(),
        )

    with pytest.raises(ValueError, match="batch_size"):
        batch_mod.generate_features_batch(
            texts=["alpha"],
            labels=["A"],
            discovered_features=discovered_features(),
            provider=FakeBatchProvider(),
            batch_size=0,
        )

    with pytest.raises(ValueError, match="No feature names"):
        batch_mod.generate_features_batch(
            texts=["alpha"],
            labels=["A"],
            discovered_features={"proposed_features": []},
            provider=FakeBatchProvider(),
        )

    provider = FakeBatchProvider(fail_first=True)
    df = batch_mod.generate_features_batch(
        texts=["alpha"],
        labels=["A"],
        discovered_features=discovered_features(),
        provider=provider,
        retry_delay=0,
    )
    assert len(provider.calls) == 2
    assert list(df["topic"]) == ["value-0"]


def test_generate_features_batch_handles_second_retry_failure_and_short_response_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    failing_provider = AlwaysFailProvider()
    df = batch_mod.generate_features_batch(
        texts=["alpha"],
        labels=["A"],
        discovered_features=discovered_features(),
        provider=failing_provider,
        retry_delay=0,
    )

    assert len(failing_provider.calls) == 2
    assert list(df["topic"]) == ["not given by LLM"]

    class ShortResponseProvider:
        def text_features(self, text_list, prompt=None):
            return [{"features": {"topic": "only-first", "length": "5"}}]

    seen = {}

    def fake_tqdm(values, desc=None, unit=None, total=None):
        seen.update({"values": values, "desc": desc, "unit": unit, "total": total})
        return values

    monkeypatch.setattr(batch_mod, "_tqdm", fake_tqdm)
    df = batch_mod.generate_features_batch(
        texts=["alpha", "beta"],
        labels=["A", "B"],
        discovered_features=discovered_features(),
        provider=ShortResponseProvider(),
        batch_size=2,
    )

    assert seen == {"values": [0], "desc": "Batch generation", "unit": "batch", "total": 1}
    assert list(df["topic"]) == ["only-first", "not given by LLM"]


def test_generate_features_from_texts_cached_writes_per_class_and_merged_csv(tmp_path: Path):
    root = tmp_path / "root"
    class_a = root / "A"
    class_b = root / "B"
    class_a.mkdir(parents=True)
    class_b.mkdir()
    (class_a / "a1.txt").write_text("alpha", encoding="utf-8")
    (class_b / "b1.txt").write_text("beta", encoding="utf-8")

    discovered_path = tmp_path / "discovered.json"
    discovered_path.write_text(json.dumps(discovered_features()), encoding="utf-8")

    result = batch_mod.generate_features_from_texts_cached(
        root_folder=root,
        discovered_features_path=discovered_path,
        provider=FakeBatchProvider(),
        output_dir=tmp_path / "out",
        batch_size=1,
    )

    assert set(result) == {"A", "B", "__merged__"}
    assert pd.read_csv(result["A"])["File"].tolist() == ["a1.txt"]
    assert pd.read_csv(result["B"])["File"].tolist() == ["b1.txt"]
    assert pd.read_csv(result["__merged__"]).shape[0] == 2


def test_generate_features_from_texts_cached_uses_default_provider_and_custom_cache_without_merge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "root"
    class_a = root / "A"
    class_a.mkdir(parents=True)
    (class_a / "a1.txt").write_text("alpha", encoding="utf-8")
    discovered_path = tmp_path / "discovered.json"
    discovered_path.write_text(json.dumps(discovered_features()), encoding="utf-8")
    provider = FakeBatchProvider()
    monkeypatch.setattr(batch_mod, "OpenAIProvider", lambda: provider)

    result = batch_mod.generate_features_from_texts_cached(
        root_folder=root,
        discovered_features_path=discovered_path,
        output_dir=tmp_path / "out",
        merge_to_single_csv=False,
        cache_file=tmp_path / "custom-cache.json",
    )

    assert set(result) == {"A"}
    assert (tmp_path / "custom-cache.json").exists()
    assert provider.calls[0]["texts"] == ["alpha"]


def test_generate_features_from_texts_cached_handles_empty_class_selection(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    discovered_path = tmp_path / "discovered.json"
    discovered_path.write_text(json.dumps(discovered_features()), encoding="utf-8")

    result = batch_mod.generate_features_from_texts_cached(
        root_folder=root,
        discovered_features_path=discovered_path,
        provider=FakeBatchProvider(),
        classes=[],
        output_dir=tmp_path / "out",
        merge_to_single_csv=True,
    )

    assert result == {}
    assert not (tmp_path / "out" / "all_feature_values.csv").exists()


def test_generate_features_from_texts_cached_rejects_missing_class(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    discovered_path = tmp_path / "discovered.json"
    discovered_path.write_text(json.dumps(discovered_features()), encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        batch_mod.generate_features_from_texts_cached(
            root_folder=root,
            discovered_features_path=discovered_path,
            provider=FakeBatchProvider(),
            classes=["missing"],
        )
