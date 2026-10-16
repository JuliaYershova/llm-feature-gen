"""Offline coverage of exhaustive discovery, retention, and resumable artifacts."""

import copy
import json
import random
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from llm_feature_gen import discover as discovery
from llm_feature_gen import discovery_map_reduce as engine
from llm_feature_gen import multiclass
from llm_feature_gen.batch import generate_features_batch
from llm_feature_gen.generate import generate_features_from_texts, load_discovered_features
from llm_feature_gen.sklearn import LLMFeatureTransformer


MODEL_ID = engine._provider_item_id("text:0")


def feature(name="topic", evidence=None, description="Observed topic", values=None):
    return {"feature": name, "description": description,
            "possible_values": values or ["ordinary", "rare"],
            "example_item_ids": evidence or [MODEL_ID]}


def merge_group(candidates):
    return {"keep": candidates[0]["id"], "merge_ids": [candidate["id"] for candidate in candidates],
            "description": "Canonical observable property",
            "possible_values": list(dict.fromkeys(value for candidate in candidates for value in candidate["possible_values"])),
            "value_mappings": [{"candidate_id": candidate["id"], "source_value": value, "canonical_value": value}
                               for candidate in candidates for value in candidate["possible_values"]]}


class FakeProvider:
    supports_response_schema = True
    cache_identity = {"model": "offline-v1"}

    def __init__(self, respond=None):
        self.calls = []
        self.respond = respond

    def _respond(self, request, prompt, kwargs):
        self.calls.append({"request": copy.deepcopy(request), "prompt": prompt, **kwargs})
        if self.respond is not None:
            return self.respond(request)
        if request["mode"] == "reduce":
            return [{"merge_groups": []}]
        ids = [item["item_id"] for item in request["items"]]
        features = [feature(evidence=ids)]
        rare = [item["item_id"] for item in request["items"] if "rare" in item["text"]]
        if rare:
            features.append(feature("rare_detail", rare, "Uncommon explicit detail"))
        return [{"processed_item_ids": ids, "proposed_features": features}]

    def text_features(self, texts, prompt=None, **kwargs):
        if "DISCOVERED_FEATURES_SPEC:" in prompt:
            schema = json.loads(prompt.split("DISCOVERED_FEATURES_SPEC:")[1])
            return [{"features": {spec["feature"]: "ordinary" for spec in schema["proposed_features"]}}
                    for text in texts]
        return self._respond(json.loads(texts[0]), prompt, kwargs)

    def image_features(self, images, prompt=None, extra_context=None, **kwargs):
        return self._respond(json.loads(extra_context), prompt, {"images": images, **kwargs})

    def usage_summary(self):
        return {"calls": len(self.calls)}


def discover(tmp_path, texts=None, provider=None, **kwargs):
    return discovery.discover_features_from_texts(
        texts if texts is not None else ["first", "second", "rare last"],
        provider=provider or FakeProvider(), output_dir=tmp_path,
        strategy="map_reduce", **kwargs,
    )


def report(tmp_path, name="discovered_text_features"):
    return json.loads((tmp_path / f"{name}_discovery_report.json").read_text())


def test_all_inputs_and_rare_last_batch_survive_and_generate_csv(tmp_path):
    provider = FakeProvider()
    schema = discover(tmp_path, provider=provider, batch_size=2, min_features=2, shuffle_seed=None)
    maps = [call for call in provider.calls if call["request"]["mode"] == "map"]
    assert [len(call["request"]["items"]) for call in maps] == [2, 1]
    assert maps[-1]["request"]["items"][0]["text"] == "rare last"
    assert maps[0]["response_schema"]["required"] == ["processed_item_ids", "proposed_features"]
    assert [spec["feature"] for spec in schema["proposed_features"]] == ["topic", "rare_detail"]
    saved = report(tmp_path)
    assert saved["processed_item_ids"] == ["text:0", "text:1", "text:2"]
    assert saved["features"]["topic"]["example_item_ids"] == ["text:0", "text:1", "text:2"]
    assert saved["features"]["rare_detail"]["example_item_ids"] == ["text:2"]
    assert saved["candidate_count"] == 3
    assert saved["warnings"] == []
    assert load_discovered_features(tmp_path / "discovered_text_features.json") == schema
    source = tmp_path / "data" / "class_a"
    source.mkdir(parents=True)
    (source / "a.txt").write_text("first")
    paths = generate_features_from_texts(source.parent, discovered_features_path=tmp_path / "discovered_text_features.json",
                                        provider=provider, output_dir=tmp_path / "csv")
    assert pd.read_csv(paths["class_a"])["rare_detail"].tolist() == ["ordinary"]


def test_semantic_merge_unions_evidence_and_preserves_conflicting_names(tmp_path):
    def respond(request):
        if request["mode"] == "map":
            item = request["items"][0]
            return [{"processed_item_ids": [item["item_id"]], "proposed_features": [
                feature("topic", [item["item_id"]], item["text"],
                        ["different"] if item["text"] == "conflict" else None)
            ]}]
        candidates = request["candidates"]
        compatible = [c for c in candidates if len(c["possible_values"]) == 2]
        return [{"merge_groups": [merge_group(compatible)]}]

    schema = discover(tmp_path, ["meaning", "synonym", "conflict", "another conflict"],
                      FakeProvider(respond), batch_size=1)
    assert [f["feature"] for f in schema["proposed_features"]] == ["topic", "topic_2"]
    compatible = next(spec for spec in schema["proposed_features"] if spec["possible_values"] == ["ordinary", "rare"])
    assert report(tmp_path)["features"][compatible["feature"]]["example_item_ids"] == ["text:0", "text:1", "text:3"]
    assert any("Renamed" in warning for warning in report(tmp_path)["warnings"])


def test_nonshrinking_catalog_uses_two_bounded_sweeps(tmp_path):
    def respond(request):
        if request["mode"] == "reduce":
            assert len(request["candidates"]) <= 2
            return [{"merge_groups": []}]
        item = request["items"][0]
        return [{"processed_item_ids": [item["item_id"]], "proposed_features": [
            feature(item["text"], [item["item_id"]])]}]

    schema = discover(tmp_path, ["a", "b", "c", "d", "e"], FakeProvider(respond),
                      batch_size=1, reduce_batch_size=2)
    assert len(schema["proposed_features"]) == 5
    assert report(tmp_path)["reduce_rounds"] == 2
    assert report(tmp_path)["provider_calls"] == 9


def test_resume_and_revalidate_corrupt_or_stale_checkpoints(tmp_path):
    provider = FakeProvider()
    options = {"batch_size": 1, "checkpoint_dir": tmp_path / "checkpoints"}
    schema = discover(tmp_path, provider=provider, **options)
    count = len(provider.calls)
    assert discover(tmp_path, provider=provider, **options) == schema
    assert len(provider.calls) == count
    assert report(tmp_path)["cached_steps"] == count
    checkpoint = next((tmp_path / "checkpoints").glob("*/map/0000.json"))
    for saved in ("invalid json", '{"input_hash": "stale"}'):
        checkpoint.write_text(saved)
        discover(tmp_path, provider=provider, **options)
        assert report(tmp_path)["provider_calls"] == 1
    saved = json.loads(checkpoint.read_text())
    saved["response"]["processed_item_ids"] = []
    checkpoint.write_text(json.dumps(saved))
    discover(tmp_path, provider=provider, **options)
    assert report(tmp_path)["provider_calls"] == 1
    discover(tmp_path, provider=provider, prompt="changed task", **options)
    assert report(tmp_path)["cached_steps"] == 0
    provider.cache_identity = {"model": "offline-v2"}
    discover(tmp_path, provider=provider, **options)
    assert report(tmp_path)["cached_steps"] == 0
    discover(tmp_path, ["changed", "second", "rare last"], provider, **options)
    assert report(tmp_path)["cached_steps"] == 0


@pytest.mark.parametrize("modality", ["text", "image", "tabular"])
def test_map_shuffle_mixes_sorted_classes_and_preserves_coverage(tmp_path, modality):
    source = tmp_path / "corpus"
    expected, labels = [], {}
    suffix = {"text": ".txt", "image": ".png", "tabular": ".csv"}[modality]
    for label, count in (("class_a", 3), ("class_b", 3), ("minority", 1)):
        folder = source / label
        folder.mkdir(parents=True)
        for index in range(count):
            path = folder / f"{index}{suffix}"
            if modality == "image":
                Image.new("RGB", (3, 3), (index, 0, 0)).save(path)
            elif modality == "tabular":
                pd.DataFrame({"body": [f"item {index}"]}).to_csv(path, index=False)
            else:
                path.write_text(f"item {index}")
            item_id = path.relative_to(source).as_posix()
            item_id += {"text": ":chunk:0", "image": "", "tabular": ":row:0"}[modality]
            expected.append(item_id)
            labels[engine._provider_item_id(item_id)] = label
    api_suffix = {"text": "texts", "image": "images", "tabular": "tabular"}[modality]
    function = getattr(discovery, f"discover_features_from_{api_suffix}")
    extra = {"text_column": "body"} if modality == "tabular" else {}
    provider = FakeProvider()
    function(source, provider=provider, strategy="map_reduce", batch_size=3, output_dir=tmp_path, **extra)
    maps = [call["request"]["items"] for call in provider.calls if call["request"]["mode"] == "map"]
    assert [len(batch) for batch in maps] == [3, 3, 1]
    assert all(len({labels[item["item_id"]] for item in batch}) > 1 for batch in maps[:-1])
    saved = report(tmp_path, f"discovered_{modality}_features")
    assert sorted(saved["processed_item_ids"]) == sorted(expected)
    assert saved["processed_item_ids"] != expected
    assert saved["config"]["shuffle_seed"] == 0
    function(source, provider=FakeProvider(), strategy="map_reduce", batch_size=3,
             output_dir=tmp_path, shuffle_seed=1, **extra)
    assert report(tmp_path, f"discovered_{modality}_features")["processed_item_ids"] != saved["processed_item_ids"]
    function(source, provider=FakeProvider(), strategy="map_reduce", batch_size=3,
             output_dir=tmp_path, shuffle_seed=None, **extra)
    assert report(tmp_path, f"discovered_{modality}_features")["processed_item_ids"] == expected


@pytest.mark.parametrize("shuffle_seed", [0, 19, None])
def test_shuffle_is_repeatable_and_preserves_inputs_random_state_and_frames(tmp_path, shuffle_seed):
    items = [{"item_id": f"item:{i}", "text": str(i), "images": [f"{i}:{frame}" for frame in range(i + 1)]}
             for i in range(4)]
    original = copy.deepcopy(items)
    by_id = {engine._provider_item_id(item["item_id"]): item for item in items}
    state = random.getstate()
    runs = []
    for _ in range(2):
        provider = FakeProvider()
        engine.discover_map_reduce(
            items, provider=provider, prompt="Inspect", system_prompt=None,
            output_path=tmp_path / "features.json", batch_size=2, reduce_batch_size=32,
            max_request_chars=24000, checkpoint_dir=None, reduce_prompt=None,
            min_features=1, modality="image", exclusions=[], extraction_settings={},
            shuffle_seed=shuffle_seed,
        )
        runs.append([call for call in provider.calls if call["request"]["mode"] == "map"])
    assert runs[0] == runs[1]
    assert items == original
    assert random.getstate() == state
    for call in runs[0]:
        assert len(call["request"]["items"]) == 2
        assert len(engine._json(call["request"])) + len("Inspect") + 600 <= 24000
        for item in call["request"]["items"]:
            original_item = by_id[item["item_id"]]
            start, count = item["frame_start"], item["frame_count"]
            assert call["images"][start:start + count] == original_item["images"]
            assert item["text"] == original_item["text"]


def test_shuffle_seed_scopes_checkpoints_even_when_item_order_matches(tmp_path):
    provider = FakeProvider()
    signatures = set()
    for seed in (0, 1, None):
        options = {"shuffle_seed": seed, "checkpoint_dir": tmp_path / "checkpoints"}
        discover(tmp_path, ["one"], provider, **options)
        saved = report(tmp_path)
        assert saved["config"]["shuffle_seed"] == seed
        assert saved["cached_steps"] == 0
        signatures.add(saved["signature"])
        discover(tmp_path, ["one"], provider, **options)
        assert report(tmp_path)["cached_steps"] == 1
        assert report(tmp_path)["provider_calls"] == 0
    assert len(signatures) == 3


def test_interrupted_run_resumes_and_preserves_previous_schema(tmp_path):
    original = discover(tmp_path)
    provider = FakeProvider()
    normal = provider.text_features

    def interrupted(texts, **kwargs):
        if len(provider.calls) == 1:
            raise RuntimeError("server down")
        return normal(texts, **kwargs)

    provider.text_features = interrupted
    options = {"batch_size": 1, "checkpoint_dir": tmp_path / "checkpoints"}
    with pytest.raises(RuntimeError, match="server down"):
        discover(tmp_path, provider=provider, **options)
    assert load_discovered_features(tmp_path / "discovered_text_features.json") == original
    provider.text_features = normal
    assert discover(tmp_path, provider=provider, **options) == original
    assert report(tmp_path)["cached_steps"] == 1


@pytest.mark.parametrize("invalid", [
    [], {"error": "failed"}, [{"processed_item_ids": [], "proposed_features": []}],
    [{"processed_item_ids": [MODEL_ID, MODEL_ID], "proposed_features": []}],
    [{"processed_item_ids": [MODEL_ID], "proposed_features": [feature(), feature()]}],
    [{"processed_item_ids": [MODEL_ID], "proposed_features": [feature(evidence=["unknown"])]}],
    [{"processed_item_ids": [MODEL_ID], "proposed_features": [feature(evidence=[MODEL_ID, MODEL_ID])]}],
    [{"processed_item_ids": [MODEL_ID], "proposed_features": [feature(values=["same", "same"])]}],
    [{"processed_item_ids": [MODEL_ID], "proposed_features": [{}, {}, {}]}],
])
def test_invalid_map_retries_with_feedback_then_fails(tmp_path, invalid):
    provider = FakeProvider(lambda request: invalid)
    with pytest.raises(ValueError, match="Invalid discovery map batch"):
        discover(tmp_path, ["one"], provider)
    assert len(provider.calls) == 2
    assert "validation_error" in provider.calls[1]["request"]
    assert len(engine._json(provider.calls[1]["request"]["validation_error"])) <= 400
    assert not (tmp_path / "discovered_text_features.json").exists()


def test_invalid_map_can_recover_and_empty_batch_is_allowed(tmp_path):
    provider = FakeProvider()

    def respond(request):
        ids = [item["item_id"] for item in request["items"]]
        if len(provider.calls) == 1:
            return [{"processed_item_ids": [], "proposed_features": []}]
        return [{"processed_item_ids": ids, "proposed_features": [] if ids == [MODEL_ID] else [feature(evidence=ids)]}]

    provider.respond = respond
    assert discover(tmp_path, ["one", "two"], provider, batch_size=1)
    with pytest.raises(ValueError, match="no evidenced features"):
        discover(tmp_path, ["one"], FakeProvider(lambda request: [{"processed_item_ids": [MODEL_ID], "proposed_features": []}]))
    with pytest.raises(ValueError, match="No eligible"):
        discover(tmp_path, [])


@pytest.mark.parametrize("changes", [
    None, {"keep": "unknown"}, {"merge_ids": ["a", "a"]}, {"merge_ids": ["a", "unknown"]},
    {"description": " "}, {"possible_values": ["yes", "yes"]}, {"value_mappings": []},
    {"value_mappings": [
        {"candidate_id": "a", "source_value": "yes", "canonical_value": "yes"},
        {"candidate_id": "a", "source_value": "yes", "canonical_value": "no"},
    ]},
    {"value_mappings": [
        {"candidate_id": "a", "source_value": "yes", "canonical_value": "yes"},
        {"candidate_id": "b", "source_value": "unknown", "canonical_value": "no"},
    ]},
    {"value_mappings": [
        {"candidate_id": "a", "source_value": "yes", "canonical_value": "yes"},
        {"candidate_id": "unknown", "source_value": "no", "canonical_value": "no"},
    ]},
    {"value_mappings": [
        {"candidate_id": "a", "source_value": "yes", "canonical_value": "yes"},
        {"candidate_id": "b", "source_value": "no", "canonical_value": "invented"},
    ]},
    {"possible_values": ["yes", "no", "unsupported"]},
])
def test_reduce_invalid_groups_preserve_candidates(changes):
    candidates = [{"id": "a", "possible_values": ["yes"]}, {"id": "b", "possible_values": ["no"]}]
    original = copy.deepcopy(candidates)
    group = {**merge_group(candidates), **changes} if changes is not None else None
    result = engine._validate_reduce({"merge_groups": [group]}, candidates)
    assert result["merge_groups"] == []
    assert result["warnings"]
    assert candidates == original


def test_invalid_reduce_is_retried_and_only_valid_groups_apply(tmp_path):
    provider = FakeProvider()

    def respond(request):
        if request["mode"] == "map":
            ids = [item["item_id"] for item in request["items"]]
            return [{"processed_item_ids": ids, "proposed_features": [feature(str(i), ids) for i in range(3)]}]
        group = merge_group(request["candidates"][:2])
        return [{"merge_groups": [group, group, None]}]

    provider.respond = respond
    schema = discover(tmp_path, ["one"], provider)
    assert len(schema["proposed_features"]) == 2
    assert len(provider.calls) == 3
    assert len(report(tmp_path)["warnings"]) >= 2
    bad = FakeProvider(lambda request: [{"processed_item_ids": [MODEL_ID], "proposed_features": [feature(), feature("other")]}]
                       if request["mode"] == "map" else [{}])
    with pytest.raises(ValueError, match="merge_groups"):
        discover(tmp_path, ["one"], bad)


def test_reduce_reconciles_synonyms_and_partial_vocabularies_and_generates(tmp_path):
    def respond(request):
        if request["mode"] == "map":
            item = request["items"][0]
            values = ["straight", "curved"] if item["text"] == "first" else ["linear", "curved", "unknown"]
            return [{"processed_item_ids": [item["item_id"]], "proposed_features": [
                feature("beak_shape", [item["item_id"]], "Observed beak shape", values)
            ]}]
        group = merge_group(request["candidates"])
        group["description"] = "Beak curvature; linear and straight describe the same shape"
        group["possible_values"] = ["straight", "curved", "unknown"]
        for mapping in group["value_mappings"]:
            if mapping["source_value"] == "linear":
                mapping["canonical_value"] = "straight"
        return [{"merge_groups": [group]}]

    provider = FakeProvider(respond)
    schema = discover(tmp_path, ["first", "second"], provider, batch_size=1, shuffle_seed=None, min_features=1)
    assert schema["proposed_features"] == [{
        "feature": "beak_shape", "description": "Beak curvature; linear and straight describe the same shape",
        "possible_values": ["straight", "curved", "unknown"],
    }]
    saved = report(tmp_path)
    assert saved["warnings"] == []
    assert saved["features"]["beak_shape"]["example_item_ids"] == ["text:0", "text:1"]
    assert saved["features"]["beak_shape"]["value_mappings"] == {
        "map_0000:feature_0000": {"straight": "straight", "curved": "curved"},
        "map_0001:feature_0000": {"linear": "straight", "curved": "curved", "unknown": "unknown"},
    }

    def generate(texts, prompt=None, response_schema=None):
        assert json.loads(prompt.split("DISCOVERED_FEATURES_SPEC:")[1]) == schema
        assert "linear" not in response_schema["properties"]["beak_shape"]["enum"]
        return [{"features": {"beak_shape": "straight"}} for text in texts]

    provider.text_features = generate
    result = generate_features_batch(["new bird"], ["bird"], tmp_path / "discovered_text_features.json", provider)
    assert result["beak_shape"].tolist() == ["straight"]


def test_reduce_composes_value_mappings_across_exact_merges_rounds_and_checkpoints(tmp_path):
    aliases = {"linear": "straight", "straight": "straight", "bent": "curved", "curved": "curved"}

    def respond(request):
        if request["mode"] == "map":
            item = request["items"][0]
            return [{"processed_item_ids": [item["item_id"]], "proposed_features": [
                feature("beak_shape", [item["item_id"]], f"Observed {item['text']} shape", [item["text"]])
            ]}]
        group = merge_group(request["candidates"])
        for mapping in group["value_mappings"]:
            mapping["canonical_value"] = aliases[mapping["source_value"]]
        group["possible_values"] = list(dict.fromkeys(mapping["canonical_value"] for mapping in group["value_mappings"]))
        return [{"merge_groups": [group]}]

    provider = FakeProvider(respond)
    texts = ["linear", "linear", "straight", "bent", "curved"]
    options = {"batch_size": 1, "reduce_batch_size": 2, "shuffle_seed": None,
               "checkpoint_dir": tmp_path / "checkpoints", "min_features": 1}
    schema = discover(tmp_path, texts, provider, **options)
    saved = report(tmp_path)
    assert schema["proposed_features"][0]["possible_values"] == ["straight", "curved"]
    assert len(schema["proposed_features"]) == 1
    assert saved["reduce_rounds"] == 2
    assert saved["features"]["beak_shape"]["support"] == 5
    assert sorted(saved["features"]["beak_shape"]["source_ids"]) == [f"map_{i:04d}:feature_0000" for i in range(5)]
    assert saved["features"]["beak_shape"]["value_mappings"] == {
        f"map_{i:04d}:feature_0000": {value: aliases[value]} for i, value in enumerate(texts)
    }
    calls = len(provider.calls)
    assert discover(tmp_path, texts, provider, **options) == schema
    resumed = report(tmp_path)
    assert len(provider.calls) == calls
    assert resumed["features"] == saved["features"]
    assert resumed["cached_steps"] == calls


def test_invalid_reconciliation_retries_then_retains_candidates(tmp_path):
    def respond(request):
        if request["mode"] == "map":
            ids = [item["item_id"] for item in request["items"]]
            return [{"processed_item_ids": ids, "proposed_features": [
                feature("beak_shape", ids, "Beak length", ["short", "long"]),
                feature("beak_shape_2", ids, "Beak curvature", ["straight", "curved"]),
            ]}]
        group = merge_group(request["candidates"])
        group["value_mappings"].pop()
        return [{"merge_groups": [group]}]

    provider = FakeProvider(respond)
    schema = discover(tmp_path, ["bird"], provider, min_features=2)
    assert [(spec["description"], spec["possible_values"]) for spec in schema["proposed_features"]] == [
        ("Beak length", ["short", "long"]), ("Beak curvature", ["straight", "curved"]),
    ]
    assert len(provider.calls) == 3
    assert "validation_error" in provider.calls[-1]["request"]
    assert any("exactly once" in warning for warning in report(tmp_path)["warnings"])


def test_reduce_keeps_different_dimensions_when_no_equivalence_is_proposed(tmp_path):
    def respond(request):
        if request["mode"] == "reduce":
            return [{"merge_groups": []}]
        ids = [item["item_id"] for item in request["items"]]
        return [{"processed_item_ids": ids, "proposed_features": [
            feature("beak_length", ids, "Beak length", ["short", "long"]),
            feature("beak_curvature", ids, "Beak curvature", ["straight", "curved"]),
        ]}]

    schema = discover(tmp_path, ["bird"], FakeProvider(respond), min_features=2)
    assert {spec["feature"] for spec in schema["proposed_features"]} == {"beak_length", "beak_curvature"}
    assert report(tmp_path)["warnings"] == []


def test_size_limits_bound_requests_and_reject_oversized_sources(tmp_path):
    entries = [{"item_id": f"text:{i}", "text": "x" * 100} for i in range(4)]
    batches = engine._pack(entries, 10, 950, "prompt", "role", "map")
    assert len(batches) == 2
    assert all(len(engine._json(engine._request("map", batch))) + 600 + 10 <= 950 for batch in batches)
    with pytest.raises(ValueError, match="text:0.*exceeds"):
        discover(tmp_path, ["x" * 24000])
    with pytest.raises(ValueError, match="'large'.*exceeds"):
        engine._pack([{"id": "large", "feature": "x", "description": "x" * 1000, "possible_values": ["x"], "example_item_ids": []}],
                     2, 700, "", None, "reduce")


@pytest.mark.parametrize("options, message", [
    ({"strategy": "other"}, "strategy"), ({"as_set": False}, "as_set"),
    ({"batch_size": 0}, "batch_size"), ({"reduce_batch_size": True}, "reduce_batch_size"),
    ({"max_request_chars": 1.5}, "max_request_chars"), ({"min_features": 0}, "min_features"),
    ({"shuffle_seed": True}, "shuffle_seed"), ({"shuffle_seed": 1.5}, "shuffle_seed"),
    ({"shuffle_seed": "0"}, "shuffle_seed"),
])
def test_bad_options_fail_before_provider_calls(tmp_path, options, message):
    provider = FakeProvider()
    options = {"strategy": "map_reduce", **options}
    with pytest.raises(ValueError, match=message):
        discovery.discover_features_from_texts(["one"], provider=provider, output_dir=tmp_path, **options)
    assert provider.calls == []


def test_text_sources_keep_identity_and_document_exclusions(tmp_path):
    source = tmp_path / "texts"
    source.mkdir()
    (source / "a.txt").write_text("first")
    (source / "b.md").write_text("rare last")
    (source / "empty.txt").write_text(" ")
    (source / "other.dat").write_text("skip")
    discover(tmp_path, source)
    saved = report(tmp_path)
    assert sorted(saved["processed_item_ids"]) == ["a.txt:chunk:0", "b.md:chunk:0"]
    assert {e["reason"] for e in saved["excluded_inputs"]} == {"empty", "unsupported_format"}
    discover(tmp_path, "raw prose")
    discover(tmp_path, str(source / "a.txt"))
    discover(tmp_path, ["", "one"])
    with pytest.raises(FileNotFoundError):
        discover(tmp_path, source / "missing.txt")
    (source / "bad.txt").write_bytes(b"\xff")
    with pytest.raises(ValueError, match="bad.txt"):
        discover(tmp_path, source)


def test_tabular_visits_all_rows_and_interleaves_files(tmp_path):
    source = tmp_path / "tables"
    source.mkdir()
    pd.DataFrame({"body": ["first", None, "rare last"]}).to_csv(source / "a.csv", index=False)
    pd.DataFrame({"body": ["second", " "]}).to_json(source / "b.json")
    provider = FakeProvider()
    schema = discovery.discover_features_from_tabular(source, "body", provider=provider, strategy="map_reduce",
                                                     output_dir=tmp_path, shuffle_seed=None)
    saved = report(tmp_path, "discovered_tabular_features")
    assert saved["processed_item_ids"] == ["a.csv:row:0", "b.json:row:0", "a.csv:row:2"]
    assert {e["reason"] for e in saved["excluded_inputs"]} == {"null", "empty"}
    assert len(schema["proposed_features"]) == 2
    with pytest.raises(ValueError, match="max_rows"):
        discovery.discover_features_from_tabular(source, "body", provider=provider, strategy="map_reduce", max_rows=1)
    with pytest.raises(ValueError, match="Column"):
        discovery.discover_features_from_tabular(source, "missing", provider=provider, strategy="map_reduce")


def test_images_keep_frame_source_positions_and_fail_on_bad_input(tmp_path):
    source = tmp_path / "images"
    source.mkdir()
    for i in range(3):
        Image.new("RGB", (3, 3), (i, 0, 0)).save(source / f"{i}.png")
    provider = FakeProvider()
    discovery.discover_features_from_images(source, provider=provider, strategy="map_reduce", batch_size=2, output_dir=tmp_path)
    first = provider.calls[0]
    assert len(first["images"]) == 2
    assert [item["frame_start"] for item in first["request"]["items"]] == [0, 1]
    discovery.discover_features_from_images([source / "0.png", source / "0.png"], provider=provider, strategy="map_reduce", output_dir=tmp_path)
    assert sorted(report(tmp_path, "discovered_image_features")["processed_item_ids"]) == ["0:0.png", "1:0.png"]
    (source / "bad.png").write_text("bad")
    with pytest.raises(ValueError, match="bad.png"):
        discovery.discover_features_from_images(source, provider=provider, strategy="map_reduce")


def test_videos_visit_every_file_without_global_downsampling(tmp_path, monkeypatch):
    source = tmp_path / "videos"
    source.mkdir()
    for i in range(6):
        (source / f"{i}.mp4").write_bytes(b"video")
    monkeypatch.setattr(discovery, "extract_key_frames", lambda path, frame_limit: [f"{path}:{i}" for i in range(frame_limit)])
    audio = tmp_path / "audio.wav"

    def extract_audio(path):
        audio.write_bytes(b"audio")
        return str(audio)

    monkeypatch.setattr(discovery, "extract_audio_track", extract_audio)
    provider = FakeProvider()
    provider.transcribe_audio = lambda path: "rare transcript"
    discovery.discover_features_from_videos(source, provider=provider, strategy="map_reduce", output_dir=tmp_path,
                                            num_frames=3, max_videos_to_sample=1, max_total_frames_payload=1)
    maps = [call for call in provider.calls if call["request"]["mode"] == "map"]
    assert len(maps) == 6
    assert all(len(call["images"]) == 3 and len(call["request"]["items"]) == 1 for call in maps)
    assert all(call["request"]["items"][0]["text"] == "rare transcript" for call in maps)
    assert not audio.exists()
    discovery.discover_features_from_videos(source / "0.mp4", provider=provider, strategy="map_reduce", output_dir=tmp_path, use_audio=False)
    monkeypatch.setattr(discovery, "extract_audio_track", lambda path: None)
    discovery.discover_features_from_videos(source / "0.mp4", provider=provider, strategy="map_reduce", output_dir=tmp_path)
    monkeypatch.setattr(discovery, "extract_key_frames", lambda path, frame_limit: [])
    with pytest.raises(ValueError, match="No frames"):
        discovery.discover_features_from_videos(source, provider=provider, strategy="map_reduce")


def test_custom_prompts_default_provider_and_checkpoint_identity(tmp_path, monkeypatch):
    provider = FakeProvider()
    monkeypatch.setattr(discovery, "OpenAIProvider", lambda: provider)
    discovery.discover_features_from_texts(["one"], strategy="map_reduce", output_dir=tmp_path,
        prompt="Inspect {n_classes} groups, request {min_features}: {class_list}", system_prompt="role", min_features=3)
    assert "Inspect 2 groups, request 3" in provider.calls[0]["prompt"]
    assert provider.calls[0]["system_prompt"] == "role"
    provider.cache_identity = None
    discover(tmp_path, provider=provider, checkpoint_dir=tmp_path / "disabled", reduce_prompt="Consolidate only equivalent topics")
    assert not (tmp_path / "disabled").exists()
    assert any("cache_identity" in warning for warning in report(tmp_path)["warnings"])
    assert "Consolidate only" in provider.calls[-1]["prompt"]
    provider.cache_identity = lambda: {"version": 3}
    assert engine._provider_identity(provider)["identity"] == {"version": 3}
    assert engine._provider_identity(object()) is None
    for type_ in (engine.LocalProvider, engine.OpenAIProvider):
        configured = object.__new__(type_)
        configured.text_model = "test-model"
        identity = engine._provider_identity(configured)
        assert identity["type"] == f"{type_.__module__}.{type_.__qualname__}"


def test_custom_provider_can_opt_out_of_discovery_checkpoints(tmp_path):
    provider = FakeProvider()
    provider.cache_identity = lambda: None
    options = {"provider": provider, "checkpoint_dir": tmp_path / "checkpoints"}
    assert engine._provider_identity(provider) is None
    discover(tmp_path, **options)
    calls = len(provider.calls)
    discover(tmp_path, **options)
    assert len(provider.calls) == calls * 2
    assert report(tmp_path)["cached_steps"] == 0
    assert not (tmp_path / "checkpoints").exists()


@pytest.mark.parametrize("setting", ["vision_model", "audio_model", "whisper_model_size", "api_version"])
def test_discovery_identity_includes_multimodal_settings(setting):
    provider = object.__new__(engine.LocalProvider)
    provider.text_model = "text-model"
    before = engine._provider_identity(provider)
    setattr(provider, setting, "changed")
    after = engine._provider_identity(provider)
    assert after[setting] == "changed"
    assert after != before


@pytest.mark.parametrize("modality", ["text", "image", "video", "tabular"])
@pytest.mark.parametrize("custom", [False, True])
def test_multiclass_forwards_options_and_context(tmp_path, monkeypatch, modality, custom):
    suffix = {"text": "texts", "image": "images", "video": "videos", "tabular": "tabular"}[modality]
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return {"proposed_features": []}

    monkeypatch.setattr(multiclass, f"discover_features_from_{suffix}", capture)
    function = multiclass.discover_features_multiclass if modality == "text" else getattr(multiclass, f"discover_features_multiclass_{suffix}")
    extra = {"text_column": "body"} if modality == "tabular" else {}
    function("source", classes=["A", "B", "C"], strategy="map_reduce", batch_size=2,
             reduce_batch_size=3, checkpoint_dir=tmp_path, max_request_chars=5000,
             reduce_prompt="merge", shuffle_seed=17, prompt="Custom {class_list}" if custom else None, **extra)
    assert captured["strategy"] == "map_reduce"
    assert captured["num_classes"] == 3
    assert captured["checkpoint_dir"] == tmp_path
    assert "  - C" in captured["prompt"]
    assert captured["max_request_chars"] == 5000
    assert captured["shuffle_seed"] == 17


def test_sklearn_discovers_once_and_preserves_discovery_options(tmp_path):
    provider = FakeProvider()
    transformer = LLMFeatureTransformer(provider=provider, output_dir=tmp_path,
        discovery_strategy="map_reduce", discovery_batch_size=1,
        discovery_checkpoint_dir=tmp_path / "checkpoints", discovery_shuffle_seed=17, min_features=2)
    result = transformer.fit_transform(["one", "rare last"], ["A", "B"])
    assert list(result.columns) == ["topic", "rare_detail"]
    calls = len(provider.calls)
    transformer.transform(["new"])
    assert len(provider.calls) == calls
    assert transformer.get_params()["discovery_strategy"] == "map_reduce"
    assert report(tmp_path)["config"]["shuffle_seed"] == 17
    transformer.discovered_features = transformer.discovered_features_
    transformer.fit(["supplied schema"])
    assert len(provider.calls) == calls


def test_map_reduce_sklearn_pipeline_and_clone(tmp_path):
    pytest.importorskip("sklearn")
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    transformer = LLMFeatureTransformer(
        provider=FakeProvider(), discovery_strategy="map_reduce",
        discovery_batch_size=1, discovery_reduce_batch_size=4,
        discovery_checkpoint_dir=tmp_path / "checkpoints", output_dir=tmp_path,
        discovery_shuffle_seed=17,
    )
    cloned = clone(transformer)
    assert cloned.get_params()["discovery_strategy"] == "map_reduce"
    assert cloned.discovery_checkpoint_dir == transformer.discovery_checkpoint_dir
    assert cloned.discovery_shuffle_seed == 17
    pipe = Pipeline([("llm", cloned), ("encode", OneHotEncoder(handle_unknown="ignore"))])
    assert pipe.fit_transform(["one", "rare last"], ["A", "B"]).shape == (2, 2)
    calls = len(cloned.provider_.calls)
    assert pipe.transform(["new input"]).shape == (1, 2)
    assert len(cloned.provider_.calls) == calls


def test_multiclass_pipeline_generates_train_and_test_with_one_schema(tmp_path):
    source = tmp_path / "discovery"
    source.mkdir()
    (source / "a.txt").write_text("rare last")
    for split in ("train", "test"):
        for label in ("A", "B"):
            folder = tmp_path / split / label
            folder.mkdir(parents=True)
            (folder / "item.txt").write_text("one")
    provider = FakeProvider()
    result = multiclass.run_multiclass_pipeline(
        source, tmp_path / "train", tmp_path / "test", ["A", "B"], provider,
        output_dir=tmp_path / "results", strategy="map_reduce", batch_size=1,
        checkpoint_dir=tmp_path / "checkpoints",
        shuffle_seed=17,
    )
    assert len(provider.calls) == 2
    assert report(tmp_path / "results")["config"]["shuffle_seed"] == 17
    for split in ("train", "test"):
        frame = pd.read_csv(result[f"{split}_csv_paths"]["__merged__"])
        assert len(frame) == 2
        assert {"topic", "rare_detail"} <= set(frame.columns)


def test_atomic_write_cleans_up_on_failure(tmp_path, monkeypatch):
    path = tmp_path / "saved.json"
    path.write_text("previous")

    def fail(source, destination):
        raise OSError("disk unavailable")

    monkeypatch.setattr(engine.os, "replace", fail)
    with pytest.raises(OSError, match="disk unavailable"):
        engine._write_json(path, {"new": "value"})
    assert path.read_text() == "previous"
    assert not list(tmp_path.glob("*.tmp"))


def test_report_without_usage_and_empty_packing(tmp_path):
    provider = FakeProvider()
    provider.usage_summary = None
    discover(tmp_path, provider=provider)
    assert "provider_usage" not in report(tmp_path)
    assert engine._pack([], 2, 24000, "task", None, "map") == []


def test_incomplete_lineage_does_not_publish_a_schema(tmp_path, monkeypatch):
    schema = discover(tmp_path)
    merge = engine._merge

    def lost_lineage(catalog, keep, sources):
        merge(catalog, keep, sources)
        catalog[keep]["source_ids"] = []

    monkeypatch.setattr(engine, "_merge", lost_lineage)
    with pytest.raises(ValueError, match="Incomplete discovery"):
        discover(tmp_path, batch_size=1)
    assert load_discovered_features(tmp_path / "discovered_text_features.json") == schema


def test_empty_extracted_document_is_reported(tmp_path, monkeypatch):
    source = tmp_path / "documents"
    source.mkdir()
    (source / "a.txt").write_text("one")
    (source / "empty.docx").write_bytes(b"empty document")
    monkeypatch.setattr(discovery, "extract_text_from_file", lambda path: ["one"] if path.suffix == ".txt" else [])
    discover(tmp_path, source)
    assert report(tmp_path)["excluded_inputs"] == [{"item_id": "empty.docx", "reason": "empty"}]


def test_final_names_cannot_overwrite_generation_metadata(tmp_path):
    def respond(request):
        if request["mode"] == "reduce":
            return [{"merge_groups": []}]
        ids = [item["item_id"] for item in request["items"]]
        return [{"processed_item_ids": ids, "proposed_features": [
            feature("raw_llm_output", ids), feature("Class", ids), feature("File", ids),
            feature("raw_llm_output_2", ids),
        ]}]

    schema = discover(tmp_path, ["one"], FakeProvider(respond))
    names = [spec["feature"] for spec in schema["proposed_features"]]
    assert names == ["raw_llm_output_2", "Class_2", "File_2", "raw_llm_output_2_2"]
    assert len(set(names)) == len(names)


def test_provider_requests_hide_class_named_paths_but_reports_keep_provenance(tmp_path):
    source = tmp_path / "corpus"
    for label, text in (("billing", "Need support"), ("access", "rare detail")):
        folder = source / label
        folder.mkdir(parents=True)
        (folder / "ticket.txt").write_text(text)
    provider = FakeProvider()
    discover(tmp_path, source, provider, batch_size=1)
    for call in provider.calls:
        serialized = json.dumps(call["request"])
        assert "billing" not in serialized
        assert "access" not in serialized
        assert "ticket.txt" not in serialized
    assert set(report(tmp_path)["processed_item_ids"]) == {
        "access/ticket.txt:chunk:0", "billing/ticket.txt:chunk:0",
    }
