"""Bounded, evidenced discovery with conservative reduction and checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from itertools import groupby
from pathlib import Path
from typing import Any, Dict, List, Optional

from .batch import _provider_identity as _cache_provider_identity
from .contracts import FeatureMergeGroup, MapDiscoveryResponse, ReduceDiscoveryResponse
from .generate import _provider_call_kwargs, _validate_discovered_schema
from .prompts import map_discovery_contract, reduce_discovery_contract
from .providers import LocalProvider, OpenAIProvider


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _provider_item_id(source_id: str) -> str:
    """Keep class names and filenames out of the discovery model's evidence IDs."""
    return f"item_{_digest(source_id)[:16]}"


def _write_json(path: Path, value: Any) -> None:
    """Replace an artifact only after its complete contents have been written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _validate_map(raw: Any, item_ids: set[str]) -> Dict[str, Any]:
    result = MapDiscoveryResponse.model_validate(raw).model_dump()
    processed = result["processed_item_ids"]
    if len(processed) != len(set(processed)) or set(processed) != item_ids:
        raise ValueError("processed_item_ids must include every input ID exactly once")
    names = [feature["feature"] for feature in result["proposed_features"]]
    if len(names) != len(set(names)):
        raise ValueError("Map feature names must be unique within a batch")
    features, warnings, retry_features = [], [], []
    for feature in result["proposed_features"]:
        evidence = feature["example_item_ids"]
        values = feature["possible_values"]
        if len(values) != len(set(values)):
            raise ValueError("Feature possible_values must be unique")
        known_evidence = list(dict.fromkeys(item_id for item_id in evidence if item_id in item_ids))
        if known_evidence != evidence:
            warnings.append(f"Removed unknown or duplicate evidence IDs from feature {feature['feature']!r}")
        feature["example_item_ids"] = known_evidence
        if known_evidence:
            features.append(feature)
        else:
            retry_features.append(feature["feature"])
    result.update(proposed_features=features, warnings=warnings, retry_features=retry_features,
                  errors=[f"Feature {name!r} needs at least one supplied evidence ID" for name in retry_features])
    return result


def _validate_reduce(raw: Any, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(raw, dict) or not isinstance(raw.get("merge_groups"), list):
        raise ValueError("Reduce response must contain a merge_groups array")
    known = {candidate["id"]: candidate for candidate in candidates}
    groups, errors, warnings, seen = [], [], [], set()
    for index, raw_group in enumerate(raw["merge_groups"]):
        try:
            # Some local models spell out unchanged candidates as singleton groups.
            if isinstance(raw_group, dict) and raw_group.get("merge_ids") == [raw_group.get("keep")]:
                if not isinstance(raw_group.get("keep"), str) or raw_group["keep"] not in known:
                    raise ValueError("Singleton group needs a known survivor ID")
                continue
            # Accept the old redundant vocabulary, but mappings are authoritative.
            supplied_values = raw_group.get("possible_values") if isinstance(raw_group, dict) else None
            payload = ({key: value for key, value in raw_group.items() if key != "possible_values"}
                       if isinstance(raw_group, dict) else raw_group)
            group = FeatureMergeGroup.model_validate(payload).model_dump()
            ids = group["merge_ids"]
            if (
                len(ids) != len(set(ids)) or group["keep"] not in ids
                or not set(ids) <= known.keys() or seen.intersection(ids)
            ):
                raise ValueError("Merge groups need distinct known IDs and a survivor, without overlap")
            pairs = [(mapping["candidate_id"], mapping["source_value"]) for mapping in group["value_mappings"]]
            expected = {(source, value) for source in ids for value in known[source]["possible_values"]}
            if len(pairs) != len(set(pairs)) or set(pairs) != expected:
                missing, extra = expected - set(pairs), set(pairs) - expected
                raise ValueError(
                    "value_mappings must cover every merged candidate value exactly once; "
                    f"missing={sorted(missing)!r}, extra={sorted(extra)!r}"
                )
            values = list(dict.fromkeys(mapping["canonical_value"] for mapping in group["value_mappings"]))
            if supplied_values is not None and supplied_values != values:
                warnings.append(f"Derived canonical vocabulary from mappings for survivor {group['keep']!r}")
            group["possible_values"] = values
            groups.append(group)
            seen.update(ids)
        except ValueError as exc:
            errors.append(f"Reduce group {index}: {exc}")
    return {"merge_groups": groups, "warnings": warnings, "errors": errors}


def _request(mode: str, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    if mode == "reduce":
        return {"mode": mode, "candidates": [
            {**{key: candidate[key] for key in ("id", "feature", "description", "possible_values")},
             "example_item_ids": [_provider_item_id(source) for source in candidate["example_item_ids"][:3]]}
            for candidate in entries
        ]}
    items, frame_start = [], 0
    for entry in entries:
        item = {"item_id": _provider_item_id(entry["item_id"]), "text": entry["text"]}
        if "images" in entry:
            item.update(frame_start=frame_start, frame_count=len(entry["images"]))
            frame_start += len(entry["images"])
        items.append(item)
    return {"mode": mode, "items": items}


def _pack(entries: List[Dict[str, Any]], count: int, limit: int,
          prompt: str, system_prompt: Optional[str], mode: str) -> List[List[Dict[str, Any]]]:
    batches, batch = [], []
    # Reserve room for the bounded validation feedback sent on the second attempt.
    overhead = len(prompt) + len(system_prompt or "") + 600
    for entry in entries:
        proposed = [*batch, entry]
        if batch and (len(proposed) > count or overhead + len(_json(_request(mode, proposed))) > limit):
            batches.append(batch)
            batch = []
        batch.append(entry)
        if overhead + len(_json(_request(mode, batch))) > limit:
            source = entry.get("item_id", entry.get("id"))
            raise ValueError(f"Discovery item {source!r} exceeds max_request_chars={limit}")
    if batch:
        batches.append(batch)
    return batches


def _provider_identity(provider: Any) -> Optional[Dict[str, Any]]:
    """Reuse cache fingerprints, requiring explicit identity for custom discovery."""
    if type(provider) not in (LocalProvider, OpenAIProvider) and getattr(provider, "cache_identity", None) is None:
        return None
    return _cache_provider_identity(provider)


def _step(provider: Any, entries: List[Dict[str, Any]], mode: str, prompt: str,
          system_prompt: Optional[str], path: Optional[Path],
          stats: Dict[str, int]) -> Dict[str, Any]:
    def validate(raw, current_entries):
        if mode == "map":
            return _validate_map(raw, {_provider_item_id(entry["item_id"]) for entry in current_entries})
        return _validate_reduce(raw, current_entries)

    request = _request(mode, entries)
    fingerprint = _digest({"request": request, "entries": entries, "prompt": prompt})
    if path is not None and path.exists():
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved["input_hash"] == fingerprint:
                result = validate(saved["response"], entries)
                diagnostics = saved["warnings"]
                if (result["errors"] or result["warnings"] or not isinstance(diagnostics, list)
                        or not all(isinstance(warning, str) for warning in diagnostics)):
                    raise ValueError("Invalid checkpoint result or diagnostics")
                result["warnings"] = diagnostics
                stats["cached_steps"] += 1
                return result
        except (OSError, ValueError, KeyError, TypeError):
            pass
    schema = MapDiscoveryResponse if mode == "map" else ReduceDiscoveryResponse
    kwargs = _provider_call_kwargs(provider, system_prompt, schema.model_json_schema())
    images = [image for entry in entries for image in entry.get("images", [])]
    key = "proposed_features" if mode == "map" else "merge_groups"
    accepted, warnings, pending = [], [], None
    current_entries = entries
    for attempt in range(2):
        if images:
            response = provider.image_features(
                images, prompt=prompt, as_set=True, extra_context=_json(request), **kwargs,
            )
        else:
            response = provider.text_features([_json(request)], prompt=prompt, **kwargs)
        stats["provider_calls"] += 1
        try:
            if not isinstance(response, list) or len(response) != 1:
                raise ValueError("Discovery provider must return exactly one response per batch")
            result = validate(response[0], current_entries)
        except ValueError as exc:
            if attempt:
                raise ValueError(f"Invalid discovery {mode} batch: {exc}") from exc
            errors = [str(exc)]
        else:
            warnings.extend(result["warnings"])
            if mode == "map":
                corrected = [feature for feature in result[key] if pending is None or feature["feature"] in pending]
                accepted.extend(corrected)
                if pending is None:
                    pending = result["retry_features"]
                else:
                    recovered = {feature["feature"] for feature in corrected}
                    pending = [name for name in pending if name not in recovered]
                errors = [f"Correct only feature {name!r}: cite at least one supplied evidence ID" for name in pending]
            else:
                accepted.extend(result[key])
                merged = {source for group in accepted for source in group["merge_ids"]}
                current_entries = [entry for entry in entries if entry["id"] not in merged]
                errors = result["errors"]
                warnings.extend(f"Rejected {error}" for error in errors)
            if not errors or attempt or (mode == "reduce" and len(current_entries) < 2):
                break
        request = _request(mode, current_entries)
        feedback = "; ".join(errors)[:500]
        while len(_json(feedback)) > 400:
            feedback = feedback[:-1]
        request["validation_error"] = feedback
    else:
        raise AssertionError("Unreachable")  # pragma: no cover
    if mode == "map":
        warnings.extend(f"Discarded unsupported map feature {name!r}: no known evidence IDs" for name in pending)
    warnings = list(dict.fromkeys(warnings))
    payload = {key: accepted}
    if mode == "map":
        payload["processed_item_ids"] = result["processed_item_ids"]
    if path is not None:
        _write_json(path, {"input_hash": fingerprint, "response": payload, "warnings": warnings})
    return {**payload, "warnings": warnings}


def _merge(catalog: Dict[str, Dict[str, Any]], keep: str, sources: List[str],
           reconciliation: Optional[Dict[str, Any]] = None) -> None:
    survivor = catalog[keep]
    if reconciliation is not None:
        translations = {(mapping["candidate_id"], mapping["source_value"]): mapping["canonical_value"]
                        for mapping in reconciliation["value_mappings"]}
        for source in sources:
            candidate = catalog[source]
            candidate["value_mappings"] = {
                original: {value: translations[(source, current)] for value, current in mapping.items()}
                for original, mapping in candidate["value_mappings"].items()
            }
        survivor["description"] = reconciliation["description"]
        survivor["possible_values"] = reconciliation["possible_values"]
    for source in sources:
        if source != keep:
            merged = catalog.pop(source)
            survivor["source_ids"].extend(merged["source_ids"])
            survivor["value_mappings"].update(merged["value_mappings"])
            survivor["example_item_ids"] = sorted(set(
                survivor["example_item_ids"] + merged["example_item_ids"]
            ))


def discover_map_reduce(
    items: List[Dict[str, Any]], *, provider: Any, prompt: str,
    system_prompt: Optional[str], output_path: Path, batch_size: int,
    reduce_batch_size: int, max_request_chars: int, checkpoint_dir: Optional[Path],
    reduce_prompt: Optional[str], min_features: int, modality: str,
    exclusions: List[Dict[str, str]], extraction_settings: Dict[str, Any],
    shuffle_seed: Optional[int] = 0,
) -> Dict[str, Any]:
    """Map all supplied items, preserve every candidate, and write a shared schema."""
    if not items:
        raise ValueError("No eligible discovery inputs found")
    items = list(items)
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(items)
    map_prompt = prompt + "\n\n" + map_discovery_contract
    reduction_prompt = (reduce_prompt if reduce_prompt is not None else prompt) + "\n\n" + reduce_discovery_contract
    config = {
        "version": 4, "strategy": "map_reduce", "modality": modality,
        "provider": _provider_identity(provider), "prompt": map_prompt,
        "system_prompt": system_prompt, "reduce_prompt": reduction_prompt,
        "batch_size": batch_size, "reduce_batch_size": reduce_batch_size,
        "shuffle_seed": shuffle_seed,
        "max_request_chars": max_request_chars, "min_features": min_features,
        "extraction": extraction_settings,
    }
    signature = _digest({"config": config, "items": items, "exclusions": exclusions})
    warnings: List[str] = []
    checkpoints = None
    if checkpoint_dir is not None:
        if config["provider"] is None:
            warnings.append("Checkpoint reuse disabled: custom provider needs cache_identity")
        else:
            checkpoints = Path(checkpoint_dir) / signature
            _write_json(checkpoints / "run.json", {"signature": signature, "config": config})
    stats = {"provider_calls": 0, "cached_steps": 0}
    batches = _pack(items, batch_size, max_request_chars, map_prompt, system_prompt, "map")
    catalog, all_sources, processed = {}, [], []
    for index, batch in enumerate(batches):
        path = checkpoints / "map" / f"{index:04d}.json" if checkpoints is not None else None
        source_ids = {_provider_item_id(item["item_id"]): item["item_id"] for item in batch}
        result = _step(provider, batch, "map", map_prompt, system_prompt, path, stats)
        warnings.extend(result["warnings"])
        processed.extend(source_ids[item_id] for item_id in result["processed_item_ids"])
        for position, feature in enumerate(result["proposed_features"]):
            feature["example_item_ids"] = [source_ids[item_id] for item_id in feature["example_item_ids"]]
            source_id = f"map_{index:04d}:feature_{position:04d}"
            catalog[source_id] = {**feature, "id": source_id, "source_ids": [source_id],
                                  "value_mappings": {source_id: {value: value for value in feature["possible_values"]}}}
            all_sources.append(source_id)
    if not catalog:
        raise ValueError("Discovery produced no evidenced features")
    exact = {}
    for candidate in list(catalog.values()):
        key = _json({key: candidate[key] for key in ("feature", "description")}) + _json(sorted(candidate["possible_values"]))
        if key in exact:
            _merge(catalog, exact[key], [candidate["id"]])
        else:
            exact[key] = candidate["id"]
    rounds = 0
    compared = set()
    # ponytail: one cross-name sweep, not all pairs; semantic search can expand recall if needed.
    for phase in ("families", "cross", "families"):
        while len(catalog) > 1:
            before = len(catalog)
            ordered = sorted(catalog.values(), key=lambda candidate: (candidate["feature"], candidate["id"]))
            families = ([ordered[::2] + ordered[1::2]] if phase == "cross" else
                        [list(family) for _, family in groupby(ordered, key=lambda candidate: candidate["feature"])])
            groups = [group for family in families
                      for group in _pack(family, reduce_batch_size, max_request_chars, reduction_prompt, system_prompt, "reduce")
                      if len(group) > 1 and _digest(group) not in compared]
            if not groups:
                break
            for index, group in enumerate(groups):
                compared.add(_digest(group))
                path = checkpoints / "reduce" / f"{rounds}_{index:04d}.json" if checkpoints is not None else None
                result = _step(provider, group, "reduce", reduction_prompt, system_prompt, path, stats)
                warnings.extend(result["warnings"])
                for merge in result["merge_groups"]:
                    _merge(catalog, merge["keep"], merge["merge_ids"], merge)
            rounds += 1
            if phase == "cross" or len(catalog) == before:
                break
    feature_specs, provenance, names = [], {}, {"File", "Class", "raw_llm_output"}
    for candidate in catalog.values():
        original = candidate["feature"]
        name, suffix = original, 2
        while name in names:
            name = f"{original}_{suffix}"
            suffix += 1
        if name != original:
            warnings.append(f"Renamed conflicting feature {original!r} to {name!r}")
        names.add(name)
        feature_specs.append({"feature": name, "description": candidate["description"],
                              "possible_values": candidate["possible_values"]})
        provenance[name] = {"original_name": original, "source_ids": candidate["source_ids"],
                            "value_mappings": candidate["value_mappings"],
                            "example_item_ids": candidate["example_item_ids"],
                            "support": len(candidate["example_item_ids"])}
    lineage = [source for candidate in catalog.values() for source in candidate["source_ids"]]
    if sorted(processed) != sorted(item["item_id"] for item in items) or sorted(lineage) != sorted(all_sources):
        raise ValueError("Incomplete discovery input or candidate coverage")
    schema = {"proposed_features": feature_specs}
    _validate_discovered_schema(schema)
    if len(feature_specs) < min_features:
        warnings.append(f"Requested {min_features} corpus features; discovered {len(feature_specs)}")
    report = {
        "signature": signature, "config": config, "eligible_item_count": len(items),
        "processed_item_ids": processed, "excluded_inputs": exclusions,
        "map_batches": len(batches), "candidate_count": len(all_sources),
        "feature_count": len(feature_specs), "reduce_rounds": rounds,
        "features": provenance, "warnings": warnings, **stats,
    }
    usage_summary = getattr(provider, "usage_summary", None)
    if callable(usage_summary):
        report["provider_usage"] = usage_summary()
    _write_json(output_path.with_name(f"{output_path.stem}_discovery_report.json"), report)
    _write_json(output_path, [schema])
    return schema
