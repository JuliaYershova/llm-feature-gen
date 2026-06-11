"""Shared response contracts and validation helpers."""

from __future__ import annotations

import json
from typing import Any, Dict, List, TypedDict


class ProviderResponseError(ValueError):
    """Raised when a provider response cannot be used as feature data."""


class FeatureValuesPayload(TypedDict):
    """Normalized feature-generation payload."""

    features: Dict[str, Any]


class DiscoveryPayload(TypedDict):
    """Normalized discovery payload."""

    proposed_features: List[Dict[str, Any]]


def parse_json_object_from_markdown(text: str) -> Dict[str, Any]:
    """Parse a JSON object that may be wrapped in a Markdown code fence."""
    if not text:
        raise ProviderResponseError("Empty JSON response")

    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()

    try:
        parsed = json.loads(candidate)
    except Exception as exc:
        raise ProviderResponseError("Invalid JSON response") from exc

    if not isinstance(parsed, dict):
        raise ProviderResponseError("JSON response must be an object")

    return parsed


def normalize_feature_values_response(raw: Any) -> FeatureValuesPayload:
    """Normalize provider output into one validated feature-value payload.

    Accepted legacy shapes:
    - {"features": {"feature_name": "value"}}
    - {"features": "{\"feature_name\": \"value\"}"}
    - {"feature_name": "value"}
    - [{"feature_name": "value"}]

    Error payloads, empty outputs, non-object feature data, and plain text
    responses are rejected with ``ProviderResponseError``.
    """
    if isinstance(raw, list):
        if len(raw) == 1 and isinstance(raw[0], dict):
            raw = raw[0]
        else:
            raise ProviderResponseError("Feature response list must contain one object")

    if not isinstance(raw, dict) or not raw:
        raise ProviderResponseError("Feature response must be a non-empty object")

    if raw.get("error"):
        raise ProviderResponseError(str(raw["error"]))

    features = raw.get("features", raw)

    if isinstance(features, str):
        features = parse_json_object_from_markdown(features)

    if not isinstance(features, dict) or not features:
        raise ProviderResponseError("Feature response must contain a non-empty feature object")

    return {"features": features}
