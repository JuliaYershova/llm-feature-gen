"""Token usage tracking shared by the providers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class Usage:
    """Cumulative token counts for a single provider instance."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> Dict[str, int]:
        data = asdict(self)
        data["total_tokens"] = self.total_tokens
        return data


class BaseProvider:
    """Accumulate token usage reported by provider responses.

    Providers record every successful response with :meth:`_record_usage`.
    Callers read the totals with :meth:`usage_summary` and clear them with
    :meth:`reset_usage`, for example between discovery and generation.

    Responses without a ``usage`` payload still count as a call, so the call
    count stays accurate for backends that do not report tokens.
    """

    @property
    def _usage(self) -> Usage:
        """The counter, created on first use.

        Lazily initialised so that providers built through alternative paths
        (subclasses, ``object.__new__``) can never end up without one.
        """
        usage = getattr(self, "_usage_state", None)
        if usage is None:
            usage = Usage()
            self._usage_state = usage
        return usage

    @_usage.setter
    def _usage(self, usage: Usage) -> None:
        self._usage_state = usage

    def _record_usage(self, response: Any) -> None:
        self._usage.calls += 1
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        self._usage.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
        self._usage.completion_tokens += getattr(usage, "completion_tokens", 0) or 0

    def usage_summary(self) -> Dict[str, int]:
        """Token counts accumulated since the provider was created or reset."""
        return self._usage.as_dict()

    def reset_usage(self) -> None:
        """Clear the accumulated counts."""
        self._usage_state = Usage()