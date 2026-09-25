"""Filtering: ask one yes/no question of every item, and keep the ones the model says yes to.

Higher layer: built only on the base layer.

    question = Boolean("Does this log record a release that failed?", min_confidence=0.8)
    result = filter_items(provider, question, {"log-1": text1, "log-2": text2})
    [v.id for v in result.matched]

Each item gets its own call, so an item's answer can't depend on the others (the same
rule ``rerank`` follows). Item ids are never sent. Every item ends in exactly one
outcome, in input order:

- ``match``: decided yes
- ``no``: decided no
- ``unsure``: undecided (below the question's ``min_confidence``, or a tie); never a match
- ``failed``: the provider raised ``ProviderError`` (including a timeout)

``context`` (optional) is shared background sent with every item: the state becomes
``{"context": context, "item": <the item>}``.

This module never touches files. Reading them, cutting long text, and output formatting
belong to the caller (``semop filter`` does all three).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from ._each import Result, ask_each, ask_each_async, check_concurrency
from .errors import ProviderError
from .provider import AsyncProvider, Provider
from .types import Answer, Boolean, State

Outcome = Literal["match", "no", "unsure", "failed"]


@dataclass(frozen=True)
class Verdict:
    id: str
    outcome: Outcome
    answer: Answer | None = None          # None when failed
    error: ProviderError | None = None    # set when failed


@dataclass(frozen=True)
class Filtered:
    question: Boolean
    verdicts: list[Verdict]   # one per item, in input order

    def _with(self, outcome: Outcome) -> list[Verdict]:
        return [v for v in self.verdicts if v.outcome == outcome]

    @property
    def matched(self) -> list[Verdict]:
        return self._with("match")

    @property
    def rejected(self) -> list[Verdict]:
        return self._with("no")

    @property
    def unsure(self) -> list[Verdict]:
        return self._with("unsure")

    @property
    def failed(self) -> list[Verdict]:
        return self._with("failed")

    @property
    def models(self) -> set[str]:
        """Every model the provider reported answering. More than one: answers may not compare."""
        return {v.answer.call.model for v in self.verdicts
                if v.answer and v.answer.call and v.answer.call.model}


def filter_items(provider: Provider, question: Boolean, items: Mapping[str, State], *,
                 context: Any = None) -> Filtered:
    """Ask ``question`` of every item, one call each."""
    states = _states(question, items, context)
    return _filtered(question, list(states), ask_each(provider, question, list(states.values())))


async def filter_items_async(provider: AsyncProvider, question: Boolean,
                             items: Mapping[str, State], *, context: Any = None,
                             concurrency: int = 8) -> Filtered:
    """``filter_items`` with up to ``concurrency`` calls in flight at once."""
    check_concurrency(concurrency)
    states = _states(question, items, context)
    results = await ask_each_async(provider, question, list(states.values()), concurrency)
    return _filtered(question, list(states), results)


def _states(question: Boolean, items: Mapping[str, State], context: Any) -> dict[str, Any]:
    if not isinstance(question, Boolean):
        raise TypeError("filtering needs a Boolean question")
    if not all(isinstance(id, str) and id for id in items):
        raise ValueError("item ids must be non-empty strings")
    if context is None:
        return dict(items)
    return {id: {"context": context, "item": item} for id, item in items.items()}


def _filtered(question: Boolean, ids: list[str], results: list[Result]) -> Filtered:
    verdicts = []
    for id, result in zip(ids, results, strict=True):
        if isinstance(result, ProviderError):
            verdicts.append(Verdict(id, "failed", error=result))
        elif not result.decided:
            verdicts.append(Verdict(id, "unsure", result))
        else:
            verdicts.append(Verdict(id, "match" if result.value else "no", result))
    return Filtered(question, verdicts)
