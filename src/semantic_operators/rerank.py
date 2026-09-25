"""Reranking: score each candidate's relevance to a query, then sort.

Higher layer: built only on the base layer (types + Provider).

Retrieval stays in your code. You pass the candidates it found, in retrieval order,
and a relevance rubric (a ``Score``). Each candidate is scored on its own, one
provider call per candidate, and ordinary code sorts the results:

    relevance = Score("How useful is the document for answering the query?",
                      ["no useful information", "on topic but doesn't answer",
                       "partly answers", "answers with minor gaps", "fully answers"])

    ranking = rerank(provider, relevance, query="How do I reset my password?",
                     candidates={"doc-1": {...}, "doc-2": {...}})
    ranking.top(10)

What each call sees is ``{"query": ..., "document": ..., "context": ...}`` (context only
when you pass one). Candidate ids and positions are never sent, so a document's score
doesn't depend on which other documents came with it.

The ranking key is the expected rubric level (``answer.value``), not confidence.
Ties keep retrieval order. Scores aren't normalized across documents: all of them can
be relevant, or none. ``reweighted`` re-sorts by your own level weights, without
calling the provider again.

A candidate that comes back undecided or fails is kept in ``unscored``, never given a
made-up score. ``top`` refuses a partial ranking unless you ask for one, and refuses a
ranking whose answers came from different models (a moving alias such as
``jev-latest`` can change mid-run), since those scores aren't on one scale.
"""

import asyncio
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .errors import ProviderError
from .provider import AsyncProvider, Provider
from .types import Answer, Score

_NAME = "relevance"


@dataclass(frozen=True)
class Ranked:
    id: str
    answer: Answer
    score: float    # the ranking key: the expected level, or your weighted value
    position: int   # where retrieval put it (0 = first); breaks ties


@dataclass(frozen=True)
class Ranking:
    question: Score
    ranked: list[Ranked]                          # best first
    unscored: dict[str, Answer | ProviderError]   # undecided answers or errors, by id

    @property
    def complete(self) -> bool:
        return not self.unscored

    @property
    def models(self) -> set[str]:
        """Every model the provider reported answering. More than one: scores may not compare."""
        answers = [r.answer for r in self.ranked]
        answers += [u for u in self.unscored.values() if isinstance(u, Answer)]
        return {a.call.model for a in answers if a.call and a.call.model}

    def top(self, k: int, *, allow_partial: bool = False) -> list[Ranked]:
        """The ``k`` best candidates. Raises if any candidate is unscored (unless
        ``allow_partial``) or if answers came from more than one model."""
        if not self.complete and not allow_partial:
            raise ValueError(f"{len(self.unscored)} candidate(s) unscored: "
                             f"{', '.join(self.unscored)}; pass allow_partial=True to accept")
        if len(self.models) > 1:
            raise ValueError(f"scores came from different models: {', '.join(sorted(self.models))}")
        return self.ranked[:k]


def rerank(provider: Provider, question: Score, *, query: str, candidates: Mapping[str, Any],
           context: Any = None) -> Ranking:
    """Score every candidate against ``query``, one call each, and rank them."""
    states = _states(question, query, candidates, context)
    results: list[Answer | ProviderError] = []
    for state in states.values():
        try:
            results.append(provider.ask(state, {_NAME: question})[_NAME])
        except ProviderError as error:
            results.append(error)
    return _rank(question, list(states), results)


async def rerank_async(provider: AsyncProvider, question: Score, *, query: str,
                       candidates: Mapping[str, Any], context: Any = None,
                       concurrency: int = 8) -> Ranking:
    """``rerank`` with up to ``concurrency`` calls in flight at once.

    A ``ProviderError`` only affects its own candidate. Any other exception cancels
    the remaining calls and propagates.
    """
    if not isinstance(concurrency, int) or isinstance(concurrency, bool) or concurrency < 1:
        raise ValueError("concurrency must be a positive integer")
    states = _states(question, query, candidates, context)
    slots = asyncio.Semaphore(concurrency)

    async def one(state: dict[str, Any]) -> Answer | ProviderError:
        async with slots:
            try:
                return (await provider.ask(state, {_NAME: question}))[_NAME]
            except ProviderError as error:
                return error

    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(one(state)) for state in states.values()]
    return _rank(question, list(states), [task.result() for task in tasks])


def reweighted(ranking: Ranking, weights: Sequence[float]) -> Ranking:
    """Re-rank by ``sum(weight[level] * probability[level])``, from the probabilities
    the provider already returned. No provider call.

    One weight per level, strictly increasing. Unlike the default expected level,
    uneven weights can change the order, so evaluate them before relying on them.
    """
    levels = ranking.question.levels
    if (len(weights) != len(levels)
            or not all(isinstance(w, (int, float)) and not isinstance(w, bool) and math.isfinite(w)
                       for w in weights)
            or any(a >= b for a, b in zip(weights, weights[1:]))):
        raise ValueError(f"need {len(levels)} finite, strictly increasing weights, one per level")
    rescored = [Ranked(r.id, r.answer, _weighted(r.answer, levels, weights), r.position)
                for r in ranking.ranked]
    return Ranking(ranking.question, _sorted(rescored), dict(ranking.unscored))


def _states(question: Score, query: str, candidates: Mapping[str, Any],
            context: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(question, Score):
        raise TypeError("rerank needs a Score question (a relevance rubric)")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not all(isinstance(id, str) and id for id in candidates):
        raise ValueError("candidate ids must be non-empty strings")
    extra = {} if context is None else {"context": context}
    return {id: {"query": query, "document": document, **extra}
            for id, document in candidates.items()}


def _rank(question: Score, ids: list[str], results: list[Answer | ProviderError]) -> Ranking:
    ranked, unscored = [], {}
    for position, (id, result) in enumerate(zip(ids, results, strict=True)):
        if isinstance(result, Answer) and result.decided:
            ranked.append(Ranked(id, result, float(result.value), position))
        else:
            unscored[id] = result
    return Ranking(question, _sorted(ranked), unscored)


def _sorted(items: list[Ranked]) -> list[Ranked]:
    # Highest score first; exact ties keep retrieval order. No rounding, no "close enough".
    return sorted(items, key=lambda r: (-r.score, r.position))


def _weighted(answer: Answer, levels: list[str], weights: Sequence[float]) -> float:
    total = math.fsum(answer.probabilities[level] for level in levels)
    return math.fsum(w * answer.probabilities[level] for w, level in zip(weights, levels)) / total
