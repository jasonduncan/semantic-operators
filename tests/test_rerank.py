"""Reranking, offline. A scripted provider returns a fixed relevance distribution per
document, so ordering, gaps, and concurrency can be checked exactly."""

import asyncio
import math

import pytest

from semantic_operators import Call, ProviderError, Score, make_answer
from semantic_operators.bench import ndcg
from semantic_operators.rerank import rerank, rerank_async, reweighted

relevance = Score("How useful is the document for the query?", ["none", "some", "full"])


def dist(low, mid, high):
    return {"none": low, "some": mid, "full": high}


class Scripted:
    """Answers from ``script[document]``: a distribution, "error", or "undecided"."""

    def __init__(self, script, models=None):
        self.script, self.models, self.states = script, models or {}, []

    def ask(self, state, questions):
        self.states.append(state)
        doc = state["document"]
        entry = self.script[doc]
        if entry == "error":
            raise ProviderError("Scripted", f"failed on {doc}")
        (name, question), = questions.items()
        q = Score(question.instructions, question.levels, min_confidence=0.9) \
            if entry == "undecided" else question
        p = dist(0.2, 0.5, 0.3) if entry == "undecided" else entry
        value = sum(i * p[level] for i, level in enumerate(question.levels))
        call = Call("Scripted", self.models.get(doc, "model-a"))
        return {name: make_answer(q, value, p, call=call)}


def ids(items):
    return [r.id for r in items]


def test_sorts_by_expected_level_and_ties_keep_retrieval_order():
    provider = Scripted({"a": dist(0.6, 0.4, 0), "b": dist(0, 0.2, 0.8),
                         "c": dist(0.6, 0.4, 0), "d": dist(0, 0.5, 0.5)})
    ranking = rerank(provider, relevance, query="q",
                     candidates={"A": "a", "B": "b", "C": "c", "D": "d"})
    assert ids(ranking.ranked) == ["B", "D", "A", "C"]  # A and C tie: retrieval order wins
    assert ranking.complete and ids(ranking.top(2)) == ["B", "D"]
    assert ranking.ranked[0].score == pytest.approx(1.8)


def test_each_call_sees_only_query_document_and_context():
    provider = Scripted({"a": dist(1, 0, 0)})
    rerank(provider, relevance, query="q", candidates={"doc-1": "a"})
    rerank(provider, relevance, query="q", candidates={"doc-1": "a"}, context={"user": "admin"})
    assert provider.states == [{"query": "q", "document": "a"},
                               {"query": "q", "document": "a", "context": {"user": "admin"}}]


def test_unscored_candidates_are_kept_and_partial_needs_consent():
    provider = Scripted({"a": dist(0, 0, 1), "b": "error", "c": "undecided"})
    ranking = rerank(provider, relevance, query="q", candidates={"A": "a", "B": "b", "C": "c"})
    assert ids(ranking.ranked) == ["A"] and not ranking.complete
    assert isinstance(ranking.unscored["B"], ProviderError)
    assert not ranking.unscored["C"].decided  # probabilities kept for inspection
    with pytest.raises(ValueError, match="unscored"):
        ranking.top(10)
    assert ids(ranking.top(10, allow_partial=True)) == ["A"]


def test_scores_from_different_models_are_refused():
    provider = Scripted({"a": dist(0, 0, 1), "b": dist(1, 0, 0)}, models={"b": "model-b"})
    ranking = rerank(provider, relevance, query="q", candidates={"A": "a", "B": "b"})
    assert ranking.models == {"model-a", "model-b"}
    with pytest.raises(ValueError, match="different models"):
        ranking.top(10)


def test_reweighting_can_change_the_order_without_calling_again():
    # Same expected level (1.0), different shapes: B is a sure "some", A is split.
    provider = Scripted({"a": dist(0.5, 0, 0.5), "b": dist(0, 1, 0)})
    ranking = rerank(provider, relevance, query="q", candidates={"A": "a", "B": "b"})
    calls = len(provider.states)
    rewarding_full = reweighted(ranking, [0, 10, 100])
    assert ids(rewarding_full.ranked) == ["A", "B"] and rewarding_full.ranked[0].score == 50
    assert len(provider.states) == calls              # no new provider calls
    assert ranking.ranked[0].answer.value == 1.0      # the original answers are untouched
    for bad in ([0, 1], [0, 0, 1], [0, 1, float("inf")], [True, 2, 3]):
        with pytest.raises(ValueError):
            reweighted(ranking, bad)


def test_input_checks():
    provider = Scripted({})
    with pytest.raises(TypeError):
        rerank(provider, object(), query="q", candidates={})
    with pytest.raises(ValueError):
        rerank(provider, relevance, query="  ", candidates={})
    empty = rerank(provider, relevance, query="q", candidates={})
    assert empty.complete and empty.ranked == [] and provider.states == []


def test_async_respects_the_concurrency_limit_and_keeps_ids_matched():
    class Slow:
        def __init__(self):
            self.running = self.peak = 0
            self.inner = Scripted({str(i): dist(0, 1 - i / 10, i / 10) for i in range(10)})

        async def ask(self, state, questions):
            self.running += 1
            self.peak = max(self.peak, self.running)
            await asyncio.sleep(0.01 * (10 - int(state["document"])))  # later docs finish first
            self.running -= 1
            return self.inner.ask(state, questions)

    provider = Slow()
    candidates = {f"doc-{i}": str(i) for i in range(10)}
    ranking = asyncio.run(rerank_async(provider, relevance, query="q",
                                       candidates=candidates, concurrency=3))
    assert provider.peak == 3
    assert ids(ranking.ranked) == [f"doc-{i}" for i in reversed(range(10))]


def test_async_unexpected_errors_propagate():
    class Broken:
        async def ask(self, state, questions):
            raise RuntimeError("bug in a provider")

    with pytest.raises(ExceptionGroup):
        asyncio.run(rerank_async(Broken(), relevance, query="q", candidates={"A": "a"}))


def test_ndcg():
    grades = {"A": 2, "B": 0, "C": 1}
    assert ndcg(["A", "C", "B"], grades) == 1.0
    # gains: A = 2**2-1 = 3, C = 1; discounts 1/log2(position + 2)
    ideal = 3 / math.log2(2) + 1 / math.log2(3)
    assert ndcg(["B", "C", "A"], grades) == pytest.approx((1 / math.log2(3) + 3 / math.log2(4)) / ideal)
    assert ndcg(["A", "B"], {"A": 0, "B": 0}) is None  # nothing relevant: undefined
