"""Timeouts, offline: with_timeout on async providers, and SDK timeouts from TypeSafe."""

import asyncio
import time

import httpx2
import pytest
import typesafe_sdk as ts

from semantic_operators import Boolean, ProviderError, ProviderTimeout, Score, make_answer, with_timeout
from semantic_operators.providers.typesafe import TypeSafe
from semantic_operators.rerank import rerank_async

question = {"q": Boolean("Is it?")}


class Sleepy:
    """Answers "yes" after ``delays[state]`` seconds."""

    def __init__(self, delays):
        self.delays = delays

    async def ask(self, state, questions):
        await asyncio.sleep(self.delays[state if isinstance(state, str) else state["document"]])
        return {name: _answer(q) for name, q in questions.items()}


def _answer(q):
    if isinstance(q, Score):
        return make_answer(q, 1.0, {level: [0, 1, 0][i] for i, level in enumerate(q.levels)})
    return make_answer(q, True, {"true": 0.9, "false": 0.1})


def test_provider_timeout_is_both_error_types():
    error = ProviderTimeout("X", "too slow")
    assert isinstance(error, ProviderError) and isinstance(error, TimeoutError)
    assert str(error) == "X: too slow" and error.provider == "X"


def test_fast_calls_pass_through():
    provider = with_timeout(Sleepy({"hi": 0}), 1.0)
    assert asyncio.run(provider.ask("hi", question))["q"].value is True


def test_slow_calls_raise_provider_timeout_promptly():
    provider = with_timeout(Sleepy({"hi": 5}), 0.05)
    start = time.perf_counter()
    with pytest.raises(ProviderTimeout, match="no answer within 0.05s"):
        asyncio.run(provider.ask("hi", question))
    assert time.perf_counter() - start < 1


def test_a_providers_own_timeout_is_not_rewrapped():
    class OwnTimeout:
        async def ask(self, state, questions):
            raise ProviderTimeout("Inner", "SDK gave up")

    with pytest.raises(ProviderTimeout, match="Inner: SDK gave up"):
        asyncio.run(with_timeout(OwnTimeout(), 5).ask("hi", question))


def test_outside_cancellation_still_cancels():
    async def scenario():
        task = asyncio.create_task(with_timeout(Sleepy({"hi": 5}), 10).ask("hi", question))
        await asyncio.sleep(0.01)
        task.cancel()
        await task

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(scenario())


@pytest.mark.parametrize("seconds", [0, -1, float("inf"), float("nan"), True, "5"])
def test_bad_limits_are_rejected(seconds):
    with pytest.raises(ValueError):
        with_timeout(Sleepy({}), seconds)


def test_rerank_leaves_timed_out_candidates_unscored():
    relevance = Score("How useful is the document?", ["none", "some", "full"])
    provider = with_timeout(Sleepy({"fast": 0, "slow": 5}), 0.05)
    ranking = asyncio.run(rerank_async(provider, relevance, query="q",
                                       candidates={"A": "fast", "B": "slow"}))
    assert [r.id for r in ranking.ranked] == ["A"]
    assert isinstance(ranking.unscored["B"], ProviderTimeout)


def test_typesafe_sdk_timeout_becomes_provider_timeout():
    def too_slow(request):
        raise httpx2.ReadTimeout("slow", request=request)

    client = ts.TypeSafeClient(api_key="test", transport=httpx2.MockTransport(too_slow),
                               retry=ts.RetryPolicy(max_retries=0))
    with pytest.raises(ProviderTimeout) as caught:
        TypeSafe(client).ask("hello", question)
    assert isinstance(caught.value.__cause__, ts.TypeSafeAPITimeoutError)
