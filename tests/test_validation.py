"""Malformed questions are rejected when created; malformed provider output never
becomes a confident-looking answer. Offline."""

import asyncio
import threading
import time

import pytest

from semantic_operators import Boolean, Choice, ProviderError, Score, make_answer
from semantic_operators.providers.laya import AsyncLaya, Laya

yes_no = Boolean("Is it?", min_confidence=0.9)
pick = Choice("Which?", {"a": None, "b": None})
rate = Score("How much?", ["low", "medium", "high"])


# Questions

@pytest.mark.parametrize("make", [
    lambda: Boolean(""),
    lambda: Boolean("Is it?", min_confidence=1.5),
    lambda: Boolean("Is it?", min_confidence=float("nan")),
    lambda: Choice("Which?", {"a": None}),
    lambda: Choice("Which?", {"": None, "b": None}),
    lambda: Score("How much?", ["x", "x"]),
    lambda: Score("How much?", ["only one"]),
])
def test_malformed_questions_are_rejected(make):
    with pytest.raises(ValueError):
        make()


# Provider output

@pytest.mark.parametrize("question, value, probabilities", [
    (yes_no, True, {"true": float("nan"), "false": float("nan")}),  # not finite
    (yes_no, True, {"true": 1.2, "false": -0.2}),                   # out of range
    (yes_no, True, {"true": 0.2, "false": 0.8}),                    # value disagrees
    (yes_no, "yes", {"true": 0.9, "false": 0.1}),                   # not a bool
    (pick, "zzz", {"a": 0.5, "b": 0.5}),                            # not an option
    (pick, "b", {"a": 0.95, "b": 0.05}),                            # not the top option
    (pick, "a", {"a": 0.9}),                                        # missing an option
    (pick, "a", {"a": 0.9, "b": 0.9}),                              # doesn't sum to 1
    (rate, 5.0, {"low": 0.1, "medium": 0.2, "high": 0.7}),          # outside 0..2
    (rate, 0.3, {"low": 0.1, "medium": 0.2, "high": 0.7}),          # disagrees with expected 1.6
])
def test_malformed_output_raises(question, value, probabilities):
    with pytest.raises(ValueError):
        make_answer(question, value, probabilities)


def test_rounded_probabilities_are_accepted():
    # Laya rounds to 4 decimals, so totals can be slightly off 1.
    answer = make_answer(pick, "a", {"a": 0.6667, "b": 0.3334})
    assert answer.value == "a" and answer.confidence == 0.6667
    # A real TypeSafe answer: 2-decimal rounding puts the score 0.01 off its expectation.
    answer = make_answer(rate, 0.58, {"low": 0.43, "medium": 0.57, "high": 0.0})
    assert answer.value == 0.58 and answer.confidence == 0.57


def test_provider_turns_malformed_output_into_provider_error():
    class LyingModel:
        def predict(self, state, questions):
            return {"answers": {"q": {"choice": "zzz", "probabilities": {"a": 0.5, "b": 0.5}}}}

    with pytest.raises(ProviderError, match="not one of the options"):
        Laya(LyingModel()).ask("hi", {"q": pick})


# AsyncLaya: one prediction at a time, even after a cancellation

def test_cancelled_call_does_not_let_a_second_prediction_overlap():
    class SlowModel:
        def __init__(self):
            self.running = self.peak = 0
            self.counter = threading.Lock()

        def predict(self, state, questions):
            with self.counter:
                self.running += 1
                self.peak = max(self.peak, self.running)
            time.sleep(0.2)
            with self.counter:
                self.running -= 1
            return {"answers": {"q": {"noul": 0.9}}}

    model = SlowModel()
    provider = AsyncLaya(model)
    question = {"q": Boolean("Is it?")}

    async def scenario():
        first = asyncio.create_task(provider.ask("first", question))
        await asyncio.sleep(0.05)  # the first prediction is now running in its worker
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        await provider.ask("second", question)

    asyncio.run(scenario())
    assert model.peak == 1
