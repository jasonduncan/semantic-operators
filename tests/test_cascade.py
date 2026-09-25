"""Escalation, offline. Scripted providers answer each question with a fixed confidence,
so it's exact which answers escalate and which provider gave each one."""

import asyncio

import pytest

from semantic_operators import Boolean, Call, Choice, ProviderError, make_answer
from semantic_operators.cascade import cascade, cascade_async
from semantic_operators.operators import Operator, apply

sure = Boolean("Is it?", min_confidence=0.8)
pick = Choice("Which?", {"a": None, "b": None}, min_confidence=0.8)


class Scripted:
    """Answers each question with ``confidence[name]`` in the true/"a" direction."""

    def __init__(self, name, confidence):
        self.name, self.confidence, self.asked = name, confidence, []

    def ask(self, state, questions):
        self.asked.append(sorted(questions))
        answers = {}
        for name, q in questions.items():
            p = self.confidence[name]
            probs = {"true": p, "false": 1 - p} if isinstance(q, Boolean) else {"a": p, "b": 1 - p}
            value = True if isinstance(q, Boolean) else "a"
            answers[name] = make_answer(q, value, probs, call=Call(self.name))
        return answers


class AsyncScripted(Scripted):
    async def ask(self, state, questions):
        return Scripted.ask(self, state, questions)


def test_only_undecided_answers_escalate_in_one_call():
    fast = Scripted("fast", {"x": 0.95, "y": 0.6, "z": 0.7})
    strong = Scripted("strong", {"y": 0.9, "z": 0.99})
    answers = cascade(fast, strong).ask("hi", {"x": sure, "y": pick, "z": sure})
    assert fast.asked == [["x", "y", "z"]]
    assert strong.asked == [["y", "z"]]                     # one call, only the unsure ones
    assert {n: a.call.provider for n, a in answers.items()} == \
        {"x": "fast", "y": "strong", "z": "strong"}
    assert list(answers) == ["x", "y", "z"] and all(a.decided for a in answers.values())


def test_nothing_escalates_when_the_first_is_sure():
    fast, strong = Scripted("fast", {"x": 0.9}), Scripted("strong", {})
    cascade(fast, strong).ask("hi", {"x": sure})
    assert strong.asked == []


def test_the_last_providers_answer_stands_even_if_undecided():
    a = Scripted("a", {"x": 0.6})
    b = Scripted("b", {"x": 0.7})
    c = Scripted("c", {"x": 0.55})
    answer = cascade(a, b, c).ask("hi", {"x": sure})["x"]
    assert (answer.decided, answer.call.provider) == (False, "c")
    assert [p.asked for p in (a, b, c)] == [[["x"]]] * 3


def test_errors_propagate_instead_of_escalating():
    class Down:
        def ask(self, state, questions):
            raise ProviderError("Down", "outage")

    strong = Scripted("strong", {"x": 0.99})
    with pytest.raises(ProviderError, match="outage"):
        cascade(Down(), strong).ask("hi", {"x": sure})
    assert strong.asked == []


def test_works_with_operators_and_async():
    op = Operator("x", sure)
    provider = cascade(Scripted("fast", {"x": 0.6}), Scripted("strong", {"x": 0.9}))
    assert op(provider, "hi").call.provider == "strong"
    assert apply(provider, "hi", [op])["x"].decided

    async_provider = cascade_async(AsyncScripted("fast", {"x": 0.6}),
                                   AsyncScripted("strong", {"x": 0.9}))
    assert asyncio.run(async_provider.ask("hi", {"x": sure}))["x"].call.provider == "strong"


def test_escalate_below_is_separate_from_the_final_dont_know():
    always = Boolean("Is it?")  # min_confidence 0: the final answer always counts
    fast = Scripted("fast", {"x": 0.7, "y": 0.95})
    strong = Scripted("strong", {"x": 0.6})
    answers = cascade(fast, strong, escalate_below=0.8).ask("hi", {"x": always, "y": always})
    assert strong.asked == [["x"]]
    # Escalated, and strong was less sure than 0.8 too, but its answer still counts.
    assert answers["x"].decided and answers["x"].call.provider == "strong"
    assert answers["y"].call.provider == "fast"


@pytest.mark.parametrize("kwargs", [{}, {"escalate_below": 1.5}, {"escalate_below": float("nan")}])
def test_bad_setups_are_rejected(kwargs):
    providers = (Scripted("a", {}),) if not kwargs else (Scripted("a", {}), Scripted("b", {}))
    with pytest.raises(ValueError):
        cascade(*providers, **kwargs)
