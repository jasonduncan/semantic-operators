"""Flows, offline. A scripted provider answers each operator by name with a fixed value
and confidence, and records every call."""

import asyncio

import pytest

from semantic_operators import Boolean, Choice, ProviderError, Score, make_answer
from semantic_operators.bench import run_flow
from semantic_operators.flows import Undecided, flow
from semantic_operators.operators import Operator

department = Operator("department", Choice("Which team?", {"billing": None, "technical": None},
                                           min_confidence=0.8))
urgency = Operator("urgency", Score("How urgent?", ["low", "medium", "high"]))
outage = Operator("outage", Boolean("Is something down?"))


class Scripted:
    """``script[state][name]`` is (value, confidence) for that operator."""

    def __init__(self, script):
        self.script, self.calls = script, []

    def ask(self, state, questions):
        self.calls.append((state, sorted(questions)))
        return {name: self._answer(q, *self.script[state][name]) for name, q in questions.items()}

    @staticmethod
    def _answer(q, value, p):
        if isinstance(q, Boolean):
            return make_answer(q, value, {"true": p if value else 1 - p, "false": 1 - p if value else p})
        if isinstance(q, Choice):
            return make_answer(q, value, {o: p if o == value else 1 - p for o in q.options})
        level = round(value)
        probs = {lvl: (1.0 if i == level else 0.0) for i, lvl in enumerate(q.levels)}
        return make_answer(q, float(level), probs)


class AsyncScripted(Scripted):
    async def ask(self, state, questions):
        return Scripted.ask(self, state, questions)


@flow
def triage(ask, message):
    a = ask(message, department, urgency)
    if a["department"] == "technical":
        b = ask(message, outage)
        if b["outage"]:
            return "page on-call"
    return f"{a['department']}/{'urgent' if a['urgency'] >= 1.5 else 'normal'}"


SCRIPT = {
    "charged twice": {"department": ("billing", 0.95), "urgency": (1, 1)},
    "site down": {"department": ("technical", 0.9), "urgency": (2, 1), "outage": (True, 0.97)},
    "export broken": {"department": ("technical", 0.9), "urgency": (1, 1), "outage": (False, 0.9)},
    "hmm": {"department": ("billing", 0.6), "urgency": (0, 1)},
}


def test_a_flow_returns_its_value_with_a_trace():
    provider = Scripted(SCRIPT)
    result = triage(provider, "charged twice")
    assert (result.value, result.decided, result.stopped_at, result.calls) == \
        ("billing/normal", True, None, 1)
    step = result.trace[0]
    assert step.state == "charged twice" and step.answers["department"].confidence == 0.95


def test_later_steps_only_run_when_needed():
    provider = Scripted(SCRIPT)
    assert triage(provider, "site down").value == "page on-call"
    assert triage(provider, "export broken").value == "technical/normal"
    assert provider.calls == [("site down", ["department", "urgency"]), ("site down", ["outage"]),
                              ("export broken", ["department", "urgency"]),
                              ("export broken", ["outage"])]


def test_reading_an_undecided_answer_stops_the_flow():
    result = triage(Scripted(SCRIPT), "hmm")
    assert (result.value, result.decided, result.stopped_at) == (None, False, "department")
    assert result.calls == 1 and not result.trace[0].answers["department"].decided


def test_an_undecided_answer_the_flow_never_reads_doesnt_stop_it():
    @flow
    def just_urgency(ask, message):
        return ask(message, department, urgency)["urgency"]

    assert just_urgency(Scripted(SCRIPT), "hmm").value == 0.0


def test_a_flow_can_handle_undecided_itself():
    @flow
    def routed(ask, message):
        a = ask(message, department)
        try:
            return a["department"]
        except Undecided as unsure:
            return f"human (leaning {max(unsure.answer.probabilities, key=unsure.answer.probabilities.get)})"

    result = routed(Scripted(SCRIPT), "hmm")
    assert result.decided and result.value == "human (leaning billing)"


def test_full_answers_are_available_without_raising():
    @flow
    def confidence_of(ask, message):
        return ask(message, department).full["department"].confidence

    assert confidence_of(Scripted(SCRIPT), "hmm").value == 0.6


def test_provider_errors_propagate():
    class Down:
        def ask(self, state, questions):
            raise ProviderError("Down", "outage")

    with pytest.raises(ProviderError):
        triage(Down(), "charged twice")


def test_async_flows_and_mismatches():
    @flow
    async def async_triage(ask, message):
        a = await ask(message, department, urgency)
        return a["department"]

    result = asyncio.run(async_triage.call_async(AsyncScripted(SCRIPT), "charged twice"))
    assert result.value == "billing" and result.calls == 1
    with pytest.raises(TypeError, match="is async"):
        async_triage(Scripted(SCRIPT), "charged twice")
    with pytest.raises(TypeError, match="isn't async"):
        asyncio.run(triage.call_async(AsyncScripted(SCRIPT), "charged twice"))


def test_run_flow_counts_right_wrong_and_stopped():
    cases = [("charged twice", "billing/normal"), ("site down", "page on-call"),
             ("export broken", "technical/urgent"), ("hmm", "billing/normal")]
    report = run_flow(triage, Scripted(SCRIPT), cases)
    assert (report.total, report.correct, report.stopped) == (4, 2, 1)
    assert report.misses == [(2, "technical/urgent", "technical/normal")]
    assert report.accuracy_when_answered == pytest.approx(2 / 3)
    assert report.calls == 6  # two cases needed the outage step
