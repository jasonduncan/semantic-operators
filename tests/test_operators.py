"""Operators, offline. A recording provider stands in for a model: it answers every
question with fixed probabilities and remembers each call it received."""

import asyncio

import pytest

from semantic_operators import Boolean, Choice, make_answer
from semantic_operators.operators import Operator, apply, apply_async, questions

is_complaint = Operator("is_complaint", Boolean("Is the customer complaining?"))
department = Operator("department", Choice("Which team?", {"billing": None, "other": None}))


class Recorder:
    def __init__(self):
        self.calls = []

    def ask(self, state, qs):
        self.calls.append((state, dict(qs)))
        return {name: self._answer(q) for name, q in qs.items()}

    @staticmethod
    def _answer(q):
        if isinstance(q, Boolean):
            return make_answer(q, True, {"true": 0.9, "false": 0.1})
        return make_answer(q, "billing", {"billing": 0.8, "other": 0.2})


class AsyncRecorder(Recorder):
    async def ask(self, state, qs):
        return Recorder.ask(self, state, qs)


def test_operator_call_asks_one_question():
    provider = Recorder()
    assert is_complaint(provider, "I was charged twice").value is True
    assert provider.calls == [("I was charged twice", {"is_complaint": is_complaint.question})]


def test_apply_asks_all_operators_in_one_call():
    provider = Recorder()
    answers = apply(provider, "I was charged twice", [is_complaint, department])
    assert len(provider.calls) == 1
    assert (answers["is_complaint"].value, answers["department"].value) == (True, "billing")


def test_async_versions():
    provider = AsyncRecorder()
    assert asyncio.run(department.call_async(provider, "hi")).value == "billing"
    answers = asyncio.run(apply_async(provider, "hi", [is_complaint, department]))
    assert set(answers) == {"is_complaint", "department"} and len(provider.calls) == 2


def test_duplicate_names_are_rejected():
    clash = Operator("department", Boolean("Is it about a department?"))
    with pytest.raises(ValueError, match="department"):
        questions([department, clash])
