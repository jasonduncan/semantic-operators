"""Flows: a decision built from operators, written as a plain Python function.

Higher layer: built only on the base layer and operators.

    @flow
    def triage(ask, message):
        a = ask(message, department, urgency)    # one provider call, two operators
        if a["department"] == "technical":       # stops here if department is undecided
            b = ask(message, outage)             # a second call, only when it's needed
            if b["outage"]:
                return "page on-call"
        return a["department"]

    result = triage(provider, message)
    result.value, result.decided, result.stopped_at, result.trace

A flow receives an ``ask`` handle. ``ask(state, *operators)`` asks all the operators in
one call and returns their values by name. Python's ``if`` is the flow language: there's
no graph or chain builder.

"Don't know" stops the flow when the flow *uses* it: reading an undecided answer raises
``Undecided``, which ends the flow with ``decided=False`` and ``stopped_at`` naming the
operator. An undecided answer the flow never reads doesn't stop anything. A flow can
also catch ``Undecided`` itself, for example to send the case to a person.

``result.trace`` records every call: the state and the full answers (probabilities,
confidence, which model answered). Provider errors propagate, as everywhere else.

Keep in mind that each ``ask`` is a provider call. Ask everything you might need in the
first call; add a later step only when it truly depends on an earlier answer.
"""

import inspect
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from .operators import Operator, apply, apply_async
from .provider import AsyncProvider, Provider
from .types import Answer, State


class Undecided(Exception):
    """A flow read an answer the model wasn't sure of."""

    def __init__(self, name: str, answer: Answer) -> None:
        super().__init__(f"{name!r} is undecided")
        self.name, self.answer = name, answer


class Answers(Mapping[str, Any]):
    """The values from one ``ask``, by operator name.

    ``answers[name]`` is the value, or raises ``Undecided``. ``answers.full[name]`` is the
    whole ``Answer`` (probabilities, confidence), which never raises.
    """

    def __init__(self, full: dict[str, Answer]) -> None:
        self.full = full

    def __getitem__(self, name: str) -> Any:
        answer = self.full[name]
        if not answer.decided:
            raise Undecided(name, answer)
        return answer.value

    def __iter__(self) -> Iterator[str]:
        return iter(self.full)

    def __len__(self) -> int:
        return len(self.full)


@dataclass(frozen=True)
class Step:
    state: State
    answers: dict[str, Answer]


@dataclass(frozen=True)
class FlowResult:
    value: Any                 # what the flow returned; None if it stopped
    stopped_at: str | None     # the undecided operator that stopped it, if any
    trace: list[Step] = field(default_factory=list)

    @property
    def decided(self) -> bool:
        return self.stopped_at is None

    @property
    def calls(self) -> int:
        return len(self.trace)


@dataclass(frozen=True)
class Flow:
    fn: Callable[..., Any]

    @property
    def name(self) -> str:
        return self.fn.__name__

    def __call__(self, provider: Provider, *args: Any, **kwargs: Any) -> FlowResult:
        if inspect.iscoroutinefunction(self.fn):
            raise TypeError(f"{self.name} is async: use `await {self.name}.call_async(...)`")
        trace: list[Step] = []

        def ask(state: State, *operators: Operator) -> Answers:
            answers = apply(provider, state, list(operators))
            trace.append(Step(state, answers))
            return Answers(answers)

        try:
            return FlowResult(self.fn(ask, *args, **kwargs), None, trace)
        except Undecided as stop:
            return FlowResult(None, stop.name, trace)

    async def call_async(self, provider: AsyncProvider, *args: Any, **kwargs: Any) -> FlowResult:
        if not inspect.iscoroutinefunction(self.fn):
            raise TypeError(f"{self.name} isn't async: call it as `{self.name}(provider, ...)`")
        trace: list[Step] = []

        async def ask(state: State, *operators: Operator) -> Answers:
            answers = await apply_async(provider, state, list(operators))
            trace.append(Step(state, answers))
            return Answers(answers)

        try:
            return FlowResult(await self.fn(ask, *args, **kwargs), None, trace)
        except Undecided as stop:
            return FlowResult(None, stop.name, trace)


def flow(fn: Callable[..., Any]) -> Flow:
    """Make ``fn(ask, ...)`` a flow: call it as ``fn(provider, ...)``, get a ``FlowResult``."""
    return Flow(fn)
