"""Escalation: ask a fast model first, and a stronger one only about what it wasn't sure of.

Higher layer: built only on the base layer (types + Provider).

    provider = cascade(Laya(model), TypeSafe(client), escalate_below=0.8)  # still a provider
    answers = provider.ask(message, questions)
    answers["department"].call.provider      # "Laya", or "TypeSafe" if escalated

The first provider is asked every question. Any answer it isn't sure of is re-asked of
the next provider, all of them in one call, and so on down the list. "Not sure" means
undecided (a tie, or below the question's ``min_confidence``) or, with
``escalate_below``, a confidence under that number. The last provider's answer stands.
Each answer's ``call`` shows which provider gave it.

Two thresholds, two decisions: ``escalate_below`` decides when to ask a stronger model;
a question's ``min_confidence`` decides when the final answer is a "don't know".

It escalates when a model is *unsure*, never when it *fails*: a ``ProviderError`` from
any provider propagates, because quietly routing around an outage would hide it.

It's only as good as the first model's confidence. If that model is confidently wrong,
nothing escalates. Check with a benchmark before relying on it (``benchmarks/run_cascade.py``).
"""

from collections.abc import Mapping

from .provider import AsyncProvider, Provider
from .types import Answer, Question, State


def cascade(*providers: Provider, escalate_below: float = 0.0) -> Provider:
    """Providers in order, fastest or cheapest first. Needs at least two."""
    _check(providers, escalate_below)
    return _Cascade(providers, escalate_below)


def cascade_async(*providers: AsyncProvider, escalate_below: float = 0.0) -> AsyncProvider:
    """``cascade`` for async providers."""
    _check(providers, escalate_below)
    return _AsyncCascade(providers, escalate_below)


class _Cascade:
    def __init__(self, providers: tuple[Provider, ...], escalate_below: float) -> None:
        self.providers, self.escalate_below = providers, escalate_below

    def ask(self, state: State, questions: Mapping[str, Question]) -> dict[str, Answer]:
        answers: dict[str, Answer] = {}
        pending = dict(questions)
        for provider in self.providers:
            answers |= provider.ask(state, pending)
            pending = _unsure(pending, answers, self.escalate_below)
            if not pending:
                break
        return {name: answers[name] for name in questions}


class _AsyncCascade:
    def __init__(self, providers: tuple[AsyncProvider, ...], escalate_below: float) -> None:
        self.providers, self.escalate_below = providers, escalate_below

    async def ask(self, state: State, questions: Mapping[str, Question]) -> dict[str, Answer]:
        answers: dict[str, Answer] = {}
        pending = dict(questions)
        for provider in self.providers:
            answers |= await provider.ask(state, pending)
            pending = _unsure(pending, answers, self.escalate_below)
            if not pending:
                break
        return {name: answers[name] for name in questions}


def _unsure(pending: dict[str, Question], answers: dict[str, Answer],
           escalate_below: float) -> dict[str, Question]:
    return {name: q for name, q in pending.items()
            if not answers[name].decided or answers[name].confidence < escalate_below}


def _check(providers: tuple[object, ...], escalate_below: float) -> None:
    if len(providers) < 2:
        raise ValueError("cascade needs at least two providers")
    if not isinstance(escalate_below, (int, float)) or isinstance(escalate_below, bool) \
            or not 0 <= escalate_below <= 1:
        raise ValueError(f"escalate_below must be a number from 0 to 1, got {escalate_below!r}")
