"""Named operators: define a semantic judgment once, then use it anywhere.

Higher layer: built only on the base layer (types + Provider).

An operator is a name plus a question. The name is what identifies its answer,
so several operators can be asked together in one provider call (System One
models answer many questions in a single pass):

    is_complaint = Operator("is_complaint", Boolean("Is the customer complaining?"))
    urgency = Operator("urgency", Score("How urgent is this?", ["low", "medium", "high"]))

    is_complaint(provider, message).value                     # one operator, one call
    answers = apply(provider, message, [is_complaint, urgency])  # both, one call
    answers["urgency"].value

Operators don't hold a provider: you pass it in, so the same operator runs on
any provider. The wording is part of the operator (it changes the answers), so
keep operators in code, under version control, and benchmark them as they are.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from .provider import AsyncProvider, Provider
from .types import Answer, Question, State


@dataclass(frozen=True)
class Operator:
    name: str
    question: Question

    def __call__(self, provider: Provider, state: State) -> Answer:
        return provider.ask(state, {self.name: self.question})[self.name]

    async def call_async(self, provider: AsyncProvider, state: State) -> Answer:
        return (await provider.ask(state, {self.name: self.question}))[self.name]


def questions(operators: Sequence[Operator]) -> dict[str, Question]:
    """The operators as a ``{name: question}`` dict, e.g. for ``bench.run``."""
    names = [op.name for op in operators]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"operator names must be unique; repeated: {', '.join(duplicates)}")
    return {op.name: op.question for op in operators}


def apply(provider: Provider, state: State, operators: Sequence[Operator]) -> dict[str, Answer]:
    """Ask all ``operators`` about ``state`` in one provider call. Answers are keyed by name."""
    return provider.ask(state, questions(operators))


async def apply_async(provider: AsyncProvider, state: State,
                      operators: Sequence[Operator]) -> dict[str, Answer]:
    """``apply`` for an ``AsyncProvider``."""
    return await provider.ask(state, questions(operators))

