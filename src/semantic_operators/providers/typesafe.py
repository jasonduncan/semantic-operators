"""TypeSafe's hosted API, which serves the Jev models (``pip install semantic-operators[typesafe]``).

Translation only: our questions -> SDK questions, one ``system_one`` call,
SDK answers -> our ``Answer``. You create and own the SDK client:
``TypeSafeClient`` for ``TypeSafe``, ``AsyncTypeSafeClient`` for ``AsyncTypeSafe``.
The model is a setting: ``jev-latest`` by default, or pin one such as ``jev-1.13.0``.

Any SDK failure (network, auth, rate limit, server error) or an unexpected
response is raised as ``ProviderError``. The SDK's own retry policy still applies.
"""

from collections.abc import Mapping

import typesafe_sdk as ts

from ..errors import ProviderError
from ..types import Answer, Boolean, Choice, Question, Score, State, make_answer


class TypeSafe:
    def __init__(self, client: ts.TypeSafeClient, model: str = "jev-latest") -> None:
        self.client = client
        self.model = model

    def ask(self, state: State, questions: Mapping[str, Question]) -> dict[str, Answer]:
        try:
            response = self.client.system_one(
                state=state,
                questions={name: _to_typesafe(q) for name, q in questions.items()},
                model=self.model,
            )
        except ts.TypeSafeError as error:
            raise ProviderError("TypeSafe", str(error) or type(error).__name__) from error
        return _decode(questions, response)


class AsyncTypeSafe:
    def __init__(self, client: ts.AsyncTypeSafeClient, model: str = "jev-latest") -> None:
        self.client = client
        self.model = model

    async def ask(self, state: State, questions: Mapping[str, Question]) -> dict[str, Answer]:
        try:
            response = await self.client.system_one(
                state=state,
                questions={name: _to_typesafe(q) for name, q in questions.items()},
                model=self.model,
            )
        except ts.TypeSafeError as error:
            raise ProviderError("TypeSafe", str(error) or type(error).__name__) from error
        return _decode(questions, response)


def _decode(questions: Mapping[str, Question], response: ts.SystemOneResponse) -> dict[str, Answer]:
    try:
        return {name: _from_typesafe(q, response.answers[name]) for name, q in questions.items()}
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise ProviderError("TypeSafe", f"unexpected response: {error!r}") from error


def _to_typesafe(question: Question) -> ts.Noul | ts.Choice | ts.Score:
    match question:
        case Boolean():
            described = question.true is not None or question.false is not None
            criteria = {"true": question.true, "false": question.false} if described else None
            return ts.Noul(instructions=question.instructions, criteria=criteria)
        case Choice():
            return ts.Choice(instructions=question.instructions, criteria=dict(question.options))
        case Score():
            return ts.Score(instructions=question.instructions, criteria=list(question.levels))


def _from_typesafe(question: Question, answer: ts.Answer) -> Answer:
    match question:
        case Boolean():
            # TypeSafe returns one number: the probability the answer is "true".
            p = answer.noul
            return make_answer(question, p > 0.5, {"true": p, "false": 1 - p}, answer)
        case Choice():
            probabilities = {option: answer.probabilities[option] for option in question.options}
            return make_answer(question, answer.choice, probabilities, answer)
        case Score():
            # TypeSafe keys probabilities by level index (0, 1, 2...); we key them by level text.
            probabilities = {level: answer.probabilities[i] for i, level in enumerate(question.levels)}
            return make_answer(question, answer.score, probabilities, answer)
