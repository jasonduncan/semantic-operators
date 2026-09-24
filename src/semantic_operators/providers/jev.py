"""Jev, via the TypeSafe SDK (``pip install semantic-operators[jev]``).

Translation only: our questions -> SDK questions, one ``system_one`` call,
SDK answers -> our ``Answer``. You create and own the SDK client.
"""

from collections.abc import Mapping

import typesafe_sdk as ts

from ..types import Answer, Boolean, Choice, Question, Score, State


class Jev:
    def __init__(self, client: ts.TypeSafeClient, model: str = "jev-latest") -> None:
        self.client = client
        self.model = model

    def ask(self, state: State, questions: Mapping[str, Question]) -> dict[str, Answer]:
        response = self.client.system_one(
            state=state,
            questions={name: _to_jev(q) for name, q in questions.items()},
            model=self.model,
        )
        return {name: _from_jev(q, response.answers[name]) for name, q in questions.items()}


def _to_jev(question: Question) -> ts.Noul | ts.Choice | ts.Score:
    match question:
        case Boolean():
            described = question.true is not None or question.false is not None
            criteria = {"true": question.true, "false": question.false} if described else None
            return ts.Noul(instructions=question.instructions, criteria=criteria)
        case Choice():
            return ts.Choice(instructions=question.instructions, criteria=dict(question.options))
        case Score():
            return ts.Score(instructions=question.instructions, criteria=list(question.levels))


def _from_jev(question: Question, answer: ts.Answer) -> Answer:
    match question:
        case Boolean():
            # Jev returns one number: the probability the answer is "true".
            p = answer.noul
            return Answer(value=p > 0.5, probabilities={"true": p, "false": 1 - p}, raw=answer)
        case Choice():
            probabilities = {option: answer.probabilities[option] for option in question.options}
            return Answer(value=answer.choice, probabilities=probabilities, raw=answer)
        case Score():
            # Jev keys probabilities by level index (0, 1, 2...); we key them by level text.
            probabilities = {level: answer.probabilities[i] for i, level in enumerate(question.levels)}
            return Answer(value=answer.score, probabilities=probabilities, raw=answer)
