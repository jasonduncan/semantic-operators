"""Laya, the open-weight Jev-compatible model (``pip install semantic-operators[laya]``).

Runs locally. You load the model with the ``laya`` package and pass it in:
``laya.load("convaiinnovations/laya")`` for one checkpoint, or ``laya.Router()`` to
pick a checkpoint by language. Both have the ``predict`` method used here.

``AsyncLaya`` runs predictions in a worker thread so they don't block the event
loop, one at a time (the model is a single local compute resource; Laya's own
HTTP server serializes calls the same way).

Any failure inside the model (for example, options too long to fit, out of memory)
or an unexpected result is raised as ``ProviderError``. The ``laya`` package has
no error base class of its own, so every exception from ``predict`` is wrapped.
"""

import asyncio
from collections.abc import Mapping
from typing import Any

from ..errors import ProviderError
from ..types import Answer, Boolean, Choice, Question, Score, State, make_answer


class Laya:
    def __init__(self, model: Any) -> None:
        self.model = model

    def ask(self, state: State, questions: Mapping[str, Question]) -> dict[str, Answer]:
        try:
            result = self.model.predict(state, {name: _to_laya(q) for name, q in questions.items()})
        except Exception as error:
            raise ProviderError("Laya", str(error) or type(error).__name__) from error
        try:
            answers = result["answers"]
            return {name: _from_laya(q, answers[name]) for name, q in questions.items()}
        except (KeyError, TypeError, ValueError) as error:
            raise ProviderError("Laya", f"unexpected result: {error!r}") from error


class AsyncLaya:
    def __init__(self, model: Any) -> None:
        self._laya = Laya(model)
        self._lock = asyncio.Lock()

    async def ask(self, state: State, questions: Mapping[str, Question]) -> dict[str, Answer]:
        async with self._lock:
            return await asyncio.to_thread(self._laya.ask, state, questions)


# Laya takes and returns plain dicts in Jev's request/response shape.


def _to_laya(question: Question) -> dict[str, Any]:
    match question:
        case Boolean():
            q: dict[str, Any] = {"type": "noul", "instructions": question.instructions}
            if question.true is not None or question.false is not None:
                q["criteria"] = {"true": question.true, "false": question.false}
            return q
        case Choice():
            return {"type": "choice", "instructions": question.instructions,
                    "criteria": dict(question.options)}
        case Score():
            return {"type": "score", "instructions": question.instructions,
                    "criteria": list(question.levels)}


def _from_laya(question: Question, answer: dict[str, Any]) -> Answer:
    match question:
        case Boolean():
            p = answer["noul"]
            return make_answer(question, p > 0.5, {"true": p, "false": 1 - p}, answer)
        case Choice():
            probabilities = {option: answer["probabilities"][option] for option in question.options}
            return make_answer(question, answer["choice"], probabilities, answer)
        case Score():
            # Laya keys probabilities by level index as a string ("0", "1", ...).
            probabilities = {level: answer["probabilities"][str(i)]
                             for i, level in enumerate(question.levels)}
            return make_answer(question, answer["score"], probabilities, answer)
