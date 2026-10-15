"""TypeSafe's hosted API, which serves the Jev models (``pip install semantic-operators[typesafe]``).

Translation only: our questions -> SDK questions, one ``system_one`` call,
SDK answers -> our ``Answer``. You create and own the SDK client:
``TypeSafeClient`` for ``TypeSafe``, ``AsyncTypeSafeClient`` for ``AsyncTypeSafe``.
The model is a setting: ``jev-latest`` by default, or pin one such as ``jev-1.13.0``.

Any SDK failure (network, auth, rate limit, server error) or an unexpected
response is raised as ``ProviderError`` (``ProviderTimeout`` when the SDK times
out). Retries and timeouts are the client's: the
SDK retries by default, so pass ``retry=ts.RetryPolicy(max_retries=0, timeout=...)``
when you want every failure to reach you. Each answer's ``call`` records the model
TypeSafe reports and the tokens it used.

Images (``semantic_operators.Image``) in the state or in any question text are sent as
``assets`` (multipart file parts), and every request with an image or a Box question
carries the ``TypeSafe-Danger-Zone-Images: true`` header. An image that breaks the
limits raises ``ImageError`` (a ``ValueError``) before anything is sent. Box questions
need ``model="jev-1.14.0"`` (or ``jev-preview``).
"""

from collections.abc import Mapping
from typing import Any

import typesafe_sdk as ts

from ..errors import ProviderError, ProviderTimeout
from ..images import collect, replace_images
from ..types import (Answer, Boolean, Box, Call, Choice, Question, Region, Score, State, complement, make_answer,
                     token_count)


class TypeSafe:
    def __init__(self, client: ts.TypeSafeClient, model: str = "jev-latest") -> None:
        self.client = client
        self.model = model

    def ask(self, state: State, questions: Mapping[str, Question]) -> dict[str, Answer]:
        try:
            response = self.client.system_one(**_request(state, questions, self.model))
        except ts.TypeSafeError as error:
            raise _error(error) from error
        return _decode(questions, response)


class AsyncTypeSafe:
    def __init__(self, client: ts.AsyncTypeSafeClient, model: str = "jev-latest") -> None:
        self.client = client
        self.model = model

    async def ask(self, state: State, questions: Mapping[str, Question]) -> dict[str, Answer]:
        try:
            response = await self.client.system_one(**_request(state, questions, self.model))
        except ts.TypeSafeError as error:
            raise _error(error) from error
        return _decode(questions, response)


IMAGES_HEADER = "TypeSafe-Danger-Zone-Images"


def _request(state: State, questions: Mapping[str, Question], model: str) -> dict[str, Any]:
    """The ``system_one`` arguments. Images go in ``assets`` keyed by their placeholder
    (the SDK sends them as multipart file parts), with the images header turned on."""
    images = collect(state, dict(questions))  # raises ImageError (a ValueError) before sending
    request: dict[str, Any] = {
        "state": replace_images(state),
        "questions": {name: _to_typesafe(q) for name, q in questions.items()},
        "model": model,
    }
    if images or any(isinstance(q, Box) for q in questions.values()):
        request["assets"] = {ph: ts.Image(image.data) for ph, (image, _) in images.items()}
        request["extra_headers"] = {IMAGES_HEADER: "true"}
    return request


def _error(error: ts.TypeSafeError) -> ProviderError:
    kind = ProviderTimeout if isinstance(error, TimeoutError) else ProviderError
    return kind("TypeSafe", str(error) or type(error).__name__)


def _decode(questions: Mapping[str, Question], response: ts.SystemOneResponse) -> dict[str, Answer]:
    try:
        usage = response.usage
        call = Call("TypeSafe", response.model if isinstance(response.model, str) else None,
                    token_count(usage.input_tokens), token_count(usage.output_tokens))
        return {name: _from_typesafe(q, response.answers[name], call)
                for name, q in questions.items()}
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise ProviderError("TypeSafe", f"unexpected response: {error!r}") from error


def _to_typesafe(question: Question) -> ts.Noul | ts.Choice | ts.Score | ts.Box:
    match question:
        case Boolean():
            described = question.true is not None or question.false is not None
            criteria = {"true": question.true, "false": question.false} if described else None
            return ts.Noul(instructions=question.instructions, criteria=criteria)
        case Choice():
            return ts.Choice(instructions=question.instructions, criteria=dict(question.options))
        case Score():
            return ts.Score(instructions=question.instructions, criteria=list(question.levels))
        case Box():
            return ts.Box(instructions=question.instructions)


def _from_typesafe(question: Question, answer: ts.Answer, call: Call) -> Answer:
    match question:
        case Boolean():
            # TypeSafe returns one number: the probability the answer is "true".
            p = answer.noul
            probabilities = {"true": p, "false": complement(p)}
            return make_answer(question, p > 0.5, probabilities, answer, call=call)
        case Choice():
            probabilities = {option: answer.probabilities[option] for option in question.options}
            return make_answer(question, answer.choice, probabilities, answer, call=call)
        case Score():
            # TypeSafe keys probabilities by level index (0, 1, 2...); we key them by level text.
            probabilities = {level: answer.probabilities[i] for i, level in enumerate(question.levels)}
            return make_answer(question, answer.score, probabilities, answer, call=call)
        case Box():
            b = answer.box
            return make_answer(question, Region(b.top, b.bottom, b.left, b.right), {}, answer,
                               call=call, confidence=answer.confidence)
