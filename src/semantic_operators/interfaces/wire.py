"""The JSON shape shared by the CLI and the MCP server. Standard library only.

A request:

    {
      "state": "The payment failed and now I cannot sign in.",
      "questions": [
        {"name": "is_complaint", "type": "boolean",
         "instructions": "Is the sender reporting a problem?", "min_confidence": 0.8},
        {"name": "department", "type": "choice",
         "instructions": "Which department should handle this?",
         "options": [{"name": "billing", "description": "Charges and refunds"},
                     "technical"]},
        {"name": "urgency", "type": "score",
         "instructions": "How urgent is this?", "levels": ["low", "medium", "high"]}
      ],
      "request_id": "ticket-42"
    }

``state`` is text, a JSON object, or an array. Questions and options are arrays so
their order is explicit; an option is a name or ``{"name", "description"}``. A Boolean
can describe its outcomes with ``"true"`` and ``"false"``. ``min_confidence`` is
optional everywhere. ``request_id`` is optional and echoed back. ``schema_version``
is optional; if given it must be ``"1"``, so a future format can't be misread. Unknown
fields are rejected, so typos fail loudly instead of being ignored.

A response has ``answers`` keyed by question name, each with ``type``, ``value``
(``null`` when undecided), ``decided``, ``confidence``, and ``probabilities`` (in the
question's option/level order), plus one ``call`` for the provider call. A failure is
``{"error": {"code", "message"}}``. ``Answer.raw`` is never included.
"""

from collections.abc import Mapping
from typing import Any

from ..types import Answer, Boolean, Choice, Question, Score, State

_REQUEST_FIELDS = {"schema_version", "state", "questions", "request_id"}
_QUESTION_FIELDS = {
    "boolean": {"name", "type", "instructions", "min_confidence", "true", "false"},
    "choice": {"name", "type", "instructions", "min_confidence", "options"},
    "score": {"name", "type", "instructions", "min_confidence", "levels"},
}


class WireError(ValueError):
    """A request that doesn't match the shape above. ``path`` says where."""

    def __init__(self, path: str, message: str) -> None:
        super().__init__(f"{path}: {message}" if path else message)
        self.path = path


def decode(request: Any) -> tuple[State, dict[str, Question], str | None]:
    """Turn a parsed JSON request into ``(state, questions, request_id)``."""
    if not isinstance(request, dict):
        raise WireError("", "the request must be a JSON object")
    _no_unknown(request, _REQUEST_FIELDS, "")
    if request.get("schema_version", "1") != "1":
        raise WireError("schema_version", 'only "1" is supported')
    if "state" not in request:
        raise WireError("state", "required")
    state = request["state"]
    if not isinstance(state, (str, dict, list)):
        raise WireError("state", "must be text, a JSON object, or an array")
    request_id = request.get("request_id")
    if request_id is not None and not isinstance(request_id, str):
        raise WireError("request_id", "must be a string")
    specs = request.get("questions")
    if not isinstance(specs, list) or not specs:
        raise WireError("questions", "must be a non-empty array")

    questions: dict[str, Question] = {}
    for i, spec in enumerate(specs):
        path = f"questions[{i}]"
        name, question = _question(spec, path)
        if name in questions:
            raise WireError(f"{path}.name", f"{name!r} is used twice")
        questions[name] = question
    return state, questions, request_id


def encode(answers: Mapping[str, Answer], questions: Mapping[str, Question],
           request_id: str | None = None) -> dict[str, Any]:
    """The JSON response for a set of answers (all from one provider call)."""
    first = next(iter(answers.values()))
    call = first.call
    response: dict[str, Any] = {} if request_id is None else {"request_id": request_id}
    response["answers"] = {
        name: {
            "type": _TYPES[type(questions[name])],
            "value": answer.value,
            "decided": answer.decided,
            "confidence": answer.confidence,
            "probabilities": answer.probabilities,
        }
        for name, answer in answers.items()
    }
    response["call"] = None if call is None else {
        "provider": call.provider, "model": call.model,
        "input_tokens": call.input_tokens, "output_tokens": call.output_tokens,
    }
    return response


def error(code: str, message: str, request_id: str | None = None) -> dict[str, Any]:
    """A failure response. ``code``: invalid_request, provider_error, or timeout."""
    response: dict[str, Any] = {} if request_id is None else {"request_id": request_id}
    response["error"] = {"code": code, "message": message}
    return response


_TYPES = {Boolean: "boolean", Choice: "choice", Score: "score"}


def _question(spec: Any, path: str) -> tuple[str, Question]:
    if not isinstance(spec, dict):
        raise WireError(path, "must be a JSON object")
    kind = spec.get("type")
    if kind not in _QUESTION_FIELDS:
        raise WireError(f"{path}.type", "must be 'boolean', 'choice', or 'score'")
    _no_unknown(spec, _QUESTION_FIELDS[kind], path)
    name = spec.get("name")
    if not isinstance(name, str) or not name:
        raise WireError(f"{path}.name", "must be a non-empty string")
    extra: dict[str, Any] = {}
    if "min_confidence" in spec:
        extra["min_confidence"] = spec["min_confidence"]
    instructions = spec.get("instructions")
    try:
        match kind:
            case "boolean":
                for side in ("true", "false"):
                    if spec.get(side) is not None and not isinstance(spec[side], str):
                        raise WireError(f"{path}.{side}", "must be a string")
                return name, Boolean(instructions, spec.get("true"), spec.get("false"), **extra)
            case "choice":
                return name, Choice(instructions, _options(spec.get("options"), path), **extra)
            case _:
                levels = spec.get("levels")
                if not isinstance(levels, list):
                    raise WireError(f"{path}.levels", "must be an array of strings")
                return name, Score(instructions, list(levels), **extra)
    except WireError:
        raise
    except ValueError as problem:  # the question types check themselves
        raise WireError(path, str(problem)) from None


def _options(options: Any, path: str) -> dict[str, str | None]:
    if not isinstance(options, list):
        raise WireError(f"{path}.options", "must be an array")
    result: dict[str, str | None] = {}
    for i, option in enumerate(options):
        where = f"{path}.options[{i}]"
        if isinstance(option, str):
            name, description = option, None
        elif isinstance(option, dict):
            _no_unknown(option, {"name", "description"}, where)
            name, description = option.get("name"), option.get("description")
            if description is not None and not isinstance(description, str):
                raise WireError(f"{where}.description", "must be a string")
        else:
            raise WireError(where, "must be a name or {\"name\", \"description\"}")
        if not isinstance(name, str) or not name:
            raise WireError(where, "needs a non-empty name")
        if name in result:
            raise WireError(where, f"option {name!r} is used twice")
        result[name] = description
    return result


def _no_unknown(obj: dict[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(obj) - allowed)
    if unknown:
        where = f"{path}." if path else ""
        raise WireError(f"{where}{unknown[0]}", "unknown field")
