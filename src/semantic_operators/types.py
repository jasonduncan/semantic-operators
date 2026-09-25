"""The three question types, the one answer type, and how answers are built.

These are our words, not any provider's. A provider translates them into
its own API and translates its answers back with ``make_answer``.
"""

from dataclasses import dataclass, field
from typing import Any

# What a question is asked about: text, or JSON-shaped data.
State = str | dict[str, Any] | list[Any]


# Every question takes an optional keyword ``min_confidence`` (0 to 1). When the
# model's confidence in its answer is below it, the answer comes back undecided
# (``value is None``) instead of as a guess. The default, 0, never withholds an answer.


@dataclass(frozen=True)
class Boolean:
    """A yes/no question. ``true``/``false`` optionally describe each outcome."""

    instructions: str
    true: str | None = None
    false: str | None = None
    min_confidence: float = field(default=0.0, kw_only=True)


@dataclass(frozen=True)
class Choice:
    """Pick one of several named options. Maps option name -> description."""

    instructions: str
    options: dict[str, str | None]
    min_confidence: float = field(default=0.0, kw_only=True)


@dataclass(frozen=True)
class Score:
    """Rate on an ordered rubric. ``levels[0]`` is score 0, ``levels[1]`` is 1, ..."""

    instructions: str
    levels: list[str]
    min_confidence: float = field(default=0.0, kw_only=True)


Question = Boolean | Choice | Score


@dataclass(frozen=True)
class Answer:
    """The answer to one question.

    - Boolean: ``value`` is a bool; ``probabilities`` has keys "true" and "false".
    - Choice:  ``value`` is the chosen option name; ``probabilities`` is keyed by option,
               in the same order as the question's options.
    - Score:   ``value`` is the expected score (a float, may fall between levels);
               ``probabilities`` is keyed by level text, in level order.

    ``value`` is ``None`` when the answer is undecided: the model was split evenly
    between answers, or its ``confidence`` was below the question's ``min_confidence``.
    ``probabilities`` are always kept, so you can still see what it was leaning toward.

    ``confidence`` is the probability of the model's own answer (for Score, of the
    nearest level). It's the model's view, not a guarantee: check it with a benchmark.

    ``raw`` is the provider's own answer object, for when you need more.
    """

    value: bool | str | float | None
    probabilities: dict[str, float]
    confidence: float
    raw: Any = field(default=None, repr=False)

    @property
    def decided(self) -> bool:
        return self.value is not None


def make_answer(question: Question, value: bool | str | float, probabilities: dict[str, float],
                raw: Any = None) -> Answer:
    """Build an ``Answer`` from a provider's value and probabilities.

    Providers call this so every provider handles ties and ``min_confidence`` the same way.
    """
    match question:
        case Boolean():
            confidence = max(probabilities["true"], probabilities["false"])
            tied = probabilities["true"] == probabilities["false"]
        case Choice():
            confidence = max(probabilities.values())
            tied = list(probabilities.values()).count(confidence) > 1
        case Score():
            nearest = min(int(float(value) + 0.5), len(question.levels) - 1)
            confidence = probabilities[question.levels[nearest]]
            tied = False  # an expected score is always a single number
    if tied or confidence < question.min_confidence:
        return Answer(None, probabilities, confidence, raw)
    return Answer(value, probabilities, confidence, raw)
