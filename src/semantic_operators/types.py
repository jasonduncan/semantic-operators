"""The three question types and the one answer type.

These are our words, not any provider's. A provider translates them into
its own API and translates its answers back into an ``Answer``.
"""

from dataclasses import dataclass, field
from typing import Any

# What a question is asked about: text, or JSON-shaped data.
State = str | dict[str, Any] | list[Any]


@dataclass(frozen=True)
class Boolean:
    """A yes/no question. ``true``/``false`` optionally describe each outcome."""

    instructions: str
    true: str | None = None
    false: str | None = None


@dataclass(frozen=True)
class Choice:
    """Pick one of several named options. Maps option name -> description."""

    instructions: str
    options: dict[str, str | None]


@dataclass(frozen=True)
class Score:
    """Rate on an ordered rubric. ``levels[0]`` is score 0, ``levels[1]`` is 1, ..."""

    instructions: str
    levels: list[str]


Question = Boolean | Choice | Score


@dataclass(frozen=True)
class Answer:
    """The answer to one question.

    - Boolean: ``value`` is a bool; ``probabilities`` has keys "true" and "false".
    - Choice:  ``value`` is the chosen option name; ``probabilities`` is keyed by option.
    - Score:   ``value`` is the expected score (a float, may fall between levels);
               ``probabilities`` is keyed by level text.

    ``raw`` is the provider's own answer object, for when you need more.
    """

    value: bool | str | float
    probabilities: dict[str, float]
    raw: Any = field(default=None, repr=False)
