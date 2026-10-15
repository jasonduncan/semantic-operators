"""The three question types, the one answer type, and how answers are built.

These are our words, not any provider's. A provider translates them into
its own API and translates its answers back with ``make_answer``.
"""

import math
from dataclasses import dataclass, field
from typing import Any

# What a question is asked about: text, or JSON-shaped data. Any value (or the whole
# state) may be an ``images.Image``; see images.py.
State = str | dict[str, Any] | list[Any]


# Every question takes an optional keyword ``min_confidence`` (0 to 1). When the
# model's confidence in its answer is below it, the answer comes back undecided
# (``value is None``) instead of as a guess. The default, 0, never withholds an answer.
#
# Questions check themselves when created and raise ValueError if they're malformed.


def _check_question(instructions: Any, min_confidence: Any) -> None:
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError("instructions must be a non-empty string")
    if not _is_number(min_confidence) or not 0 <= min_confidence <= 1:
        raise ValueError(f"min_confidence must be a number from 0 to 1, got {min_confidence!r}")


def _check_labels(labels: Any, what: str) -> None:
    if len(labels) < 2:
        raise ValueError(f"need at least 2 {what}, got {len(labels)}")
    if not all(isinstance(label, str) and label for label in labels):
        raise ValueError(f"{what} must be non-empty strings")
    if len(set(labels)) != len(labels):
        raise ValueError(f"{what} must be unique")


@dataclass(frozen=True)
class Boolean:
    """A yes/no question. ``true``/``false`` optionally describe each outcome."""

    instructions: str
    true: str | None = None
    false: str | None = None
    min_confidence: float = field(default=0.0, kw_only=True)

    def __post_init__(self) -> None:
        _check_question(self.instructions, self.min_confidence)


@dataclass(frozen=True)
class Choice:
    """Pick one of several named options. Maps option name -> description."""

    instructions: str
    options: dict[str, str | None]
    min_confidence: float = field(default=0.0, kw_only=True)

    def __post_init__(self) -> None:
        _check_question(self.instructions, self.min_confidence)
        _check_labels(list(self.options), "options")


@dataclass(frozen=True)
class Score:
    """Rate on an ordered rubric. ``levels[0]`` is score 0, ``levels[1]`` is 1, ..."""

    instructions: str
    levels: list[str]
    min_confidence: float = field(default=0.0, kw_only=True)

    def __post_init__(self) -> None:
        _check_question(self.instructions, self.min_confidence)
        _check_labels(self.levels, "levels")  # levels become probability keys


@dataclass(frozen=True)
class Box:
    """Locate one thing in an image: the answer's ``value`` is a ``Region``.

    Needs an image in the request, and a model that supports it (TypeSafe:
    ``jev-1.14.0`` / ``jev-preview`` only). There is no "not found" answer: the
    model always returns some box, so pair it with a presence check (``Boolean``),
    or use ``semantic_operators.locate``, which does. One box per question: ask
    one Box per target.
    """

    instructions: str
    min_confidence: float = field(default=0.0, kw_only=True)

    def __post_init__(self) -> None:
        _check_question(self.instructions, self.min_confidence)


@dataclass(frozen=True)
class Region:
    """A box's edges as fractions (0 to 1) of the image's height (top, bottom) and width (left, right)."""

    top: float
    bottom: float
    left: float
    right: float

    def __post_init__(self) -> None:
        edges = (self.top, self.bottom, self.left, self.right)
        if not all(_is_number(e) and 0 <= e <= 1 for e in edges):
            raise ValueError(f"box edges must be numbers from 0 to 1, got {edges!r}")
        if self.top > self.bottom or self.left > self.right:
            raise ValueError(f"box edges are out of order: {self!r}")

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def center(self) -> tuple[float, float]:
        """(x, y) as fractions of width and height."""
        return ((self.left + self.right) / 2, (self.top + self.bottom) / 2)

    def padded(self, fraction: float) -> "Region":
        """Grown by ``fraction`` of its own size on every side, clipped to the image.

        Boxes can be tight or cut a little off; padding before cropping is the cheap fix.
        """
        dx, dy = self.width * fraction, self.height * fraction
        return Region(max(0.0, self.top - dy), min(1.0, self.bottom + dy),
                      max(0.0, self.left - dx), min(1.0, self.right + dx))

    def pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        """(left, top, right, bottom) in pixels, e.g. for ``PIL.Image.crop``."""
        return (round(self.left * width), round(self.top * height),
                round(self.right * width), round(self.bottom * height))


Question = Boolean | Choice | Score | Box


@dataclass(frozen=True)
class Call:
    """What one provider call reported about itself. Every answer from that call shares it.

    ``model`` is the model the provider says answered, exactly as reported. It can differ
    from the name you asked for (an alias such as ``jev-latest``), and it's only as
    specific as the provider makes it. Token counts are ``None`` when not reported.
    """

    provider: str
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True)
class Answer:
    """The answer to one question.

    - Boolean: ``value`` is a bool; ``probabilities`` has keys "true" and "false".
    - Choice:  ``value`` is the chosen option name; ``probabilities`` is keyed by option,
               in the same order as the question's options.
    - Score:   ``value`` is the expected score (a float, may fall between levels);
               ``probabilities`` is keyed by level text, in level order.
    - Box:     ``value`` is a ``Region``; ``probabilities`` is empty, and ``confidence``
               is the provider's confidence in the box (0 when it reports none).

    ``value`` is ``None`` when the answer is undecided: the model was split evenly
    between answers, or its ``confidence`` was below the question's ``min_confidence``.
    ``probabilities`` are always kept, so you can still see what it was leaning toward.

    ``confidence`` is the probability of the model's own answer: the chosen option for
    Choice, the chosen side for Boolean, the nearest level for Score. It's the model's
    view, not a guarantee: check it with a benchmark.

    ``raw`` is the provider's own answer object, for when you need more.

    ``call`` records which model answered and what the call used (see ``Call``).
    """

    value: bool | str | float | Region | None
    probabilities: dict[str, float]
    confidence: float
    raw: Any = field(default=None, repr=False)
    call: Call | None = field(default=None, repr=False, kw_only=True)

    @property
    def decided(self) -> bool:
        return self.value is not None


# Providers round what they return (TypeSafe to 2 decimals, Laya to 4), so each
# probability or score may be off by up to half a hundredth. Totals and expected
# scores are checked with margins that allow exactly that much rounding and no more.
ROUNDING = 0.005


def make_answer(question: Question, value: bool | str | float | Region,
                probabilities: dict[str, float], raw: Any = None, *, call: Call | None = None,
                confidence: float | None = None) -> Answer:
    """Build an ``Answer`` from a provider's value and probabilities.

    Providers call this so every provider handles ties and ``min_confidence`` the same way.
    It also checks the provider's output: the probabilities must cover exactly the
    question's outcomes, be finite, lie in [0, 1] and sum to 1, and the value must agree
    with them. Malformed output raises ValueError, which providers turn into ProviderError,
    so a broken response never looks like a confident answer.

    ``confidence`` is for Box answers only (they have no probabilities): the provider's
    confidence in the box, or ``None`` when it reports none (treated as 0).
    """
    if not isinstance(question, Box) and confidence is not None:
        raise ValueError("confidence is given only for Box answers")
    match question:
        case Boolean():
            _check_distribution(probabilities, ["true", "false"])
            p_true, p_false = probabilities["true"], probabilities["false"]
            tied = p_true == p_false
            if not isinstance(value, bool) or (not tied and value != (p_true > p_false)):
                raise ValueError(f"Boolean value {value!r} disagrees with its probabilities")
            confidence = max(p_true, p_false)
        case Choice():
            _check_distribution(probabilities, list(question.options))
            if value not in question.options:
                raise ValueError(f"Choice value {value!r} is not one of the options")
            top = max(probabilities.values())
            tied = list(probabilities.values()).count(top) > 1
            if probabilities[value] != top:
                raise ValueError(f"Choice value {value!r} is not the most likely option")
            confidence = probabilities[value]
        case Score():
            _check_distribution(probabilities, question.levels)
            highest = len(question.levels) - 1
            if not _is_number(value) or not 0 <= value <= highest:
                raise ValueError(f"Score value {value!r} is outside 0..{highest}")
            expected = sum(i * probabilities[level] for i, level in enumerate(question.levels))
            margin = ROUNDING * (1 + sum(range(len(question.levels))))  # score + each level
            if abs(value - expected) > margin:
                raise ValueError(f"Score value {value!r} disagrees with its probabilities")
            confidence = probabilities[question.levels[min(int(value + 0.5), highest)]]
            tied = False  # an expected score is always a single number
        case Box():
            if not isinstance(value, Region):
                raise ValueError(f"Box value {value!r} is not a Region")
            if probabilities:
                raise ValueError("Box answers have no probabilities")
            confidence = 0.0 if confidence is None else confidence
            if not _is_number(confidence) or not 0 <= confidence <= 1:
                raise ValueError(f"Box confidence {confidence!r} is not from 0 to 1")
            tied = False
    if tied or confidence < question.min_confidence:
        return Answer(None, probabilities, confidence, raw, call=call)
    return Answer(value, probabilities, confidence, raw, call=call)


def complement(p: float) -> float:
    """``1 - p`` without float noise, for providers that report only p(true).

    ``1 - 0.07`` is 0.9299999999999999, which would put a 0.93 answer under a
    ``min_confidence`` of 0.93. Rounding to 10 places removes the noise and nothing else.
    """
    return round(1 - p, 10)


def token_count(x: Any) -> int | None:
    """A reported token count, or ``None`` if what was reported isn't a count."""
    return x if isinstance(x, int) and not isinstance(x, bool) and x >= 0 else None


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _check_distribution(probabilities: Any, outcomes: list[str]) -> None:
    if not isinstance(probabilities, dict) or set(probabilities) != set(outcomes):
        raise ValueError(f"probabilities must cover exactly {outcomes}")
    if not all(_is_number(p) and 0 <= p <= 1 for p in probabilities.values()):
        raise ValueError("probabilities must be finite numbers from 0 to 1")
    if abs(math.fsum(probabilities.values()) - 1) > ROUNDING * len(outcomes) + 1e-9:
        raise ValueError("probabilities must sum to 1")
