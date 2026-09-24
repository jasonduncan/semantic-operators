"""A tiny benchmark: run labeled cases through a provider and score the answers.

Higher layer: built only on the base layer (types + Provider).

For each question we report:
- accuracy:   how often ``value`` matches the label. For Score, the value is
              rounded to the nearest level first.
- p(correct): the average probability the provider gave the labeled answer.
              Two providers can be equally accurate while one is far more sure
              of itself when it's right (and, worse, when it's wrong).
"""

import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .provider import Provider
from .types import Answer, Boolean, Choice, Question, Score, State


@dataclass(frozen=True)
class Case:
    """One labeled input. ``expected`` maps question name -> label:
    a bool for Boolean, an option name for Choice, a level name for Score."""

    state: State
    expected: dict[str, bool | str]


@dataclass
class QuestionStats:
    correct: int = 0
    total: int = 0
    p_correct: list[float] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def mean_p_correct(self) -> float:
        return statistics.fmean(self.p_correct) if self.p_correct else 0.0


@dataclass
class Miss:
    case: int
    question: str
    expected: bool | str
    got: Answer


@dataclass
class Report:
    questions: dict[str, QuestionStats]
    latencies_ms: list[float]
    misses: list[Miss]


def run(provider: Provider, questions: Mapping[str, Question], cases: Sequence[Case],
        warmup: int = 1) -> Report:
    """Ask every case's state all ``questions`` in one call per case.

    The first ``warmup`` cases are also run once beforehand, untimed, so one-time
    setup (loading, connecting) doesn't count as latency.
    """
    for case in cases[:warmup]:
        provider.ask(case.state, questions)

    stats = {name: QuestionStats() for name in questions}
    latencies: list[float] = []
    misses: list[Miss] = []
    for i, case in enumerate(cases):
        start = time.perf_counter()
        answers = provider.ask(case.state, questions)
        latencies.append((time.perf_counter() - start) * 1000)

        for name, label in case.expected.items():
            answer, s = answers[name], stats[name]
            ok = _matches(questions[name], answer, label)
            s.total += 1
            s.correct += ok
            s.p_correct.append(answer.probabilities[_key(label)])
            if not ok:
                misses.append(Miss(i, name, label, answer))
    return Report(stats, latencies, misses)


def _matches(question: Question, answer: Answer, label: bool | str) -> bool:
    match question:
        case Boolean() | Choice():
            return answer.value == label
        case Score():
            # Round half up (Python's round() would send 0.5 to 0 but 1.5 to 2).
            return question.levels[int(float(answer.value) + 0.5)] == label


def _key(label: bool | str) -> str:
    # Boolean probabilities are keyed "true"/"false"; everything else by its label.
    return str(label).lower() if isinstance(label, bool) else label
