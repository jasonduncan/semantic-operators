"""A tiny benchmark: run labeled cases through a provider and score the answers.

Higher layer: built only on the base layer (types + Provider).

For each question we report:
- accuracy:   how often the provider's decision matches the label.
- p(correct): the average probability the provider gave the labeled answer.
              Two providers can be equally accurate while one is far more sure
              of itself when it's right (and, worse, when it's wrong).

``stability`` compares runs of differently worded versions of the same
questions: how often does the decision stay the same when only the wording changes?
"""

import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .provider import Provider
from .types import Answer, Boolean, Choice, Question, Score, State

# A decision is an answer reduced to one discrete label:
# a bool for Boolean, an option name for Choice, a level index (0, 1, ...) for Score.
Decision = bool | str | int


@dataclass(frozen=True)
class Case:
    """One labeled input. ``expected`` maps question name -> the correct decision.

    Score labels are level indexes, not level text, so the same labels still apply
    when the rubric is reworded.
    """

    state: State
    expected: dict[str, Decision]


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
    expected: Decision
    got: Answer


@dataclass
class Report:
    questions: dict[str, QuestionStats]
    latencies_ms: list[float]
    misses: list[Miss]
    decisions: list[dict[str, Decision]]  # per case, per question


def decide(question: Question, answer: Answer) -> Decision:
    """Reduce an answer to a discrete decision (Score: round half up to a level index)."""
    match question:
        case Boolean() | Choice():
            return answer.value
        case Score():
            return int(float(answer.value) + 0.5)


def run(provider: Provider, questions: Mapping[str, Question], cases: Sequence[Case],
        warmup: int = 1) -> Report:
    """Ask every case's state all ``questions`` in one call per case.

    The first ``warmup`` cases are also run once beforehand, untimed, so one-time
    setup (loading, connecting) doesn't count as latency.
    """
    for case in cases[:warmup]:
        provider.ask(case.state, questions)

    answers: list[dict[str, Answer]] = []
    latencies: list[float] = []
    for case in cases:
        start = time.perf_counter()
        answers.append(provider.ask(case.state, questions))
        latencies.append((time.perf_counter() - start) * 1000)
    return score(questions, cases, answers, latencies)


def score(questions: Mapping[str, Question], cases: Sequence[Case],
          answers: Sequence[Mapping[str, Answer]], latencies_ms: list[float]) -> Report:
    """Score already-collected answers (one dict of answers per case) against the labels."""
    stats = {name: QuestionStats() for name in questions}
    misses: list[Miss] = []
    decisions: list[dict[str, Decision]] = []
    for i, (case, case_answers) in enumerate(zip(cases, answers, strict=True)):
        decisions.append({name: decide(q, case_answers[name]) for name, q in questions.items()})
        for name, label in case.expected.items():
            answer, s = case_answers[name], stats[name]
            ok = decisions[-1][name] == label
            s.total += 1
            s.correct += ok
            s.p_correct.append(answer.probabilities[_key(questions[name], label)])
            if not ok:
                misses.append(Miss(i, name, label, answer))
    return Report(stats, latencies_ms, misses, decisions)


def stability(reports: Sequence[Report]) -> dict[str, float]:
    """Per question: the fraction of cases whose decision is identical in every report.

    Pass reports from differently worded versions of the same questions. Labels play
    no part: a model can be perfectly stable and consistently wrong.
    """
    names = reports[0].decisions[0].keys()
    cases = range(len(reports[0].decisions))
    return {name: sum(len({r.decisions[i][name] for r in reports}) == 1 for i in cases) / len(cases)
            for name in names}


def _key(question: Question, label: Decision) -> str:
    # Answer.probabilities keys: "true"/"false", option names, or level text.
    match question:
        case Boolean():
            return str(label).lower()
        case Choice():
            return str(label)
        case Score():
            return question.levels[int(label)]
