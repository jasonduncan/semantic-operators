"""A tiny benchmark: run labeled cases through a provider and score the answers.

Higher layer: built only on the base layer (types + Provider).

For each question we report:
- accuracy:   how often the provider's decision matches the label. An undecided
              answer ("don't know") counts as not correct, but not as a miss.
- answered:   how many cases got a decision at all (the rest were undecided).
- p(correct): the average probability the provider gave the labeled answer.
              Two providers can be equally accurate while one is far more sure
              of itself when it's right (and, worse, when it's wrong).

``at_min_confidence`` re-scores a report as if every question had a stricter
``min_confidence``, without asking the provider again: the trade-off between
answering fewer cases and being right more often when it does answer.

``run_async`` is ``run`` for an ``AsyncProvider``, with up to ``concurrency``
calls in flight at once.

``stability`` compares runs of differently worded versions of the same
questions: how often does the decision stay the same when only the wording changes?
"""

import asyncio
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

from .provider import AsyncProvider, Provider
from .types import Answer, Boolean, Choice, Question, Score, State

# A decision is an answer reduced to one discrete label:
# a bool for Boolean, an option name for Choice, a level index (0, 1, ...) for Score,
# or None when the answer is undecided.
Decision = bool | str | int | None


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
    undecided: int = 0
    total: int = 0
    p_correct: list[float] = field(default_factory=list)

    @property
    def answered(self) -> int:
        return self.total - self.undecided

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def accuracy_when_answered(self) -> float:
        return self.correct / self.answered if self.answered else 0.0

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
    latencies_ms: list[float]  # per call
    total_ms: float            # wall-clock time for all cases (excludes warm-up)
    misses: list[Miss]                    # wrong answers (undecided ones aren't misses)
    answers: list[dict[str, Answer]]      # per case, per question
    decisions: list[dict[str, Decision]]  # per case, per question


def decide(question: Question, answer: Answer) -> Decision:
    """Reduce an answer to a discrete decision (Score: round half up to a level index)."""
    if answer.value is None:
        return None
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
    begin = time.perf_counter()
    for case in cases:
        start = time.perf_counter()
        answers.append(provider.ask(case.state, questions))
        latencies.append((time.perf_counter() - start) * 1000)
    total = (time.perf_counter() - begin) * 1000
    return score(questions, cases, answers, latencies, total)


async def run_async(provider: AsyncProvider, questions: Mapping[str, Question],
                    cases: Sequence[Case], warmup: int = 1, concurrency: int = 4) -> Report:
    """Like ``run``, but with up to ``concurrency`` calls in flight at once.

    Per-call latency is measured from when a call gets a concurrency slot. It still
    includes any queueing inside the provider (``AsyncLaya`` runs one call at a time),
    so under concurrency it measures what a caller waits, not model speed.
    ``total_ms`` shows the throughput gain, if any.
    """
    for case in cases[:warmup]:
        await provider.ask(case.state, questions)

    slots = asyncio.Semaphore(concurrency)

    async def one(case: Case) -> tuple[dict[str, Answer], float]:
        async with slots:
            start = time.perf_counter()
            answers = await provider.ask(case.state, questions)
            return answers, (time.perf_counter() - start) * 1000

    begin = time.perf_counter()
    results = await asyncio.gather(*(one(case) for case in cases))  # keeps case order
    total = (time.perf_counter() - begin) * 1000
    return score(questions, cases, [a for a, _ in results], [ms for _, ms in results], total)


def score(questions: Mapping[str, Question], cases: Sequence[Case],
          answers: Sequence[Mapping[str, Answer]], latencies_ms: list[float],
          total_ms: float) -> Report:
    """Score already-collected answers (one dict of answers per case) against the labels."""
    stats = {name: QuestionStats() for name in questions}
    misses: list[Miss] = []
    decisions: list[dict[str, Decision]] = []
    for i, (case, case_answers) in enumerate(zip(cases, answers, strict=True)):
        decisions.append({name: decide(q, case_answers[name]) for name, q in questions.items()})
        for name, label in case.expected.items():
            answer, s = case_answers[name], stats[name]
            decision = decisions[-1][name]
            s.total += 1
            s.p_correct.append(answer.probabilities[_key(questions[name], label)])
            if decision is None:
                s.undecided += 1
            elif decision == label:
                s.correct += 1
            else:
                misses.append(Miss(i, name, label, answer))
    return Report(stats, latencies_ms, total_ms, misses,
                  [dict(a) for a in answers], decisions)


def at_min_confidence(report: Report, questions: Mapping[str, Question], cases: Sequence[Case],
                      min_confidence: float) -> Report:
    """Re-score ``report`` as if every question had ``min_confidence``, without asking again.

    Answers whose confidence is below it become undecided. It can only make answers
    stricter: answers that were already undecided stay undecided.
    """
    stricter = [{name: replace(a, value=None) if a.confidence < min_confidence else a
                 for name, a in case_answers.items()} for case_answers in report.answers]
    return score(questions, cases, stricter, report.latencies_ms, report.total_ms)


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
