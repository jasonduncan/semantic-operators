"""Run the support-ticket benchmark through Jev and Laya.

Run:  uv run --env-file .env --extra jev --extra laya python benchmarks/run.py
Makes 21 Jev API calls (1 warm-up + 20 cases).
"""

import statistics

import laya
from typesafe_sdk import TypeSafeClient

from semantic_operators.bench import Report, run
from semantic_operators.providers.jev import Jev
from semantic_operators.providers.laya import Laya
from support_tickets import cases, questions


def show(name: str, report: Report) -> None:
    ms = sorted(report.latencies_ms)
    print(f"\n{name}: median {statistics.median(ms):.0f} ms, slowest {ms[-1]:.0f} ms per call")
    print(f"  {'question':<14}{'accuracy':>10}{'p(correct)':>12}")
    for question, s in report.questions.items():
        print(f"  {question:<14}{s.correct:>5}/{s.total:<4}{s.mean_p_correct:>12.2f}")
    print("  misses:")
    for m in report.misses:
        got = m.got.value
        got = f"{got:.2f}" if isinstance(got, float) else got
        print(f"    case {m.case:>2} {m.question:<13} expected {m.expected!s:<32} got {got}")


with TypeSafeClient() as client:
    show("Jev", run(Jev(client), questions, cases))

show("Laya", run(Laya(laya.load("convaiinnovations/laya")), questions, cases))
