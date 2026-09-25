"""Escalation: local Laya first, hosted TypeSafe only for what Laya wasn't sure of.

Run:  uv run --env-file .env --extra typesafe --extra laya python benchmarks/run_cascade.py

For each setup, per question: how many answers were right, how many ended undecided,
and how many were escalated to TypeSafe; plus how many TypeSafe calls were made.
Questions keep min_confidence 0 (always answer); escalate_below alone decides escalation.
Escalation is only as good as Laya's confidence: if Laya is confidently wrong, the
answer never reaches TypeSafe.
"""

import laya
from typesafe_sdk import TypeSafeClient

from semantic_operators.bench import Report, run
from semantic_operators.cascade import cascade
from semantic_operators.providers.laya import Laya
from semantic_operators.providers.typesafe import TypeSafe
from support_tickets import cases, questions


def show(label: str, report: Report) -> None:
    escalated = {name: sum(a[name].call.provider == "TypeSafe" for a in report.answers)
                 for name in questions}
    typesafe_calls = sum(any(a.call.provider == "TypeSafe" for a in case.values())
                         for case in report.answers)
    cells = "".join(f"{f'{s.correct:>2} right {s.undecided} unsure {escalated[n]:>2} up':>26}"
                    for n, s in report.questions.items())
    print(f"  {label:<24}{cells}{typesafe_calls:>8}/{len(cases)}")


local = Laya(laya.load("convaiinnovations/laya"))
with TypeSafeClient() as client:
    hosted = TypeSafe(client)
    print(f"\n  {'setup':<24}" + "".join(f"{n:>26}" for n in questions) + "  TypeSafe calls")
    show("Laya alone", run(local, questions, cases))
    show("TypeSafe alone", run(hosted, questions, cases, warmup=0))
    for threshold in (0.6, 0.8, 0.9):
        show(f"cascade, below {threshold}",
             run(cascade(local, hosted, escalate_below=threshold), questions, cases, warmup=0))
