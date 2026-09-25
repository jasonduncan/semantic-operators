"""A whole flow against labeled outcomes: route each support message to (team, priority).

Run:  uv run --env-file .env --extra typesafe --extra laya python benchmarks/run_flow.py

The flow stops (a "don't know") when it isn't at least 80% sure of the team. Expected
outcomes come from the suite's labels: the department, and "urgent" for high urgency.
"""

from dataclasses import replace

import laya
from typesafe_sdk import TypeSafeClient

from semantic_operators import Provider
from semantic_operators.bench import run_flow
from semantic_operators.flows import flow
from semantic_operators.operators import Operator
from semantic_operators.providers.laya import Laya
from semantic_operators.providers.typesafe import TypeSafe
from support_tickets import HIGH, cases, questions

department = Operator("department", replace(questions["department"], min_confidence=0.8))
urgency = Operator("urgency", questions["urgency"])


@flow
def route(ask, message):
    a = ask(message, department, urgency)
    return a["department"], "urgent" if a["urgency"] >= 1.5 else "normal"


labeled = [(case.state, (case.expected["department"],
                         "urgent" if case.expected["urgency"] == HIGH else "normal"))
           for case in cases]


def show(name: str, provider: Provider) -> None:
    report = run_flow(route, provider, labeled)
    print(f"\n{name}: {report.correct}/{report.total} right, {report.stopped} stopped "
          f"(not sure of the team), {len(report.misses)} wrong; "
          f"{report.accuracy_when_answered:.0%} right when it answered; {report.calls} calls")
    for i, expected, got in report.misses:
        print(f"  case {i:>2}: expected {expected}, got {got}")


with TypeSafeClient() as client:
    show("TypeSafe (jev-latest)", TypeSafe(client))
show("Laya", Laya(laya.load("convaiinnovations/laya")))
