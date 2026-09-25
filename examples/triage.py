"""Named operators in use: ticket triage built from three operators, one call per ticket.

Run:  uv run --env-file .env --extra typesafe python examples/triage.py
      uv run --extra laya python examples/triage.py --laya      (same code, local model)
"""

import sys
from dataclasses import dataclass

from semantic_operators import Boolean, Choice, Provider, Score
from semantic_operators.operators import Operator, apply

# Defined once. In a real project these live in their own module and get imported.
is_complaint = Operator("is_complaint", Boolean(
    "Is the customer complaining or expressing dissatisfaction?"))

department = Operator("department", Choice(
    "Which team should handle this message?",
    {
        "billing": "Charges, invoices, payments, refunds, or pricing.",
        "technical": "Bugs, errors, outages, or difficulty using the product.",
        "account": "Logging in, passwords, profile details, or account access and closure.",
        "other": "Anything else, such as feedback, partnerships, or general questions.",
    },
    min_confidence=0.8,  # not sure enough -> a person decides the routing
))

urgency = Operator("urgency", Score(
    "How urgently does this need a response?",
    ["low: can wait days", "medium: should be handled today", "high: needs attention now"],
))


@dataclass
class Triage:
    queue: str      # a department, or "human" when the model wasn't sure
    priority: str   # "urgent" or "normal"
    note: str


def triage(provider: Provider, message: str) -> Triage:
    answers = apply(provider, message, [is_complaint, department, urgency])  # one call
    complaint, dept, urgent = (answers[op.name] for op in (is_complaint, department, urgency))

    queue = dept.value if dept.decided else "human"
    priority = "urgent" if urgent.value >= 1.5 or (complaint.value and urgent.value >= 1.0) else "normal"
    leaning = max(dept.probabilities, key=dept.probabilities.get)
    note = (f"department {dept.confidence:.0%} sure" if dept.decided
            else f"unsure about department (leaning {leaning}, {dept.confidence:.0%})")
    return Triage(queue, priority, note)


def make_provider() -> Provider:
    if "--laya" in sys.argv:
        import laya
        from semantic_operators.providers.laya import Laya
        return Laya(laya.load("convaiinnovations/laya"))

    from typesafe_sdk import TypeSafeClient
    from semantic_operators.providers.typesafe import TypeSafe
    return TypeSafe(TypeSafeClient())  # reads TYPESAFE_API_KEY; the library itself never does


messages = [
    "I was charged twice for my subscription this month. Please refund one of them.",
    "Our CEO is presenting from your platform in 20 minutes and nothing loads.",
    "Just wanted to say the new dashboard is fantastic. Great work, team!",
    "SSO login is failing for all 500 of our users since this morning.",
    "Can you tell me more about what you do?",
]

provider = make_provider()
for message in messages:
    t = triage(provider, message)
    print(f"{t.queue:>9} | {t.priority:<6} | {t.note:<48} | {message[:50]}")
