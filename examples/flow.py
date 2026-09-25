"""A flow: ticket triage as a plain Python function, with a second step only when needed.

Run:  uv run --env-file .env --extra typesafe python examples/flow.py
      uv run --extra laya python examples/flow.py --laya      (same code, local model)
"""

import sys

from semantic_operators import Boolean, Choice, Provider, Score
from semantic_operators.flows import Undecided, flow
from semantic_operators.operators import Operator

department = Operator("department", Choice(
    "Which team should handle this message?",
    {
        "billing": "Charges, invoices, payments, refunds, or pricing.",
        "technical": "Bugs, errors, outages, or difficulty using the product.",
        "account": "Logging in, passwords, profile details, or account access and closure.",
        "other": "Anything else, such as feedback, partnerships, or general questions.",
    },
    min_confidence=0.8,
))
urgency = Operator("urgency", Score(
    "How urgently does this need a response?",
    ["low: can wait days", "medium: should be handled today", "high: needs attention now"],
))
outage = Operator("outage", Boolean(
    "Is this an outage: the whole product or site down for the customer, not one feature?",
    true="The product or site is down or won't load at all.",
    false="Something specific is broken, but the product as a whole is up.",
))


@flow
def triage(ask, message):
    a = ask(message, department, urgency)          # one call for both
    try:
        team = a["department"]
    except Undecided:
        return "human"                              # not sure which team: a person decides
    if team == "technical":
        if ask(message, outage)["outage"]:         # second call, technical tickets only
            return "technical: page on-call"
    return f"{team}: {'urgent' if a['urgency'] >= 1.5 else 'normal'}"


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
    "Our whole site has been down for an hour and our store can't take any orders!",
    "The export button does nothing when I click it. Tried Chrome and Safari.",
    "Just wanted to say the new dashboard is fantastic. Great work, team!",
    "Can you tell me more about what you do?",
]

provider = make_provider()
for message in messages:
    result = triage(provider, message)
    print(f"{result.value:<26} {result.calls} call(s)  {message[:55]}")
