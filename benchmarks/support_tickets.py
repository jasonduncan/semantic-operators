"""Support-ticket triage: 20 hand-written messages, labeled by hand.

These are authored examples, not real customer data, and one person's labels.
"Urgency" in particular is a judgment call. Treat results as a smoke test of
the providers, not a verdict on them.
"""

from semantic_operators import Boolean, Choice, Score
from semantic_operators.bench import Case

questions = {
    "is_complaint": Boolean("Is the customer complaining or expressing dissatisfaction?"),
    "department": Choice(
        "Which team should handle this message?",
        {
            "billing": "Charges, invoices, payments, refunds, or pricing.",
            "technical": "Bugs, errors, outages, or difficulty using the product.",
            "account": "Logging in, passwords, profile details, or account access and closure.",
            "other": "Anything else, such as feedback, partnerships, or general questions.",
        },
    ),
    "urgency": Score(
        "How urgently does this need a response?",
        ["low: can wait days", "medium: should be handled today", "high: needs attention now"],
    ),
}

LOW, MEDIUM, HIGH = questions["urgency"].levels


def case(state: str, complaint: bool, department: str, urgency: str) -> Case:
    return Case(state, {"is_complaint": complaint, "department": department, "urgency": urgency})


cases = [
    # Clear-cut
    case("I was charged twice for my subscription this month. Please refund one of them.",
         True, "billing", MEDIUM),
    case("Your whole site has been down for an hour and our store can't take any orders!",
         True, "technical", HIGH),
    case("How do I change the email address on my profile?",
         False, "account", LOW),
    case("Can you send me a copy of last month's invoice for our accountant?",
         False, "billing", LOW),
    case("The export button does nothing when I click it. Tried Chrome and Safari.",
         True, "technical", MEDIUM),
    case("I think someone else logged into my account, I see orders I never placed. Help!",
         True, "account", HIGH),
    case("Just wanted to say the new dashboard is fantastic. Great work, team!",
         False, "other", LOW),
    case("We're a design agency interested in a partnership. Who should we talk to?",
         False, "other", LOW),
    case("Payment failed three times at checkout and now my card is locked by the bank.",
         True, "billing", HIGH),
    case("I forgot my password and the reset email never arrives.",
         True, "account", MEDIUM),

    # Harder: sarcasm, politeness masking a problem, mixed topics, JSON state
    case("Oh great, another 'minor update' that wiped all my saved reports. Love it.",
         True, "technical", HIGH),
    case("No rush at all, but I noticed the annual plan price on your site doesn't match "
         "what I was billed.",
         True, "billing", LOW),
    case("Please delete my account and all my data. I no longer want to use this service.",
         False, "account", MEDIUM),
    case("Is there a discount for nonprofits?",
         False, "billing", LOW),
    case("The app keeps crashing, and since I can't use it I want a refund for this month.",
         True, "billing", MEDIUM),
    case("Our CEO is presenting from your platform in 20 minutes and nothing loads.",
         True, "technical", HIGH),
    case("Thanks for fixing the login bug so quickly yesterday!",
         False, "other", LOW),
    case("I've emailed three times about my refund and nobody has replied. Unacceptable.",
         True, "billing", HIGH),
    case({"channel": "chat", "plan": "enterprise", "message": "SSO login is failing for all "
          "500 of our users since this morning."},
         True, "account", HIGH),
    case({"channel": "email", "plan": "free", "message": "Where can I find your API docs?"},
         False, "other", LOW),
]
