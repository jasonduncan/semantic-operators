"""Rerank search results by relevance, and check it against the original retrieval order.

Run:  uv run --env-file .env --extra typesafe python examples/rerank.py
      uv run --extra laya python examples/rerank.py --laya      (same code, local model)

Three hand-written queries, six candidates each, listed in a deliberately keyword-ish
"retrieval order", with relevance graded by hand (0 = no use, 1 = related, 2 = answers
it). Authored examples, one person's grades: a smoke test, not a benchmark.
"""

import asyncio
import sys

from semantic_operators import Score
from semantic_operators.bench import ndcg
from semantic_operators.rerank import Ranking, rerank, rerank_async

relevance = Score(
    "How useful is the document for answering the query?",
    [
        "no useful information for the query",
        "on the topic, but doesn't answer the query",
        "partly answers the query",
        "answers the query with minor gaps",
        "fully answers the query",
    ],
)

# query -> [(id, title, text, grade)], in retrieval order
SEARCHES = {
    "How do I reset my password?": [
        ("pw-rules", "Password requirements",
         "Passwords must be at least 12 characters and include a number.", 1),
        ("billing-faq", "Billing FAQ",
         "You can reset your billing cycle date once per year from the Billing page.", 0),
        ("admin-policy", "Password policy for admins",
         "Admins can require members to change passwords every 90 days.", 1),
        ("2fa", "Two-factor authentication",
         "Add an authenticator app under Settings > Security to protect your account.", 0),
        ("recover", "Recovering access to your account",
         "On the sign-in page, click 'Forgot password' and follow the emailed link "
         "to choose a new password.", 2),
        ("change-pw", "Changing your password",
         "While signed in, go to Settings > Security > Change password.", 2),
    ],
    "Why was I charged twice this month?": [
        ("newsletter", "This month's product news",
         "This month we shipped dark mode and faster search.", 0),
        ("payment-method", "Updating your payment method",
         "Replace your card under Billing > Payment methods.", 0),
        ("refunds", "Refund policy",
         "Refunds are available within 30 days of a charge.", 1),
        ("contact", "Contacting billing support",
         "Email billing@example.com with your invoice number.", 1),
        ("pending", "Understanding pending charges",
         "A temporary authorization hold can appear next to the final charge. "
         "The hold disappears within 3 to 5 days.", 2),
        ("proration", "Charges after changing plans",
         "Upgrading mid-cycle bills a prorated amount right away, so you may see two "
         "charges in the same month.", 2),
    ],
    "Can I export my data to CSV?": [
        ("import-csv", "Importing contacts from CSV",
         "Upload a CSV file under Contacts > Import.", 0),
        ("shortcuts", "Keyboard shortcuts",
         "Press ? anywhere to see all keyboard shortcuts.", 0),
        ("retention", "Data retention",
         "We keep deleted data for 30 days before removing it permanently.", 0),
        ("gdpr", "Downloading all your data",
         "Request a full archive of your account data as a ZIP of JSON files.", 1),
        ("reports", "Exporting reports",
         "Open any report and choose Export > CSV or PDF.", 2),
        ("api", "Bulk data API",
         "The /export endpoint returns your records; add format=csv for CSV output.", 2),
    ],
}


def show(query: str, ranking: Ranking) -> tuple[float, float]:
    docs = SEARCHES[query]
    grades = {id: grade for id, _, _, grade in docs}
    before = ndcg([id for id, *_ in docs], grades)
    after = ndcg([r.id for r in ranking.top(6, allow_partial=True)], grades)
    print(f"\n{query}\n  NDCG: retrieval order {before:.2f} -> reranked {after:.2f}"
          + ("" if ranking.complete else f"  ({len(ranking.unscored)} unscored)"))
    titles = {id: title for id, title, _, _ in docs}
    for r in ranking.ranked:
        print(f"    {r.score:4.2f}  grade {grades[r.id]}  {titles[r.id]}")
    return before, after


def candidates(query: str) -> dict[str, dict[str, str]]:
    return {id: {"title": title, "text": text} for id, title, text, _ in SEARCHES[query]}


async def with_typesafe() -> list[tuple[float, float]]:
    from typesafe_sdk import AsyncTypeSafeClient
    from semantic_operators.providers.typesafe import AsyncTypeSafe

    async with AsyncTypeSafeClient() as client:  # reads TYPESAFE_API_KEY; the library doesn't
        provider = AsyncTypeSafe(client)
        return [show(q, await rerank_async(provider, relevance, query=q,
                                           candidates=candidates(q), concurrency=6))
                for q in SEARCHES]


def with_laya() -> list[tuple[float, float]]:
    import laya
    from semantic_operators.providers.laya import Laya

    provider = Laya(laya.load("convaiinnovations/laya"))
    return [show(q, rerank(provider, relevance, query=q, candidates=candidates(q)))
            for q in SEARCHES]


results = with_laya() if "--laya" in sys.argv else asyncio.run(with_typesafe())
print(f"\nMean NDCG: retrieval order {sum(b for b, _ in results) / len(results):.2f} "
      f"-> reranked {sum(a for _, a in results) / len(results):.2f}")
