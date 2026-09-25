# Benchmarking

The point of one interface over many System One models is being able to compare them
honestly. `semantic_operators.bench` holds the tools; `benchmarks/` holds a small suite
and the scripts that produced the results below.

## Tools

| Function | What it measures |
|---|---|
| `run(provider, questions, cases)` | Per question: accuracy, how many were answered (not "don't know"), and p(correct), the average probability given to the right answer. Plus latency and every miss. |
| `run_async(provider, questions, cases, concurrency=8)` | The same, with calls in flight at once. `total_ms` shows the speed-up. |
| `at_min_confidence(report, questions, cases, 0.8)` | Re-scores a run as if every question had that `min_confidence`, without asking again: answered fewer vs right more often. |
| `stability(reports)` | Runs of differently worded versions of the same questions: how often the decision stays the same. Labels play no part. |
| `run_flow(flow, provider, [(input, expected), ...])` | A whole flow: right, wrong, stopped at "don't know", and provider calls made. |
| `ndcg(order, grades)` | An ordering (e.g. a reranked list) against graded relevance labels, 0 to 1. |

A `Case(state, expected)` labels one input: a bool for Boolean, an option name for
Choice, a level index (0, 1, ...) for Score, so labels survive rewording a rubric.
`operators.questions([...])` turns operators into the dict `run` takes.

Accuracy counts a "don't know" as not correct, but not as a miss. Score answers are
rounded to the nearest level (half up) before comparing.

## The support suite

`benchmarks/support_tickets.py`: 20 hand-written support messages, each labeled for
three questions (is it a complaint, which department, how urgent), with the questions
in three wordings. The messages and labels are authored by one person, and urgency is
a judgment call. **It's a smoke test, not a verdict.** Your own labeled data is the only
thing that tells you which model to use for your task.

```sh
uv run --env-file .env --extra typesafe --extra laya python benchmarks/run.py
```

## What we measured

TypeSafe `jev-1.13.0` (via `jev-latest`) and Laya's English checkpoint (`laya` 0.3.20),
September 2026. Small numbers, so read them as directions, not rankings.

**Wording matters, for both models** (`run.py`). Right answers out of 20, per wording
(descriptive / plain / reworded), and how often the decision stayed the same across all
three:

| | TypeSafe | stable | Laya | stable |
|---|---|---|---|---|
| is_complaint | 20 / 20 / 20 | 100% | 15 / 17 / 16 | 90% |
| department | 18 / 14 / 17 | 75% | 17 / 11 / 11 | 40% |
| urgency | 17 / 19 / 11 | 60% | 6 / 7 / 6 | 40% |

Record the wording along with any result: it's part of the operator.

**TypeSafe's confidence means something; Laya's urgency confidence doesn't**
(`run_confidence.py`). At `min_confidence=0.8`, TypeSafe answered about half the urgency
questions and got every one it answered right (9 of 9 and 10 of 10 in two runs). Laya
answered 7 and got 4 right.

**Concurrency helps a hosted API, not a local model** (`run_async.py`). 20 cases: TypeSafe
went from about 3 s at concurrency 1 to about 0.6 s at 8; Laya stayed about 1.8 s, since
one local model answers one call at a time.

**Escalation works, but this pairing doesn't save calls** (`run_cascade.py`). Laya first,
TypeSafe for what Laya wasn't sure of, right answers out of 20:

| Setup | complaint | department | urgency | TypeSafe calls |
|---|---|---|---|---|
| Laya alone | 15 | 17 | 6 | 0/20 |
| TypeSafe alone | 20 | 18 | 17 | 20/20 |
| cascade, `escalate_below=0.6` | 17 | 18 | 10 | 16/20 |
| cascade, `escalate_below=0.9` | 20 | 18 | 16 | 20/20 |

A message costs a TypeSafe call if *any* of its questions escalates, and Laya was
unsure of something in almost every message (and confidently wrong on others).

**Flows** (`run_flow.py`). Routing each message to (team, priority), stopping below 80%
sure of the team: TypeSafe 18/20 right, 1 stopped, 1 wrong. Laya 2/20 right, 14 stopped,
4 wrong.

**Reranking** (`examples/rerank.py`). Three hand-graded searches, six candidates each,
NDCG@10: retrieval order 0.56; reranked by TypeSafe 1.00, by Laya 0.84. The same person
wrote the documents and the grades, so the relevant ones are probably easier to spot
than in real search results.

**A known issue in Laya's English checkpoint:** it ships a broken calibration value for
Choice questions with 11 or more options. The `laya` package warns at load time and
clamps it; treat confidence on those questions as uncalibrated.
