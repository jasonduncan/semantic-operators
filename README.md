# Semantic Operators

```sh
pip install "semantic-operators[typesafe]"
```

**Write a semantic judgment once, run it on any System One model, and get a "don't
know" instead of a guess when the model isn't sure.**

System One models are small, fast models that classify instead of generating text,
as opposed to a chat LLM. You ask them typed questions (yes/no, pick one, rate on a
scale) about text or data, and they return answers with probabilities.
[Jev](https://typesafe.ai) was the first; [Laya](https://huggingface.co/convaiinnovations/laya)
is an open-weight, Jev-compatible alternative you can run locally. More are coming.

| Provider | Models | Where it runs | Install |
|----------|--------|---------------|---------|
| `providers.typesafe.TypeSafe` | Jev (`jev-latest`, or pin a version) | TypeSafe's hosted API (needs `TYPESAFE_API_KEY`) | `[typesafe]` |
| `providers.laya.Laya` | Laya checkpoints (English, multilingual, typed-decisions) | on your machine (~800 MB download on first use) | `[laya]` |

A provider is the service or runtime you talk to; the model is a setting.

**Contents:** [Install](#install) · [Quick start](#quick-start) ·
[Questions and answers](#questions-and-answers) · [Operators](#operators) ·
[Flows](#flows) · [Reranking](#reranking) · [Escalation](#escalation) ·
[Command line and MCP](#command-line-and-mcp) · [Errors and timeouts](#errors-and-timeouts) ·
[Async](#async) · [Benchmarking](#benchmarking) · [Adding a provider](#adding-a-provider) ·
[How it's built](#how-its-built)

More in [`docs/`](https://github.com/jasonduncan/semantic-operators/tree/main/docs): [command line and MCP](https://github.com/jasonduncan/semantic-operators/blob/main/docs/cli-and-mcp.md),
[benchmarking and results](https://github.com/jasonduncan/semantic-operators/blob/main/docs/benchmarks.md), [writing a provider](https://github.com/jasonduncan/semantic-operators/blob/main/docs/writing-a-provider.md).

## Install

```sh
pip install "semantic-operators[typesafe]"        # TypeSafe (hosted Jev)
pip install "semantic-operators[laya]"            # Laya (local; pulls in torch)
pip install "semantic-operators[typesafe,laya]"   # both
uv tool install "semantic-operators[typesafe,mcp]" # the semop command and MCP server
```

The core alone (`pip install semantic-operators`) has no dependencies. Python 3.11+.

## Quick start

```python
from typesafe_sdk import TypeSafeClient
from semantic_operators import Boolean, Choice, Score
from semantic_operators.providers.typesafe import TypeSafe

with TypeSafeClient() as client:          # you create and own the client
    provider = TypeSafe(client)           # model defaults to "jev-latest"
    answers = provider.ask(
        "I was charged twice and I'm furious.",
        {
            "is_complaint": Boolean("Is the customer complaining?"),
            "department": Choice("Which team should handle this?",
                                 {"billing": "Payments, refunds", "other": "Anything else"}),
            "urgency": Score("How urgent is this?", ["low", "medium", "high"]),
        },
    )

answers["department"].value           # "billing"
answers["department"].probabilities   # {"billing": 0.97, "other": 0.03}
```

Swapping to Laya changes only how the provider is built:

```python
import laya
from semantic_operators.providers.laya import Laya

provider = Laya(laya.load("convaiinnovations/laya"))   # or Laya(laya.Router())
answers = provider.ask(state, questions)                # same questions, same answers
```

`examples/` has runnable versions (`hello.py`, `compare.py`, `triage.py`, `flow.py`,
`rerank.py`), e.g. `uv run --env-file .env --extra typesafe python examples/hello.py`.

## Questions and answers

A System One model is asked **named questions about a piece of state** (text, a JSON
object, or an array) and answers each with probabilities. There are three kinds:

| Question  | You give it                                   | `answer.value`                        |
|-----------|-----------------------------------------------|---------------------------------------|
| `Boolean` | instructions (+ optional `true`/`false` meanings) | `True` / `False`                  |
| `Choice`  | instructions + named options (at least 2)     | the chosen option name                |
| `Score`   | instructions + ordered levels, lowest first   | expected level index, e.g. `1.7` of 0..2 |

Every `Answer` has:

- `value`, and `decided` (whether there is a value; see below)
- `probabilities`: a dict in the question's option/level order
- `confidence`: the probability of its own answer (for Score, of the nearest level)
- `call`: which provider and model answered, and the tokens used
- `raw`: the provider's own answer object

Questions check themselves when created (a bad one raises `ValueError`), and every
answer is checked before you see it: a malformed response from a model raises
`ProviderError`, never a confident-looking answer.

**"Don't know."** A model that's split, or not sure enough, should say so rather than
guess. Give any question a `min_confidence`; below it the answer comes back undecided
(`value is None`, `decided is False`), with its probabilities kept. An exact tie is
always undecided.

```python
department = Choice("Which team?", {"billing": None, "technical": None}, min_confidence=0.8)
answer = provider.ask(message, {"department": department})["department"]
if answer.decided:
    route(answer.value)
else:
    send_to_a_human(answer.probabilities)   # still shows what it was leaning toward
```

Confidence is the model's own view, not a guarantee: check it with a benchmark.

**Which model answered.** Aliases like `jev-latest` move, so every answer records what
its call reported: `answers["department"].call` is
`Call(provider='TypeSafe', model='jev-1.13.0', input_tokens=291, output_tokens=20)`.
All answers from one call share it; anything unreported is `None`.

## Operators

An **operator** is a semantic judgment defined once, with a name, and used anywhere:

```python
from semantic_operators.operators import Operator, apply

is_complaint = Operator("is_complaint", Boolean("Is the customer complaining?"))
urgency = Operator("urgency", Score("How urgent is this?", ["low", "medium", "high"]))

is_complaint(provider, message).value                        # one operator, one call
answers = apply(provider, message, [is_complaint, urgency])  # several, still one call
```

- System One models answer many questions in one pass, so `apply` asks any set of
  operators in one call. Names must be unique.
- An operator doesn't hold a provider, so the same operator runs on any provider.
- The wording is part of the operator: it changes the answers. Keep operators in code
  and benchmark them as written.

## Flows

A flow is a decision built from operators, written as a plain Python function. Python's
`if` is the flow language:

```python
from semantic_operators.flows import Undecided, flow

@flow
def triage(ask, message):
    a = ask(message, department, urgency)          # one call for both operators
    try:
        team = a["department"]
    except Undecided:
        return "human"                              # not sure which team: a person decides
    if team == "technical" and ask(message, outage)["outage"]:   # second call, only if needed
        return "page on-call"
    return team

result = triage(provider, message)
result.value, result.decided, result.stopped_at, result.calls, result.trace
```

- `ask(state, *operators)` asks them in one call and returns their values by name;
  `answers.full[name]` is the whole `Answer`.
- **"Don't know" stops the flow when the flow uses it.** Reading an undecided answer
  raises `Undecided`; unless caught, the flow ends with `decided=False` and `stopped_at`
  naming the operator. An undecided answer the flow never reads stops nothing.
- `result.trace` records every call: the state and the full answers.
- Each `ask` is a provider call: ask everything you might need up front, and add a
  later step only when it depends on an earlier answer.

## Reranking

Your retrieval (search, a vector index, a database) finds candidates; a System One model
reorders them by relevance. Each candidate is scored on its own against a relevance
rubric, one call each, and plain code sorts them:

```python
from semantic_operators.rerank import rerank, reweighted

relevance = Score("How useful is the document for answering the query?",
                  ["no useful information", "on topic but doesn't answer", "partly answers",
                   "answers with minor gaps", "fully answers"])

ranking = rerank(provider, relevance, query="How do I reset my password?",
                 candidates={"doc-1": {"title": ..., "text": ...},     # in retrieval order
                             "doc-2": {"title": ..., "text": ...}})
for r in ranking.top(10):
    r.id, r.score, r.answer.probabilities
```

- Each call sees only `{"query", "document"}` (plus `"context"` if you pass one), so a
  score doesn't depend on the other candidates.
- Ranked by expected rubric level, not confidence; ties keep retrieval order; scores
  aren't normalized across documents (all can be relevant, or none).
- Undecided or failed candidates go to `ranking.unscored`, never a made-up score.
  `top()` refuses a partial ranking unless `allow_partial=True`, and refuses scores from
  more than one model (`ranking.models`).
- `reweighted(ranking, [0, 10, 40, 80, 100])` re-sorts by your own level weights from
  the probabilities already returned. `rerank_async(..., concurrency=8)` runs calls in
  parallel. `bench.ndcg(order, grades)` checks an ordering against graded labels.

## Escalation

Ask a fast or local model first, and a stronger one only about what it wasn't sure of:

```python
from semantic_operators.cascade import cascade

provider = cascade(Laya(model), TypeSafe(client), escalate_below=0.8)   # still a provider
answers = provider.ask(message, questions)
answers["department"].call.provider    # "Laya", or "TypeSafe" if it was escalated
```

- Answers with confidence under `escalate_below` (or undecided) are re-asked of the next
  provider, all in one call. The last provider's answer stands.
- `escalate_below` decides when to ask a stronger model; a question's `min_confidence`
  still decides when the final answer is a "don't know".
- It escalates when a model is unsure, never when it fails: errors propagate.
- It's only as good as the first model's confidence, and a message costs a call to the
  stronger model if any of its questions escalates. Measure a pairing before relying
  on it ([results](https://github.com/jasonduncan/semantic-operators/blob/main/docs/benchmarks.md)).

## Command line and MCP

```sh
echo '{"state": "The payment failed and now I cannot sign in.",
       "questions": [{"name": "department", "type": "choice", "instructions": "Which team?",
                      "options": ["billing", "technical"], "min_confidence": 0.8}]}' \
  | semop ask --provider typesafe --pretty
```

`semop ask` prints each answer's `value` (`null` when undecided), `decided`,
`confidence`, and `probabilities`, plus the `call`. Exit code 0 means answered
(including "don't know"), 1 the provider failed or timed out, 2 the request was invalid.

`semop mcp` serves the same request and response as an MCP tool, `ask`. Register it
with Claude Code:

```sh
claude mcp add semop -e TYPESAFE_API_KEY="$TYPESAFE_API_KEY" -- semop mcp --provider typesafe
```

The provider is fixed when the server starts: nothing an agent sends can change it.
The full request/response reference, options, and agent guidance are in
[docs/cli-and-mcp.md](https://github.com/jasonduncan/semantic-operators/blob/main/docs/cli-and-mcp.md).

## Errors and timeouts

Every provider raises one error type, whatever went wrong underneath (network, bad key,
rate limit, model failure, malformed response):

```python
from semantic_operators import ProviderError, ProviderTimeout, with_timeout

try:
    answers = provider.ask(state, questions)
except ProviderError as error:
    error.provider, error.__cause__     # "TypeSafe", and the original exception
```

- `with_timeout(async_provider, 2.0)` limits each call of any async provider and raises
  `ProviderTimeout` (a `ProviderError` and a `TimeoutError`). It's still a provider, so
  the limit carries through operators, flows, and reranking.
- Sync code sets its timeout on the client (`TypeSafeClient(timeout=5)`). A local model
  can't be interrupted mid-prediction, and a hosted call may still finish on the server.
- Retries belong to the client you build. The TypeSafe SDK retries by default;
  `TypeSafeClient(retry=RetryPolicy(max_retries=0))` turns that off.

## Async

Every provider has an async twin with the same contract, `await provider.ask(...)`:
`AsyncTypeSafe(AsyncTypeSafeClient())` and `AsyncLaya(model)`. So do the higher layers:
`apply_async`, `op.call_async`, `rerank_async`, `cascade_async`, and async flows
(`async def`, then `await flow.call_async(provider, ...)`).

`AsyncLaya` runs the local model in a worker thread, one call at a time, so concurrency
speeds up a hosted API but not a single local model.

## Benchmarking

```sh
uv run --env-file .env --extra typesafe --extra laya python benchmarks/run.py
```

`semantic_operators.bench` scores providers against labeled cases: accuracy, how often
they said "don't know", the probability they gave the right answer, latency, and how
stable their answers are when only the wording changes. It also scores whole flows
(`run_flow`) and rankings (`ndcg`). `benchmarks/` holds a 20-message support suite and
the scripts. The tools and everything measured so far are in
[docs/benchmarks.md](https://github.com/jasonduncan/semantic-operators/blob/main/docs/benchmarks.md); in short, both models are sensitive to
wording, and TypeSafe's confidence tracks correctness far better than Laya's.

## Adding a provider

A provider is anything with one method:

```python
def ask(self, state, questions: Mapping[str, Question]) -> dict[str, Answer]: ...
```

Translate the questions into the model's API, make one call, build each answer with
`make_answer` (which enforces the rules above), and raise `ProviderError` for any
failure. Everything else (operators, flows, reranking, benchmarks, `semop`) then works
with it unchanged. A skeleton and the rules are in
[docs/writing-a-provider.md](https://github.com/jasonduncan/semantic-operators/blob/main/docs/writing-a-provider.md).

## How it's built

```
src/semantic_operators/
  types.py, provider.py, errors.py   base: questions, answers, the provider protocol
  providers/typesafe.py, laya.py     base: one translation file per provider
  operators.py, flows.py             named operators and flows
  rerank.py, cascade.py              reranking and escalation
  bench.py                           benchmarking
  interfaces/                        the semop CLI and MCP server
examples/  benchmarks/  tests/  docs/  ROADMAP.md
```

- **Layers:** the base layer never imports anything above it; the higher layers build
  only on it; `interfaces/` is an application that uses the library like your own code
  would, and nothing in the library imports it.
- **The library never reads API keys or environment variables.** You build the client.
  (`semop` does read them, since it builds the client for you.)
- **The core has no dependencies.** Each provider's SDK is an optional extra.
- **Our names, not a provider's:** `Boolean`, not `noul`.

Run the offline tests with `uv run --extra typesafe --extra mcp pytest`; CI runs them on
Python 3.11 and 3.13 before every release. Next steps are in [ROADMAP.md](https://github.com/jasonduncan/semantic-operators/blob/main/ROADMAP.md).

## License

MIT
