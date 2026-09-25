# Roadmap

Semantic Operators grows in small steps. Each release adds one thing that's needed,
explained, and tested, and the whole library stays small enough to read in one sitting.
Nothing here is a promise of dates.

## Next

- **Flows**, the next step of combining operators (below).
- **Request IDs** in `Call`, once a provider reports them.

Done in 0.8.0: escalation. `cascade(fast, strong, escalate_below=...)` re-asks only
unsure answers of the next provider, in one call. On the support suite, Laya first came
within one answer of TypeSafe alone but still called TypeSafe for nearly every message,
so whether a pairing saves anything is measured, not assumed (`benchmarks/run_cascade.py`).

Done in 0.7.0: timeouts. `with_timeout(provider, seconds)` limits each call of any
async provider and raises `ProviderTimeout` (a `ProviderError` and a `TimeoutError`);
TypeSafe's own SDK timeouts come back the same way. Sync code sets its timeout on the
client, and a local model can't be interrupted mid-prediction.

Done in 0.6.0: reranking (`rerank.py`), scoring each candidate on its own and sorting,
with `bench.ndcg` to compare against retrieval order.

Done in 0.5.0: every answer's `call` records the model the provider reported and the
tokens used. Retries are decided: they belong to the client you build (the TypeSafe
SDK retries by default; `RetryPolicy(max_retries=0)` turns that off).

## Then: combining operators

The name promises operators you can combine. One fact shapes how: a System One model
answers several questions in one pass for about the cost of one, so asking B only
when A says yes rarely saves time or money. Asking everything in one call and deciding
in plain Python is usually cheaper and faster. Combining earns its keep only when the
next step truly depends on the first answer: a different model, different input, or a
question that depends on A's answer. So, in order:

1. **Escalation (done in 0.8.0).** `cascade(fast, strong)` asks the first provider everything
   and re-asks only its undecided answers of the next, in one call. It's a provider
   wrapper like `with_timeout`, so operators, reranking, and benchmarks work unchanged,
   and `answer.call` shows who answered. It escalates when the model is *unsure*, never
   when it *fails* (that would hide an outage). It's only as good as the first model's
   confidence, which the benchmark measures per question.
2. **Flows (0.9.0).** A flow is a plain Python function that asks operators through
   an `ask` handle. The library records a trace (what was asked, which provider
   answered, what came back), benchmarks a whole flow against labeled outcomes, and
   "don't know" stops the flow rather than being guessed past.
3. **Choices supplied at call time:** options that only exist per call ("which of these
   five documents answers the question?").

Not planned for combining: a graph or chain language, `.then()`/`.when()` builders,
automatic splitting of work into minimal calls, or retries inside `cascade`. Python's
`if` is the flow language.

## Later

- **A generic Jev-compatible HTTP provider:** one provider for any endpoint that
  speaks Jev's request format (Laya's own `laya-serve`, OpenRouter, future clones).
- **An LLM provider (Claude)** as the top of a cascade and as a baseline: the same
  questions answered by a reasoning model, to measure when a System One model is worth
  it (accuracy, latency, cost). Fast model first, reasoning model only for what's left.
- **Benchmark suites as data files** (JSONL) with saved reports, so suites can hold
  your own labeled data and runs can be compared over time. Include "must say don't
  know" cases.
- **Batching:** many inputs in one call. Laya's `predict_batch` takes separate states
  with the same questions, which is exactly reranking's shape (one relevance question,
  many documents, each still in its own state). TypeSafe has no equivalent; putting
  every document into one shared state changes what the model sees, so that would be
  an experiment, not an optimization.
- **Caching** answers for the same input and operator.
- **Provider limits and a conformance test kit**, once there are enough providers
  with different limits (for example, TypeSafe allows at most 10 score levels).

## Not planned (for now)

- **`Rank`** as a question type, until a provider supports ranking natively. Faking it
  with Choice probabilities would misrepresent what the model did. Reranking is a
  separate recipe (score each candidate, then sort), not a `Rank` answer.
- **"Don't know" reasons** beyond "not sure enough": no current model reports others.
- **Observations, sensors, and time series:** they belong in a layer built on top,
  not in the model abstraction.
- **Heavy provenance machinery** (canonical JSON, request fingerprints, evidence
  pointers) until results are stored or cached and it's actually needed.
