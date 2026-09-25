# Roadmap

Semantic Operators grows in small steps. Each release adds one thing that's needed,
explained, and tested, and the whole library stays small enough to read in one sitting.
Nothing here is a promise of dates.

## Next

- **Which model answered, and what it cost.** `jev-latest` moves over time, so every
  call should record the model version the provider actually used, plus token usage and
  request ID where the provider reports them. Needed to trust benchmark results later.
- **Call timeouts.** A `timeout=` on `ask`, operators, and `apply`.
- **A decision on retries.** The TypeSafe provider currently inherits the SDK's default
  retries: convenient in apps, but they hide failures in benchmarks. Make it explicit.

## Then: combining operators

The name promises operators you can combine, and this is where it's headed:

- **Conditions:** ask B only when A says yes (or is confident enough).
- **Chains and small decision flows** built from operators, still one provider call
  per step, and "don't know" carried through rather than guessed past.
- **Choices supplied at call time:** options that only exist per call ("which of these
  five documents answers the question?").

## Later

- **A generic Jev-compatible HTTP provider:** one provider for any endpoint that
  speaks Jev's request format (Laya's own `laya-serve`, OpenRouter, future clones).
- **An LLM provider as a baseline:** the same questions answered by a chat model,
  to measure when a System One model is worth it (accuracy, latency, cost).
- **Benchmark suites as data files** (JSONL) with saved reports, so suites can hold
  your own labeled data and runs can be compared over time. Include "must say don't
  know" cases.
- **Batching:** many inputs in one call (Laya's `predict_batch`).
- **Caching** answers for the same input and operator.
- **Provider limits and a conformance test kit**, once there are enough providers
  with different limits (for example, TypeSafe allows at most 10 score levels).

## Not planned (for now)

- **`Rank`** as a question type, until a provider supports ranking natively. Faking it
  with Choice probabilities would misrepresent what the model did.
- **"Don't know" reasons** beyond "not sure enough": no current model reports others.
- **Observations, sensors, and time series:** they belong in a layer built on top,
  not in the model abstraction.
- **Heavy provenance machinery** (canonical JSON, request fingerprints, evidence
  pointers) until results are stored or cached and it's actually needed.
