# `semop filter`: design v0.2

**Command:** `semop filter`, also installed as `semfilter`
**Status:** Built in 0.11.0. This document describes what shipped.
**Date:** 2026-09-25
**Baseline:** `semantic-operators` 0.10.1, provider `typesafe` (the default), model `jev-1.13.0`
**Author:** Jason Duncan, drafted with Claude Code

> Grep by meaning: ask one yes/no question of every item on stdin, and print the ones
> the model says yes to.

```sh
find ~/PDLabsShare/logs -name '*.md' \
  | semfilter --files "Does this log record a release that failed or was blocked?"
```

**Changes from v0.1:** TypeSafe is now `semop`'s default provider, so `--provider` is
optional. `--keep-unsure` prints unsure items with the matches, which is what an agent
pre-filter wants. The per-item machinery is shared with `rerank`. `--context` wraps every
item the same way. `--null` and `--true`/`--false` moved to "Not in v1". Open questions
1, 2, and 5 are settled (§11).

---

## 1. Why

An agent such as Claude Code can't stop in the middle of a thought to call a tool, but
it *can* narrow its input **before** it starts thinking. What costs an agent most isn't
making a simple judgment. It's **reading**: every file it opens to find out whether the
file matters uses up context, time, and attention. A System One model can make that
first "does this matter?" pass over hundreds of items in seconds, without the text ever
reaching the agent. The agent reads only what passed, plus what the model wasn't sure of.

The same command gives a meaning-level filter to anything that runs in a shell with no
model involved: pipelines, cron jobs, `/loop` checks, and hooks. It also gives people
a grep that understands meaning.

`semop ask` answers one request. Filtering many items with it takes a shell loop, `jq`
on both sides, one process per item, and hand-written handling of failures and
undecided answers, and every item pays the process start-up cost. `semop filter` is
that loop, done once and done right.

## 2. Measurements that shaped this design

Measured 2026-09-25 against TypeSafe `jev-1.13.0` from this machine.

| What | Result | Design consequence |
|---|---|---|
| One call inside a running process | median 0.15 s, p90 ≈ 0.3 s | Keep one process for all items |
| One `semop ask` process, end to end | ≈ 0.7 s | Start-up dominates; a shell loop is ~5× slower than it needs to be |
| 48 items, 1 call at a time | 7.9 s (6 items/s) | |
| 48 items, 8 in flight | 1.1 s (42 items/s) | Default `--jobs 8` |
| 48 items, 16 in flight | 0.7 s (73 items/s), no errors or rate limiting | `--jobs` can go higher; limits at scale unknown |
| Input size | 64,000 chars (16,690 input tokens) accepted, no truncation; a codeword at the start *and* the end of the text was found at 0.97–0.99 | Default `--max-chars 32000`, and say when anything is cut |
| 22 true/false probe pairs (facts, dates, logic, fallacies, contract reading, double negatives, sarcasm) | Every answer it decided was correct | Broadly capable on short judgments; 22 probes are a smoke test, not a guarantee |
| Arithmetic and letter counting | Undecided at `min_confidence` 0.8; one wrong answer leaned "true" at 0.68 | Default `--min 0.8`, not the library's 0 |
| Questions the text can't answer ("will it rain tomorrow") | Undecided (0.36, 0.43) | "Unsure" is a first-class outcome |

**After building it** (0.11.0, same model):

- This repo's 24 docs and source files, "Does this file describe or implement the MCP
  server?": 0.9 s. It matched exactly `mcp_server.py` (0.99) and `docs/cli-and-mcp.md`
  (0.92), and flagged the two nearest calls, `cli.py` and `wire.py`, as unsure rather
  than guessing.
- Six made-up download names, "Is this an installer, disk image, or other throwaway
  download?": TypeSafe matched the three installers and was unsure only of
  `wedding-photos.zip`. Laya matched two, was unsure of two, and took 15 s, mostly
  loading the model.

## 3. Where it lives

**In this repository, as a subcommand of the existing `semop`.**

- `semop` is already this repo's application layer, and `filter` needs everything it
  has: the `--provider/--model/--timeout` flags, `backends.open_provider`, and the rule
  that the provider is fixed per process.
- The code:
  - `filtering.py`, a higher-layer library module next to `rerank.py`
  - `interfaces/filter_cli.py`, the command
  - `_each.py`, the per-item machinery (one question per state, one call each,
    calls in parallel, input order kept, `ProviderError` recorded per item), shared
    by `filtering` and `rerank`, so a fix lands in both

**When to split it out:** if this grows into an *agent toolkit* (crawling folders,
caching answers across runs, indexing, watching directories, hook integrations), that's
a product with its own users and release pace. At that point, make it a separate project
that depends on `semantic-operators` and move `semop filter` there. v1 includes none
of that.

## 4. Command

### 4.1 Synopsis

```text
semop filter [options] QUESTION < items
semfilter    [options] QUESTION < items
```

`QUESTION` is the instructions of a Boolean question. Each item is asked separately,
and the items the model answers **yes** to are printed.

### 4.2 Options

| Option | Default | Meaning |
|---|---|---|
| `--provider`, `--model`, `--timeout` | `typesafe`, as `semop ask` | Same flags, same meaning; the provider is fixed per process |
| `--min P` | `0.8` | `min_confidence`: below it an item is **unsure**, never a match |
| `--context TEXT` | none | Shared background sent with every item (a rubric, the query, a style guide) |
| `--files` | off | Each input line is a path; send the file's text |
| `--jsonl` | off | Each input line is a JSON value; send it as the item |
| `--no-path` | off | With `--files`, send only the text, not the path |
| `--max-chars N` | `32000` | Cut each item's text to N characters and report that it was cut |
| `--keep-unsure` | off | Print unsure items along with the matches |
| `-v`, `--invert` | off | Print the confident **no** items instead of the yes items |
| `--records` | off | Print one JSON record per item (every outcome), not matching lines |
| `--jobs N` | `8` | Calls in flight at once |
| `--dry-run` | off | Read the input, show the item count and the first request, send nothing |
| `-q`, `--quiet` | off | Leave out the per-item stderr lines; keep the summary line |

### 4.3 Input

- **Lines (default).** Each non-empty line is one item, and the state is the line text.
  Blank lines are skipped.
- **`--jsonl`.** Each line is parsed as JSON, and the state is the parsed value (object,
  array, or string). A line that doesn't parse is an input error (exit 2) naming the
  line number. No calls are made. JSON items aren't cut.
- **`--files`.** Each line is a path, and the state is `{"path": <as given>, "text": <file text>}`,
  or just the text with `--no-path`. A file that can't be read, isn't UTF-8, or has a NUL
  byte in its first 8 KB is **skipped** with a reason. It isn't sent and it isn't a failure.
- **`--context TEXT`** wraps every item's state as `{"context": TEXT, "item": <state>}`,
  whatever the item is (a line, a JSON value, or a file's `{"path", "text"}`).

**One call per item.** Items are never packed into one request, even though System One
models answer several questions per call. Packing would let items affect each other's
answers. `rerank` follows the same rule, and the two share the code that enforces it.

### 4.4 Output

Every item ends in exactly one outcome: **match**, **no**, **unsure**, **failed**, or
**skipped**.

- **stdout:** matches only (or confident no's with `-v`; plus unsure items with
  `--keep-unsure`), **in input order**, printed exactly as read: the line, the JSON line,
  or the path. This makes it composable: `... | semfilter ... | xargs ...`.
- **stderr:** one line per item that needs a human or agent to look at it, then a summary line:

  ```text
  ? notes/q3-planning.md                     p(true)=0.62
  ! notes/huge-export.md                     TypeSafe: 503 Service Unavailable
  - assets/logo.png                          skipped: binary
  ~ logs/2026-09-24-full-trace.md            cut to 32000 chars
  semop filter: 12 matched · 180 no · 5 unsure · 1 failed · 1 skipped of 199 · sent 198 to typesafe jev-1.13.0 · 3.1s
  ```

- **`--records`:** stdout carries one JSON object per item, in input order, for every
  outcome, including skipped:

  ```json
  {"item": "notes/q3-planning.md", "outcome": "unsure", "p_true": 0.62, "confidence": 0.62,
   "cut": false, "model": "jev-1.13.0", "error": null}
  ```

  This is the form for sorting (`jq -s 'sort_by(-.p_true)'`), audits, and anything
  that needs the probabilities. For a skipped item, `error` is the reason.

Rules:

1. **Unsure never reaches stdout unless you ask** (`--keep-unsure`). It's always listed
   on stderr (unless `-q`) and always counted.
2. **Nothing disappears silently.** Failed and skipped items are listed and counted,
   and failures change the exit code.
3. **The summary always says how many items were sent, and to which provider and
   model.** With TypeSafe as the default, `--files` sends file contents to a hosted
   API, so that must never go unnoticed.
4. If answers came from **more than one model** (possible under the `jev-latest` alias),
   the summary says so. That's the same concern behind `rerank`'s refusal to mix models.

### 4.5 Exit codes

grep-style, so `if semfilter -q ...; then` works:

| Code | Meaning |
|---|---|
| 0 | Every item answered or skipped, and something was printed (a match; a confident no with `-v`; or an unsure item with `--keep-unsure`) |
| 1 | Every item answered or skipped, and nothing was printed |
| 2 | Bad command line or input (bad JSON line, bad `--min`, missing API key); nothing sent |
| 3 | One or more items failed (provider error or timeout); the results for the rest are still printed |

This differs on purpose from `semop ask`, where 1 means the provider failed: a filter
behaves like grep, and `ask` answers one request.

## 5. Semantics

- **The question is the operator.** Its wording changes the answers, so don't reword it
  to get a result you want. To compare wordings, use `bench`. For scripts that get
  reused, keep the question in version control.
- **`--min` defaults to 0.8**, although the library default is 0. A filter's job is to
  keep things out, and the probes show the threshold is what turns weak spots
  (computation, questions with no answer in the text) into *unsure* instead of *wrong*.
  `--min 0` is allowed, but the summary says nothing will ever be unsure.
- **Retries** belong to the client, as they already do: the TypeSafe SDK's defaults apply.
  A failure after retries is a **failed** item. The run continues and doesn't stop.
- **Cutting.** `--max-chars` cuts from the end and marks the item as cut. It's a guard
  against pathological inputs (a 50 MB log), not a statement about the model's limit,
  which was not reached at 64,000 characters.
- **Cost.** Every item is a billed call, and input tokens scale with the text sent. For a
  "does this matter?" pass over many files, the start of a file usually settles it, so a
  smaller `--max-chars` is often enough (§11).
- **Privacy.** With `--files`, file contents go to the provider. `--dry-run` shows exactly
  what would be sent (and needs no API key), and the summary names the provider and
  model. The command doesn't decide what's sensitive. The caller chooses the input.

## 6. Library layer

`semantic_operators.filtering`, built only on the base layer. The CLI uses it the way
any other code would.

```python
@dataclass(frozen=True)
class Verdict:
    id: str
    outcome: Literal["match", "no", "unsure", "failed"]
    answer: Answer | None = None          # None when failed
    error: ProviderError | None = None    # set when failed

@dataclass(frozen=True)
class Filtered:
    question: Boolean
    verdicts: list[Verdict]               # one per item, in input order
    # .matched, .rejected, .unsure, .failed: the verdicts with that outcome, in order
    # .models: every model that answered

def filter_items(provider: Provider, question: Boolean, items: Mapping[str, State], *,
                 context: Any = None) -> Filtered: ...

async def filter_items_async(provider: AsyncProvider, question: Boolean,
                             items: Mapping[str, State], *, context: Any = None,
                             concurrency: int = 8) -> Filtered: ...
```

- `items` maps an id to a state, in input order, the same shape as `rerank`'s
  `candidates`. The id is never sent.
- `question` is a `Boolean`, and its own `min_confidence` decides unsure. A named
  `Operator` passes `op.question`.
- `context` wraps each state as `{"context": context, "item": state}`.
- Reading files, cutting text, skipping binaries, and output formatting belong to the
  CLI, not the library. The library never touches the filesystem.
- `with_timeout` and `cascade` wrap the provider as usual. For example, a cascade makes a
  cheap first pass and asks the stronger model only about the unsure items.

## 7. Examples

```sh
# Which downloads are throwaway? (names only, contents not sent)
ls ~/Downloads | semfilter "Is this an installer, disk image, or other throwaway download?"

# Which session logs record a blocked or failed release?
find ~/PDLabsShare/logs -name '*.md' | semfilter --files \
  "Does this log record a release that failed or was blocked?"

# Which paragraphs break the voice rules? (shared context, JSON items)
jq -c '.[]' paragraphs.json | semfilter --jsonl --context "$(cat voice-rules.md)" \
  "Does this paragraph break any of the voice rules in the context?"

# Everything, with probabilities, most likely first
find docs -name '*.md' | semfilter --files --records \
  "Does this document describe the MCP server?" | jq -s 'sort_by(-.p_true)'

# In a script: act only if something matched
if tail -n 200 app.log | semfilter -q \
     "Does this log line report an error a person should look at?" > flagged.txt; then
  notify "$(wc -l < flagged.txt) log lines need attention"
fi
```

### 7.1 Agent workflow (Claude Code)

The skill that teaches Claude Code to use `semop` gets one rule:

> Before reading more than ~10 files, or a long listing, just to find the relevant
> ones, run `semfilter --files --keep-unsure` (or plain lines) with a direct yes/no
> question. Read what it prints, and treat everything else as not relevant.

`--keep-unsure` is there because, for an agent, missing a relevant file costs more than
reading an extra one.

In a hook, it may **add context or a warning** ("this prompt looks like it touches
Muzology billing"). It must **not** make permission decisions: the `ask` tool's own
contract says it isn't for permissions.

**Why a CLI and not an MCP tool first:** an MCP `filter` tool would need the items as
tool arguments, so the agent would have to read them into its context first, which
defeats the purpose. The CLI reads the files itself. An MCP `filter` for short
lists can come later ([§9](#9-not-in-v1)).

## 8. Tests

Offline, with a fake provider (`tests/test_filtering.py`):

- Every outcome (match, no, unsure, failed, skipped as binary, unreadable, or non-UTF-8)
  lands in the right place, in input order, including when later items answer first
  under concurrency.
- Sync and async library calls agree; item ids are never sent; `--context` wraps each item.
- Unsure stays off stdout unless `--keep-unsure`; `-v` prints confident no's.
- Exit codes 0 / 1 / 2 / 3, including 3 with matches still printed.
- A bad `--jsonl` line exits 2 naming the line, with no calls made.
- `--files` sends `{"path", "text"}` (or just the text with `--no-path`), and `--max-chars`
  cuts and reports.
- `--records` covers every outcome, including skipped.
- `--dry-run` makes zero provider calls and needs no API key; a missing key exits 2 before sending.
- Mixed models are reported in the summary. `semfilter` is `semop filter`.

Live qualification (not in CI): turn the 22 probe pairs from §2 into a small labeled
suite for `bench` so model changes are visible. Then run a 500-item file set at
`--jobs 8` and `--jobs 16` and record throughput and any rate limiting.

## 9. Not in v1

- **`-0`/`--null`** input (NUL-separated, for `find -print0`) and **`--true`/`--false`**
  descriptions of what yes and no mean. Both are small; add them when someone needs them.
- **A flow as the predicate** (`--flow module:fn`): keep an item when a flow decides yes,
  and treat a flow stopped by `Undecided` as unsure. It's the same filter with a
  multi-step decision in place of one question.
- **Classify / tag** (`Choice`: put each item into one of N buckets) and **rank**
  (`Score` + sort). These are the natural next subcommands. `rerank` already has the
  library side of rank, and both would reuse `_each.py`.
- An **MCP `filter` tool** for short, inline lists.
- **Caching** answers across runs (keyed by state, question, and model).
- Walking directories and globbing inside the command. `find` already does that.
- Deduplicating identical items within a run.

## 10. Rollout

1. `filtering.py` + tests. **Done.**
2. `semop filter` and `semfilter` + tests, README section, ROADMAP entry. **Done (0.11.0).**
3. Update the Claude Code `semop` skill with the rule in §7.1. **Done.**
4. Release 0.11.0.

## 11. Open questions

Settled:

1. **Alias and provider default.** `semfilter` is installed alongside `semop`, and
   TypeSafe is the default provider for every `semop` command, so no `SEMOP_PROVIDER`
   variable is needed.
2. **Exit codes.** They stay different from `semop ask` (§4.5).
5. **Path in the state.** Sent by default, since file names carry real signal;
   `--no-path` leaves it out.

Still open:

3. **`--max-chars` default.** 32,000 is safe, but for a relevance pre-filter over many
   files it's likely more text (and more billed tokens) than needed. Measure whether
   8,000 changes any answers on a real file set before lowering it.
4. **`--jobs` default and rate limits.** No limiting was seen at 16 in flight over 48
   items. What are TypeSafe's limits at thousands of items?
