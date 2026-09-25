"""``semop filter`` (also ``semfilter``): grep by meaning.

    ls ~/Downloads | semfilter "Is this an installer or other throwaway download?"
    find logs -name '*.md' | semfilter --files "Does this log record a failed release?"

Asks one yes/no question of every input item, one call each, and prints the items the
model says yes to, in input order and exactly as read. See docs/design/semop-filter.md.

Exit codes, like grep: 0 something was printed, 1 nothing matched, 2 bad command line
or input (nothing sent), 3 some items failed (the rest are still printed).
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from typing import Any, TextIO

from ..filtering import Filtered, Verdict, filter_items_async
from ..types import Boolean
from .backends import Settings, open_provider

MATCHED, NONE, INVALID, SOME_FAILED = 0, 1, 2, 3
BINARY_CHECK_BYTES = 8192


@dataclass
class Item:
    raw: str                 # the input line, printed back exactly as read
    state: Any = None        # what's sent (None if skipped)
    cut: bool = False        # text was cut to --max-chars
    skipped: str | None = None


def add_parser(commands: Any, shared: argparse.ArgumentParser) -> None:
    command = commands.add_parser(
        "filter", parents=[shared], help="ask one yes/no question of every item on stdin",
        description="Ask one yes/no question of every item on stdin (one per line) and print "
                    "the items the model says yes to. Exit 0 printed something, 1 no matches, "
                    "2 bad input, 3 some items failed.")
    command.add_argument("question", help="the yes/no question asked of every item")
    kind = command.add_mutually_exclusive_group()
    kind.add_argument("--files", action="store_true",
                      help="each line is a file path; send the file's text")
    kind.add_argument("--jsonl", action="store_true",
                      help="each line is a JSON value; send it as the item")
    command.add_argument("--min", type=_fraction, default=0.8, metavar="P",
                         help="min_confidence: below it an item is unsure, never a match "
                              "(default: 0.8)")
    command.add_argument("--context", metavar="TEXT",
                         help="shared background sent with every item")
    command.add_argument("--keep-unsure", action="store_true",
                         help="print unsure items along with the matches")
    command.add_argument("-v", "--invert", action="store_true",
                         help="print the confident no's instead of the matches")
    command.add_argument("--records", action="store_true",
                         help="print one JSON record per item (every outcome, with probabilities)")
    command.add_argument("--jobs", type=_positive_int, default=8, metavar="N",
                         help="calls in flight at once (default: 8)")
    command.add_argument("--max-chars", type=_positive_int, default=32000, metavar="N",
                         help="cut each item's text to N characters, and say so (default: 32000)")
    command.add_argument("--no-path", action="store_true",
                         help="with --files, send only the text, not the path")
    command.add_argument("--dry-run", action="store_true",
                         help="read the input and show what would be sent; send nothing")
    command.add_argument("-q", "--quiet", action="store_true",
                         help="leave out the per-item lines on stderr; keep the summary")


def run(args: argparse.Namespace, settings: Settings, stdin: TextIO,
        stdout: TextIO | None = None, stderr: TextIO | None = None) -> int:
    stdout, stderr = stdout or sys.stdout, stderr or sys.stderr  # looked up now, not at import
    try:
        items = _read_items(stdin, args)
    except ValueError as problem:
        print(f"semop filter: {problem}", file=stderr)
        return INVALID
    question = Boolean(args.question, min_confidence=args.min)
    to_send = {str(i): item.state for i, item in enumerate(items) if item.skipped is None}

    if args.dry_run:
        return _dry_run(args, settings, question, items, to_send, stdout)

    start = time.perf_counter()
    filtered = asyncio.run(_filter(settings, question, to_send, args))
    verdicts = {v.id: v for v in filtered.verdicts}
    by_item = [(item, verdicts.get(str(i))) for i, item in enumerate(items)]

    selected = {"no"} if args.invert else {"match"}
    if args.keep_unsure:
        selected.add("unsure")
    printed = False
    for item, verdict in by_item:
        if args.records:
            print(json.dumps(_record(item, verdict), ensure_ascii=False), file=stdout)
        if verdict and verdict.outcome in selected:
            printed = True
            if not args.records:
                print(item.raw, file=stdout)
    if not args.quiet:
        for item, verdict in by_item:
            if line := _problem_line(item, verdict, args.max_chars):
                print(line, file=stderr)
    print(_summary(filtered, items, settings, args, time.perf_counter() - start), file=stderr)

    if filtered.failed:
        return SOME_FAILED
    return MATCHED if printed else NONE


async def _filter(settings: Settings, question: Boolean, to_send: dict[str, Any],
                  args: argparse.Namespace) -> Filtered:
    if not to_send:
        return Filtered(question, [])
    async with open_provider(settings) as provider:
        return await filter_items_async(provider, question, to_send, context=args.context,
                                        concurrency=args.jobs)


def _read_items(stdin: TextIO, args: argparse.Namespace) -> list[Item]:
    if stdin.isatty():
        raise ValueError("no input: pipe items to stdin, one per line")
    items = []
    for number, line in enumerate(stdin.read().splitlines(), 1):
        if not line.strip():
            continue
        if args.jsonl:
            try:
                items.append(Item(line, json.loads(line)))
            except json.JSONDecodeError as problem:
                raise ValueError(f"line {number} isn't valid JSON: {problem}") from None
        elif args.files:
            items.append(_file_item(line, args))
        else:
            text, cut = _cut(line, args.max_chars)
            items.append(Item(line, text, cut))
    return items


def _file_item(path: str, args: argparse.Namespace) -> Item:
    try:
        with open(path, "rb") as file:
            data = file.read()
    except OSError as problem:
        return Item(path, skipped=f"can't read: {problem.strerror or problem}")
    if b"\0" in data[:BINARY_CHECK_BYTES]:
        return Item(path, skipped="binary")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return Item(path, skipped="not UTF-8 text")
    text, cut = _cut(text, args.max_chars)
    return Item(path, text if args.no_path else {"path": path, "text": text}, cut)


def _cut(text: str, max_chars: int) -> tuple[str, bool]:
    return (text[:max_chars], True) if len(text) > max_chars else (text, False)


def _dry_run(args: argparse.Namespace, settings: Settings, question: Boolean, items: list[Item],
             to_send: dict[str, Any], stdout: TextIO) -> int:
    skipped = len(items) - len(to_send)
    print(f"semop filter: would send {len(to_send)} item(s) to {settings.provider} "
          f"({settings.model_name}), one call each; {skipped} skipped", file=stdout)
    if to_send:
        first = next(iter(to_send.values()))
        state = first if args.context is None else {"context": args.context, "item": first}
        print(json.dumps({"state": state, "question": question.instructions,
                          "min_confidence": question.min_confidence},
                         ensure_ascii=False, indent=2), file=stdout)
    for item in items:
        if item.skipped:
            print(f"- {item.raw}  skipped: {item.skipped}", file=stdout)
    return MATCHED


def _record(item: Item, verdict: Verdict | None) -> dict[str, Any]:
    answer = verdict.answer if verdict else None
    return {
        "item": item.raw,
        "outcome": verdict.outcome if verdict else "skipped",
        "p_true": _round(answer.probabilities["true"]) if answer else None,
        "confidence": _round(answer.confidence) if answer else None,
        "cut": item.cut,
        "model": answer.call.model if answer and answer.call else None,
        "error": str(verdict.error) if verdict and verdict.error else item.skipped,
    }


def _problem_line(item: Item, verdict: Verdict | None, max_chars: int) -> str | None:
    if item.skipped:
        return f"- {item.raw:<40} skipped: {item.skipped}"
    if verdict and verdict.outcome == "failed":
        return f"! {item.raw:<40} {verdict.error}"
    if verdict and verdict.outcome == "unsure":
        return f"? {item.raw:<40} p(true)={_round(verdict.answer.probabilities['true'])}"
    if item.cut:
        return f"~ {item.raw:<40} cut to {max_chars} chars"
    return None


def _summary(filtered: Filtered, items: list[Item], settings: Settings,
             args: argparse.Namespace, seconds: float) -> str:
    skipped = sum(1 for item in items if item.skipped)
    models = sorted(filtered.models) or [settings.model_name]
    parts = [f"{len(filtered.matched)} matched", f"{len(filtered.rejected)} no",
             f"{len(filtered.unsure)} unsure", f"{len(filtered.failed)} failed",
             f"{skipped} skipped of {len(items)}",
             f"sent {len(filtered.verdicts)} to {settings.provider} {', '.join(models)}",
             f"{seconds:.1f}s"]
    if len(filtered.models) > 1:
        parts.append("warning: answers came from more than one model")
    if args.min == 0:
        parts.append("--min 0: nothing is ever unsure")
    return "semop filter: " + " · ".join(parts)


def _round(x: float) -> float:
    return float(f"{x:.4g}")


def _fraction(text: str) -> float:
    value = float(text)
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError("must be a number from 0 to 1")
    return value


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive whole number")
    return value
