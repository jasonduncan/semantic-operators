"""The ``semop`` command. Standard library only (the MCP server needs ``[mcp]``).

    semop ask < request.json                          # TypeSafe (the default provider)
    semop ask --provider laya --request request.json --pretty
    semop mcp                                         # stdio MCP server, see mcp_server.py

``ask`` reads one JSON request (see wire.py) and prints one JSON response. Exit codes:
0 answered (a "don't know" is an answer), 1 the provider failed or timed out,
2 the request or command line was invalid.
"""

import argparse
import asyncio
import json
import sys
from collections.abc import Mapping, Sequence
from importlib.metadata import version
from typing import Any, TextIO

from ..errors import ProviderError, ProviderTimeout
from ..provider import AsyncProvider
from ..types import Question, State
from . import wire
from .backends import DEFAULT_MODELS, DEFAULT_PROVIDER, PROVIDERS, Settings, missing_setup, open_provider

OK, FAILED, INVALID = 0, 1, 2


def main(argv: Sequence[str] | None = None, stdin: TextIO | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = Settings(args.provider, args.model, args.timeout)
    if problem := missing_setup(settings):
        print(f"semop: {problem}", file=sys.stderr)
        return INVALID
    if args.command == "mcp":
        try:
            from .mcp_server import serve
        except ImportError:
            print('semop: the MCP server isn\'t installed; install it with: '
                  'pip install "semantic-operators[mcp]"', file=sys.stderr)
            return INVALID
        return serve(settings)

    try:
        text = _read(args.request, stdin or sys.stdin)
        state, questions, request_id = wire.decode(json.loads(text))
    except (OSError, UnicodeDecodeError) as problem:
        return _emit(wire.error("invalid_request", f"can't read the request: {problem}"),
                     INVALID, args.pretty)
    except json.JSONDecodeError as problem:
        return _emit(wire.error("invalid_request", f"not valid JSON: {problem}"), INVALID, args.pretty)
    except wire.WireError as problem:
        return _emit(wire.error("invalid_request", str(problem)), INVALID, args.pretty)

    response, code = asyncio.run(_ask_with(settings, state, questions, request_id))
    return _emit(response, code, args.pretty)


async def ask(provider: AsyncProvider, state: State, questions: Mapping[str, Question],
              request_id: str | None = None) -> tuple[dict[str, Any], int]:
    """One request through ``provider``: the JSON response and the exit code."""
    try:
        answers = await provider.ask(state, questions)
    except ProviderTimeout as problem:
        return wire.error("timeout", str(problem), request_id), FAILED
    except ProviderError as problem:
        return wire.error("provider_error", str(problem), request_id), FAILED
    return wire.encode(answers, questions, request_id), OK


async def _ask_with(settings: Settings, state: State, questions: Mapping[str, Question],
                    request_id: str | None) -> tuple[dict[str, Any], int]:
    async with open_provider(settings) as provider:
        return await ask(provider, state, questions, request_id)


def _read(path: str, stdin: TextIO) -> str:
    if path == "-":
        if stdin.isatty():
            raise OSError("no request given: pass --request FILE or pipe JSON to stdin")
        return stdin.read()
    with open(path, encoding="utf-8") as file:
        return file.read()


def _emit(response: dict[str, Any], code: int, pretty: bool) -> int:
    print(json.dumps(response, indent=2 if pretty else None, ensure_ascii=False))
    return code


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="semop", description="Ask System One models typed questions, from the shell or MCP.")
    parser.add_argument("--version", action="version",
                        version=f"semop {version('semantic-operators')}")
    commands = parser.add_subparsers(dest="command", required=True)

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--provider", default=DEFAULT_PROVIDER, choices=PROVIDERS,
                        help=f"which provider answers, fixed for the whole process "
                             f"(default: {DEFAULT_PROVIDER})")
    shared.add_argument("--model", help="model to use (default: "
                        + ", ".join(f"{p}: {m}" for p, m in DEFAULT_MODELS.items()) + ")")
    shared.add_argument("--timeout", type=_seconds, metavar="SECONDS",
                        help="time limit per call (a local model's first call includes loading it)")

    ask_command = commands.add_parser("ask", parents=[shared],
                                      help="answer one JSON request and print JSON")
    ask_command.add_argument("--request", default="-", metavar="FILE",
                             help="request file (default: read stdin)")
    ask_command.add_argument("--pretty", action="store_true", help="indent the JSON output")
    commands.add_parser("mcp", parents=[shared], help="run an MCP server over stdio")
    return parser


def _seconds(text: str) -> float:
    value = float(text)
    if not 0 < value < float("inf"):
        raise argparse.ArgumentTypeError("must be a positive number of seconds")
    return value
