"""An MCP server over stdio with one tool, ``ask``. Needs ``pip install "semantic-operators[mcp]"``.

    semop mcp --provider typesafe                # hosted Jev
    semop mcp --provider laya --timeout 60       # local Laya

The provider and model are fixed when the server starts: nothing an agent sends can
change them. The tool takes the same request as ``semop ask`` (see wire.py) and returns
the same response. A "don't know" is a normal result; a bad request, a provider failure,
or a timeout comes back as a tool error the agent can read.
"""

import contextlib
import json
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager
from typing import Any, Literal

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from typing_extensions import NotRequired, TypedDict  # pydantic needs these on Python 3.11

from ..provider import AsyncProvider
from . import wire
from .backends import Settings, open_provider
from .cli import OK, ask

DESCRIPTION = """\
Ask a System One model (a small, fast classifier, not a chat model) typed questions
about some text or JSON, and get answers with probabilities.

Question types: "boolean" (yes/no), "choice" (pick one of the options), "score" (rate
on ordered levels; the value is the expected level index, e.g. 1.6 of 0..2). Ask several
questions about the same state in one call: it costs about the same as one.

Each answer has value, decided, confidence, and probabilities. If decided is false the
model wasn't sure (value is null): treat that as unresolved, not as false or no, and
don't swap in the most likely option yourself. Don't reword a question or lower
min_confidence just to get the answer you want. Confidence is the model's own view,
not a guarantee. Use this for bounded judgments (classify, route, flag, score), not
open-ended reasoning or permissions. The tool only answers; it doesn't act."""


class Option(TypedDict):
    name: str
    description: NotRequired[str]


class QuestionSpec(TypedDict):
    name: str
    type: Literal["boolean", "choice", "score"]
    instructions: str
    options: NotRequired[list[str | Option]]
    levels: NotRequired[list[str]]
    true: NotRequired[str]
    false: NotRequired[str]
    min_confidence: NotRequired[float]


def build_server(open: Callable[[], AbstractAsyncContextManager[AsyncProvider]],
                 about: str) -> MCPServer:
    """A server whose ``ask`` tool uses the provider ``open()`` yields for its lifetime."""
    holder: dict[str, AsyncProvider] = {}

    @contextlib.asynccontextmanager
    async def lifespan(_: MCPServer) -> AsyncIterator[None]:
        async with open() as provider:
            holder["provider"] = provider
            yield

    # Warnings only on stderr: per-request INFO logs would fill the host's MCP log.
    server = MCPServer("semantic-operators", instructions=about, lifespan=lifespan,
                       log_level="WARNING")

    @server.tool(name="ask", description=f"{DESCRIPTION}\n\n{about}")
    async def ask_tool(state: str | dict[str, Any] | list[Any], questions: list[QuestionSpec],
                       request_id: str | None = None,
                       schema_version: Literal["1"] = "1") -> dict[str, Any]:
        request: dict[str, Any] = {"schema_version": schema_version, "state": state,
                                   "questions": questions}
        if request_id is not None:
            request["request_id"] = request_id
        try:
            state_, questions_, request_id_ = wire.decode(request)
        except wire.WireError as problem:
            raise ToolError(json.dumps(wire.error("invalid_request", str(problem), request_id)))
        response, code = await ask(holder["provider"], state_, questions_, request_id_)
        if code != OK:
            raise ToolError(json.dumps(response))
        return response

    return server


def serve(settings: Settings) -> int:
    where = "on this machine" if settings.provider == "laya" else \
        "by a hosted API: the state and questions are sent to it, and calls may be billed"
    about = f"Answers come from {settings.provider} ({settings.model_name}), {where}."
    build_server(lambda: open_provider(settings), about).run()
    return 0
