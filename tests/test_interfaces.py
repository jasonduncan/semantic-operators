"""The CLI and MCP server, offline: the request/response shape, exit codes, and the MCP
tool over a real (in-process) MCP session. A fake provider stands in for the model."""

import asyncio
import contextlib
import io
import json

import pytest

from semantic_operators import Boolean, Call, Choice, ProviderError, ProviderTimeout, Score, make_answer
from semantic_operators.interfaces import cli, wire

REQUEST = {
    "schema_version": "1",
    "request_id": "ticket-42",
    "state": "The payment failed and now I cannot sign in.",
    "questions": [
        {"name": "is_complaint", "type": "boolean", "instructions": "Is it a complaint?",
         "min_confidence": 0.8},
        {"name": "department", "type": "choice", "instructions": "Which team?",
         "options": [{"name": "billing", "description": "Charges"}, "technical"],
         "min_confidence": 0.8},
        {"name": "urgency", "type": "score", "instructions": "How urgent?",
         "levels": ["low", "medium", "high"]},
    ],
}


class Fake:
    """Sure about Booleans and Scores, split 55/45 on Choices; or fails with ``error``."""

    def __init__(self, error=None):
        self.error = error

    async def ask(self, state, questions):
        if self.error:
            raise self.error
        call = Call("Fake", "fake-1", 12, None)
        answers = {}
        for name, q in questions.items():
            if isinstance(q, Boolean):
                answers[name] = make_answer(q, True, {"true": 0.9, "false": 0.1}, call=call)
            elif isinstance(q, Choice):
                first, *rest = q.options
                probs = {first: 0.55, **{o: 0.45 / len(rest) for o in rest}}
                answers[name] = make_answer(q, first, probs, call=call)
            else:
                probs = dict(zip(q.levels, [0.1, 0.2, 0.7]))
                answers[name] = make_answer(q, 1.6, probs, call=call)
        return answers


# The request/response shape

def test_decode_builds_our_question_types():
    state, questions, request_id = wire.decode(REQUEST)
    assert request_id == "ticket-42" and state.startswith("The payment")
    assert questions["department"] == Choice("Which team?", {"billing": "Charges", "technical": None},
                                             min_confidence=0.8)
    assert questions["urgency"] == Score("How urgent?", ["low", "medium", "high"])
    assert list(questions) == ["is_complaint", "department", "urgency"]


@pytest.mark.parametrize("change, path", [
    (lambda r: r.update(schema_version="2"), "schema_version"),
    (lambda r: r.pop("state"), "state"),
    (lambda r: r.update(state=5), "state"),
    (lambda r: r.update(questions=[]), "questions"),
    (lambda r: r.update(extra=1), "extra"),
    (lambda r: r["questions"][0].update(type="rank"), "questions[0].type"),
    (lambda r: r["questions"][0].update(options=["a", "b"]), "questions[0].options"),
    (lambda r: r["questions"][1].update(options=["only"]), "questions[1]"),
    (lambda r: r["questions"][1].update(options=["a", "a"]), "questions[1].options[1]"),
    (lambda r: r["questions"][2].update(name="is_complaint"), "questions[2].name"),
    (lambda r: r["questions"][2].update(levels=["x", "x"]), "questions[2]"),
    (lambda r: r["questions"][0].update(min_confidence=2), "questions[0]"),
])
def test_decode_rejects_malformed_requests_with_a_path(change, path):
    request = json.loads(json.dumps(REQUEST))
    change(request)
    with pytest.raises(wire.WireError) as caught:
        wire.decode(request)
    assert caught.value.path == path


def test_encode_keeps_undecided_as_null_and_never_includes_raw():
    state, questions, request_id = wire.decode(REQUEST)
    answers = asyncio.run(Fake().ask(state, questions))
    response = wire.encode(answers, questions, request_id)
    department = response["answers"]["department"]
    assert department == {"type": "choice", "value": None, "decided": False, "confidence": 0.55,
                          "probabilities": {"billing": 0.55, "technical": 0.45}}
    assert response["answers"]["urgency"]["value"] == 1.6
    assert response["call"] == {"provider": "Fake", "model": "fake-1",
                                "input_tokens": 12, "output_tokens": None}
    assert "raw" not in json.dumps(response)


# The CLI

def run_cli(monkeypatch, capsys, request, provider=None, args=("--provider", "typesafe")):
    @contextlib.asynccontextmanager
    async def fake_open(settings):
        yield provider or Fake()

    monkeypatch.setattr(cli, "open_provider", fake_open)
    stdin = io.StringIO(request if isinstance(request, str) else json.dumps(request))
    code = cli.main(["ask", *args], stdin=stdin)
    return code, json.loads(capsys.readouterr().out)


def test_cli_answers_with_exit_0_even_when_undecided(monkeypatch, capsys):
    code, response = run_cli(monkeypatch, capsys, REQUEST)
    assert code == 0 and response["request_id"] == "ticket-42"
    assert response["answers"]["department"]["decided"] is False


@pytest.mark.parametrize("request_text", ["{not json", json.dumps({"state": "hi"})])
def test_cli_invalid_requests_exit_2_without_asking(monkeypatch, capsys, request_text):
    code, response = run_cli(monkeypatch, capsys, request_text, provider=Fake(RuntimeError("asked")))
    assert code == 2 and response["error"]["code"] == "invalid_request"


@pytest.mark.parametrize("error, code_name", [
    (ProviderError("Fake", "down"), "provider_error"),
    (ProviderTimeout("Fake", "too slow"), "timeout"),
])
def test_cli_provider_failures_exit_1(monkeypatch, capsys, error, code_name):
    code, response = run_cli(monkeypatch, capsys, REQUEST, provider=Fake(error))
    assert code == 1 and response == {"request_id": "ticket-42",
                                      "error": {"code": code_name, "message": str(error)}}


def test_mcp_without_the_extra_gives_an_install_hint(monkeypatch, capsys):
    import sys
    monkeypatch.setitem(sys.modules, "mcp", None)  # as if [mcp] weren't installed
    monkeypatch.delitem(sys.modules, "semantic_operators.interfaces.mcp_server", raising=False)
    assert cli.main(["mcp", "--provider", "typesafe"]) == 2
    assert 'pip install "semantic-operators[mcp]"' in capsys.readouterr().err


def test_cli_rejects_unknown_providers():
    with pytest.raises(SystemExit) as caught:
        cli.main(["ask", "--provider", "openai"])
    assert caught.value.code == 2


# The MCP server, over an in-process MCP session

def call_mcp(provider, arguments):
    mcp = pytest.importorskip("mcp")
    from semantic_operators.interfaces.mcp_server import build_server

    @contextlib.asynccontextmanager
    async def fake_open():
        yield provider

    async def session():
        async with mcp.Client(build_server(fake_open, "Answers come from a fake.")) as client:
            tools = await client.list_tools()
            return tools.tools, await client.call_tool("ask", arguments)

    return asyncio.run(session())


def test_mcp_ask_returns_the_same_response_as_the_cli():
    tools, result = call_mcp(Fake(), REQUEST)
    assert [t.name for t in tools] == ["ask"]
    assert "unresolved" in tools[0].description and "a fake" in tools[0].description
    assert not result.is_error
    state, questions, request_id = wire.decode(REQUEST)
    expected = wire.encode(asyncio.run(Fake().ask(state, questions)), questions, request_id)
    assert result.structured_content == expected


@pytest.mark.parametrize("provider, arguments, code", [
    (Fake(), {**REQUEST, "questions": [{"name": "x", "type": "choice", "instructions": "?",
                                        "options": ["only"]}]}, "invalid_request"),
    (Fake(ProviderError("Fake", "down")), REQUEST, "provider_error"),
    (Fake(ProviderTimeout("Fake", "too slow")), REQUEST, "timeout"),
])
def test_mcp_failures_are_readable_tool_errors(provider, arguments, code):
    _, result = call_mcp(provider, arguments)
    assert result.is_error
    text = result.content[0].text
    assert json.loads(text[text.index("{"):])["error"]["code"] == code
