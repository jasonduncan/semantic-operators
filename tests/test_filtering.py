"""Filtering, offline: the library (filter_items) and the command (semop filter / semfilter).

A fake provider decides from the item's text: "yes" matches, "no" doesn't, "maybe" is
unsure at the default --min 0.8, "boom" fails. Later items answer first, to check that
output keeps input order."""

import asyncio
import contextlib
import io
import json

import pytest

from semantic_operators import Boolean, Call, ProviderError, make_answer
from semantic_operators.filtering import filter_items, filter_items_async
from semantic_operators.interfaces import cli, filter_cli

QUESTION = Boolean("Is it a yes?", min_confidence=0.8)


def text_of(state):
    if isinstance(state, str):
        return state
    if "item" in state:
        return text_of(state["item"])
    return state.get("text", json.dumps(state))


class Fake:
    def __init__(self):
        self.states = []

    def decide(self, state, questions):
        self.states.append(state)
        text = text_of(state)
        if "boom" in text:
            raise ProviderError("Fake", "503 Service Unavailable")
        p = 0.95 if "yes" in text else 0.6 if "maybe" in text else 0.05
        model = "model-b" if "other model" in text else "model-a"
        (name, q), = questions.items()
        return {name: make_answer(q, p > 0.5, {"true": p, "false": round(1 - p, 10)},
                                  call=Call("Fake", model))}

    def ask(self, state, questions):
        return self.decide(state, questions)


class AsyncFake(Fake):
    async def ask(self, state, questions):
        await asyncio.sleep(0.02 / (1 + len(self.states)))  # later items finish first
        return self.decide(state, questions)


# The library

ITEMS = {"a": "yes please", "b": "no thanks", "c": "maybe", "d": "boom", "e": "yes again"}


def test_every_item_gets_one_outcome_in_input_order():
    result = filter_items(Fake(), QUESTION, ITEMS)
    assert [(v.id, v.outcome) for v in result.verdicts] == \
        [("a", "match"), ("b", "no"), ("c", "unsure"), ("d", "failed"), ("e", "match")]
    assert [v.id for v in result.matched] == ["a", "e"]
    assert isinstance(result.failed[0].error, ProviderError)
    assert result.models == {"model-a"}


def test_sync_and_async_agree_and_ids_are_never_sent():
    fake = AsyncFake()
    async_result = asyncio.run(filter_items_async(fake, QUESTION, ITEMS, concurrency=3))
    sync_result = filter_items(Fake(), QUESTION, ITEMS)
    assert [(v.id, v.outcome) for v in async_result.verdicts] == \
        [(v.id, v.outcome) for v in sync_result.verdicts]
    assert sorted(fake.states) == sorted(ITEMS.values())  # just the items, no ids


def test_context_wraps_each_item():
    fake = Fake()
    filter_items(fake, QUESTION, {"a": "yes"}, context="the rules")
    assert fake.states == [{"context": "the rules", "item": "yes"}]


def test_needs_a_boolean():
    with pytest.raises(TypeError):
        filter_items(Fake(), object(), ITEMS)


# The command

def run(monkeypatch, capsys, args, stdin, fake=None, key=True):
    fake = fake or AsyncFake()

    @contextlib.asynccontextmanager
    async def fake_open(settings):
        yield fake

    if key:
        monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    else:
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr(filter_cli, "open_provider", fake_open)
    code = cli.main(["filter", *args], stdin=io.StringIO(stdin))
    out, err = capsys.readouterr()
    return code, out.splitlines(), err.splitlines(), fake


LINES = "yes please\nno thanks\n\nmaybe\nyes again\n"


def test_prints_matches_in_input_order_like_grep(monkeypatch, capsys):
    code, out, err, _ = run(monkeypatch, capsys, ["Is it a yes?"], LINES)
    assert code == 0 and out == ["yes please", "yes again"]
    assert any(line.startswith("? maybe") for line in err)
    assert err[-1].startswith("semop filter: 2 matched · 1 no · 1 unsure · 0 failed · "
                              "0 skipped of 4")


def test_no_matches_exits_1(monkeypatch, capsys):
    code, out, _, _ = run(monkeypatch, capsys, ["Is it a yes?"], "no\nnope, no\n")
    assert code == 1 and out == []


def test_unsure_stays_off_stdout_unless_asked(monkeypatch, capsys):
    assert "maybe" not in run(monkeypatch, capsys, ["q", "-v"], LINES)[1]
    code, out, _, _ = run(monkeypatch, capsys, ["q", "--keep-unsure"], LINES)
    assert out == ["yes please", "maybe", "yes again"]


def test_invert_prints_confident_nos(monkeypatch, capsys):
    code, out, _, _ = run(monkeypatch, capsys, ["q", "-v"], LINES)
    assert code == 0 and out == ["no thanks"]


def test_failures_exit_3_and_the_rest_still_print(monkeypatch, capsys):
    code, out, err, _ = run(monkeypatch, capsys, ["q"], "yes\nboom\n")
    assert code == 3 and out == ["yes"]
    assert any(line.startswith("! boom") and "503" in line for line in err)


def test_records_cover_every_outcome(monkeypatch, capsys, tmp_path):
    binary = tmp_path / "logo.png"
    binary.write_bytes(b"\x89PNG\0\0")
    good = tmp_path / "notes.md"
    good.write_text("yes, release failed")
    code, out, _, _ = run(monkeypatch, capsys, ["q", "--files", "--records"],
                          f"{good}\n{binary}\n")
    records = [json.loads(line) for line in out]
    assert [(r["item"], r["outcome"]) for r in records] == [(str(good), "match"),
                                                          (str(binary), "skipped")]
    assert records[0]["p_true"] == 0.95 and records[0]["model"] == "model-a"
    assert records[1]["error"] == "binary"


def test_files_are_read_skipped_and_cut(monkeypatch, capsys, tmp_path):
    long = tmp_path / "long.md"
    long.write_text("yes " + "x" * 100)
    latin1 = tmp_path / "old.txt"
    latin1.write_bytes("caf\xe9".encode("latin-1"))
    missing = tmp_path / "missing.md"
    code, out, err, fake = run(monkeypatch, capsys, ["q", "--files", "--max-chars", "10"],
                               f"{long}\n{latin1}\n{missing}\n")
    assert out == [str(long)]
    assert fake.states == [{"path": str(long), "text": "yes xxxxxx"}]
    text = "\n".join(err)
    assert "cut to 10 chars" in text and "not UTF-8" in text and "can't read" in text

    _, _, _, fake = run(monkeypatch, capsys, ["q", "--files", "--no-path"], f"{long}\n")
    assert fake.states == [long.read_text()]


def test_jsonl_items_and_bad_lines(monkeypatch, capsys):
    code, out, _, fake = run(monkeypatch, capsys, ["q", "--jsonl"], '{"text": "yes"}\n"no"\n')
    assert out == ['{"text": "yes"}'] and fake.states == [{"text": "yes"}, "no"]
    code, out, err, fake = run(monkeypatch, capsys, ["q", "--jsonl"], '"yes"\n{broken\n')
    assert code == 2 and fake.states == [] and "line 2" in err[0]


def test_context_is_sent_with_every_item(monkeypatch, capsys):
    _, _, _, fake = run(monkeypatch, capsys, ["q", "--context", "rules"], "yes\nno\n")
    assert fake.states[0] == {"context": "rules", "item": "yes"}


def test_dry_run_sends_nothing_and_needs_no_key(monkeypatch, capsys):
    code, out, _, fake = run(monkeypatch, capsys, ["q", "--dry-run"], "yes\nno\n", key=False)
    assert code == 0 and fake.states == []
    assert out[0].startswith("semop filter: would send 2 item(s) to typesafe (jev-latest)")


def test_a_missing_key_stops_before_sending(monkeypatch, capsys):
    code, out, _, fake = run(monkeypatch, capsys, ["q"], "yes\n", key=False)
    assert code == 2 and fake.states == []


def test_mixed_models_are_flagged(monkeypatch, capsys):
    _, _, err, _ = run(monkeypatch, capsys, ["q"], "yes\nyes, other model\n")
    assert "more than one model" in err[-1] and "model-a, model-b" in err[-1]


def test_semfilter_is_semop_filter(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["semfilter", "q", "--dry-run"])
    monkeypatch.setattr("sys.stdin", io.StringIO("yes\n"))
    assert cli.filter_main() == 0
    assert "would send 1 item(s)" in capsys.readouterr().out
