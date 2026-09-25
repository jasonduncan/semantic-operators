"""Offline checks of answer building and scoring. No provider or model is called."""

from semantic_operators import Boolean, Choice, Score, make_answer
from semantic_operators.bench import Case, at_min_confidence, score

yes_no = Boolean("Is it?")
pick = Choice("Which?", {"a": None, "b": None, "c": None})
rate = Score("How much?", ["low", "medium", "high"])


def test_boolean_confidence_and_tie():
    answer = make_answer(yes_no, True, {"true": 0.8, "false": 0.2})
    assert answer.value is True and answer.confidence == 0.8 and answer.decided
    tie = make_answer(yes_no, False, {"true": 0.5, "false": 0.5})
    assert tie.value is None and not tie.decided


def test_choice_tie_is_undecided_but_keeps_probabilities():
    answer = make_answer(pick, "a", {"a": 0.4, "b": 0.4, "c": 0.2})
    assert answer.value is None
    assert answer.probabilities == {"a": 0.4, "b": 0.4, "c": 0.2}


def test_score_confidence_is_nearest_level():
    answer = make_answer(rate, 1.6, {"low": 0.1, "medium": 0.2, "high": 0.7})
    assert answer.value == 1.6 and answer.confidence == 0.7  # 1.6 rounds to level 2


def test_min_confidence_withholds_low_confidence_answers():
    strict = Choice("Which?", {"a": None, "b": None}, min_confidence=0.9)
    assert make_answer(strict, "a", {"a": 0.85, "b": 0.15}).value is None
    assert make_answer(strict, "a", {"a": 0.95, "b": 0.05}).value == "a"


def test_scoring_counts_undecided_separately_from_misses():
    questions = {"q": pick}
    cases = [Case("1", {"q": "a"}), Case("2", {"q": "b"}), Case("3", {"q": "c"})]
    answers = [
        {"q": make_answer(pick, "a", {"a": 0.9, "b": 0.05, "c": 0.05})},  # right, sure
        {"q": make_answer(pick, "a", {"a": 0.6, "b": 0.3, "c": 0.1})},    # wrong, unsure
        {"q": make_answer(pick, "a", {"a": 0.4, "b": 0.4, "c": 0.2})},    # tie: undecided
    ]
    report = score(questions, cases, answers, [1.0] * 3, 3.0)
    s = report.questions["q"]
    assert (s.correct, s.undecided, s.answered, len(report.misses)) == (1, 1, 2, 1)

    strict = at_min_confidence(report, questions, cases, 0.8).questions["q"]
    assert (strict.correct, strict.undecided, strict.accuracy_when_answered) == (1, 2, 1.0)
