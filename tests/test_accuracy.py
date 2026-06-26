"""Unit tests for multi-format accuracy (rule-based, no LLM judge)."""

import pytest

from crystal_metrics.accuracy import AccuracyCalculator, AnswerNormalizer


@pytest.fixture
def calc():
    return AccuracyCalculator(use_llm_grader=False)


# --- numeric -----------------------------------------------------------------
def test_numeric_exact(calc):
    r = calc.evaluate_single("How many?", "5", "5.0")
    assert r.is_correct
    assert r.match_type.startswith("numeric")


def test_numeric_within_tolerance(calc):
    r = calc.evaluate_single("Value?", "0.67", "0.67234572345763")
    assert r.is_correct
    assert r.match_type == "numeric_rounded"


def test_numeric_mismatch(calc):
    r = calc.evaluate_single("How many?", "5", "42")
    assert not r.is_correct
    assert r.match_type == "numeric_mismatch"


# --- yes/no ------------------------------------------------------------------
def test_yes_no_correct(calc):
    assert calc.evaluate_single("Is it green?", "Yes", "yes").is_correct


def test_yes_no_incorrect(calc):
    r = calc.evaluate_single("Is it green?", "Yes", "No")
    assert not r.is_correct
    assert r.match_type == "yes_no"


# --- multiple choice ---------------------------------------------------------
@pytest.mark.parametrize("pred", ["A", "(A)", "a)", "The answer is A"])
def test_choice_variants(calc, pred):
    r = calc.evaluate_single("What color?", pred, "A")
    assert r.is_correct
    assert r.match_type == "choice"


def test_choice_wrong(calc):
    r = calc.evaluate_single("What color?", "B", "A")
    assert not r.is_correct


def test_choice_from_option_text(calc):
    question = "Capital of France?\nA) London\nB) Paris\nC) Rome"
    r = calc.evaluate_single(question, "b) Paris", "B")
    assert r.is_correct


# --- text --------------------------------------------------------------------
def test_exact_text(calc):
    assert calc.evaluate_single("Q", "Paris", "paris").is_correct


def test_single_word_mismatch_no_judge(calc):
    r = calc.evaluate_single("Q", "London", "Paris")
    assert not r.is_correct
    assert r.match_type == "exact"


def test_substring_fallback_long_text(calc):
    r = calc.evaluate_single(
        "Q", "the capital of france is paris", "capital of france is paris"
    )
    assert r.is_correct  # substring fallback when judge disabled


# --- dataset aggregation -----------------------------------------------------
def test_evaluate_dataset(calc):
    preds = {
        "0": {"question": "Is it green?", "answer": "Yes"},
        "1": {"question": "How many?", "answer": "5"},
        "2": {"question": "Color?", "answer": "A"},
    }
    refs = {
        "0": {"answer": "Yes"},
        "1": {"answer": "5"},
        "2": {"answer": "B"},
    }
    out = calc.evaluate_dataset(preds, refs)
    assert out["total_samples"] == 3
    assert out["correct_samples"] == 2
    assert out["overall_accuracy"] == pytest.approx(2 / 3)
    assert "type_statistics" in out


# --- normalizer units --------------------------------------------------------
def test_extract_number_from_prose():
    assert AnswerNormalizer.extract_number("The answer is 42 objects") == 42.0


def test_normalize_yes_no_variants():
    assert AnswerNormalizer.normalize_yes_no("Correct") == "yes"
    assert AnswerNormalizer.normalize_yes_no("false") == "no"


def test_llm_grader_import_guard_message():
    # Disabled judge must never touch openai; enabling without extra raises clearly.
    # We only assert the disabled path here (no network, no import).
    c = AccuracyCalculator(use_llm_grader=False)
    assert c.llm_grader is None
