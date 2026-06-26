"""Unit tests for GRPO reward functions."""

import pytest

from crystal_metrics import rewards


def _c(text):
    """Wrap raw text as a single trainer-format completion."""
    return [{"content": text}]


GOOD_JSON = '{"reasoning_steps": ["The light is green", "Green means go"], "answer": "B"}'
BAD_JSON = "this is not json at all"
EMPTY_STEPS = '{"reasoning_steps": [], "answer": "B"}'


# --- parse_reasoning_steps ---------------------------------------------------
def test_parse_json_format():
    steps, answer = rewards.parse_reasoning_steps(GOOD_JSON)
    assert steps == ["The light is green", "Green means go"]
    assert answer == "B"


def test_parse_xml_format():
    text = "<think>Step 1: Look at the image. Step 2: It is green.</think><answer>B</answer>"
    steps, answer = rewards.parse_reasoning_steps(text)
    assert len(steps) == 2
    assert answer == "B"


# --- check_answer_match ------------------------------------------------------
@pytest.mark.parametrize("pred,gt,expected", [
    ("B", "B", True),
    ("b", "B", True),
    ("B) the answer", "B", True),
    ("C", "B", False),
    ("yes", "true", True),
    ("no", "yes", False),
    ("5", "5.0", True),
    ("5", "42", False),
])
def test_check_answer_match(pred, gt, expected):
    assert rewards.check_answer_match(pred, gt) is expected


# --- format_reward -----------------------------------------------------------
def test_format_reward_valid_and_invalid():
    out = rewards.format_reward([_c(GOOD_JSON), _c(BAD_JSON), _c(EMPTY_STEPS)])
    assert out == [1.0, 0.0, 0.0]


def test_format_reward_with_code_fence():
    fenced = "```json\n" + GOOD_JSON + "\n```"
    assert rewards.format_reward([_c(fenced)]) == [1.0]


# --- accuracy_reward (rule-based, no LLM) ------------------------------------
def test_accuracy_reward_choice():
    out = rewards.accuracy_reward([_c(GOOD_JSON)], solution=["B"], use_llm_grader=False)
    assert out[0] > 0.0
    out_wrong = rewards.accuracy_reward([_c(GOOD_JSON)], solution=["C"], use_llm_grader=False)
    assert out_wrong[0] == 0.0


# --- word_overlap_reasoning_reward -------------------------------------------
def test_word_overlap_reasoning_reward():
    refs = [["The traffic light is green", "Green means go ahead"]]
    out = rewards.word_overlap_reasoning_reward([_c(GOOD_JSON)], reference_steps=refs)
    assert 0.0 < out[0] <= 1.0


def test_reasoning_reward_zero_when_no_reference():
    out = rewards.word_overlap_reasoning_reward([_c(GOOD_JSON)], reference_steps=[[]])
    assert out[0] == 0.0


# --- causal_process_reward (CPR) ---------------------------------------------
def test_cpr_correct_beats_wrong():
    refs = [["The light is green", "Green means go"]]
    correct = rewards.causal_process_reward([_c(GOOD_JSON)], ground_truths=["B"], reference_steps=refs)
    wrong = rewards.causal_process_reward([_c(GOOD_JSON)], ground_truths=["C"], reference_steps=refs)
    assert correct[0] > wrong[0]
    # Correct answer => at least answer_weight (0.6).
    assert correct[0] >= 0.6


def test_cpr_reads_ground_truth_from_kwargs():
    refs = [["The light is green", "Green means go"]]
    out = rewards.causal_process_reward([_c(GOOD_JSON)], reference_steps=refs, solution=["B"])
    assert out[0] >= 0.6


# --- select_reward_func registry ---------------------------------------------
@pytest.mark.parametrize("name,func", [
    ("format", rewards.format_reward),
    ("accuracy", rewards.accuracy_reward),
    ("reasoning", rewards.word_overlap_reasoning_reward),
    ("reasoning_semantic", rewards.semantic_reasoning_reward),
    ("reasoning_causal", rewards.causal_process_reward),
])
def test_select_reward_func(name, func):
    assert rewards.select_reward_func(name) is func


def test_select_reward_func_unknown():
    with pytest.raises(ValueError):
        rewards.select_reward_func("does_not_exist")
