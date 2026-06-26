"""Unit tests for the model-agnostic GRPO reward adapter (no GPU, no trainer)."""

import importlib.util
import os

import pytest

# Load training/rewards_grpo.py by path (it lives outside the package).
_PATH = os.path.join(os.path.dirname(__file__), "..", "training", "rewards_grpo.py")
_spec = importlib.util.spec_from_file_location("rewards_grpo", _PATH)
rg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rg)


def _c(text):
    return [{"content": text}]


GOOD = '{"reasoning_steps": ["The light is green", "Green means go"], "answer": "B"}'


def test_build_reward_funcs_names_and_order():
    funcs = rg.build_reward_funcs(["format", "accuracy", "reasoning_causal"])
    assert len(funcs) == 3
    assert funcs[0].__name__ == "crystal_format_reward"
    assert funcs[2].__name__ == "crystal_reasoning_causal_reward"


def test_build_reward_funcs_unknown():
    with pytest.raises(ValueError):
        rg.build_reward_funcs(["nope"])


def test_accuracy_adapter_reads_solution_kwarg():
    funcs = rg.build_reward_funcs(["accuracy"])
    # The trainer passes ground truth as `solution`.
    out = funcs[0]([_c(GOOD)], solution=["B"])
    assert out[0] > 0.0
    out_wrong = funcs[0]([_c(GOOD)], solution=["C"])
    assert out_wrong[0] == 0.0


def test_cpr_adapter_passes_weights():
    funcs = rg.build_reward_funcs(["reasoning_causal"],
                                  causal_answer_weight=0.65, causal_step_weight=0.35)
    refs = [["The light is green", "Green means go"]]
    out = funcs[0]([_c(GOOD)], ground_truths=["B"], reference_steps=refs)
    # Correct answer => at least answer_weight (0.65).
    assert out[0] >= 0.65


def test_resolve_reward_names_causal():
    assert rg.resolve_reward_names(["format", "accuracy", "reasoning"], use_causal=True) == \
        ["format", "accuracy", "reasoning_causal"]


def test_resolve_reward_names_semantic():
    assert rg.resolve_reward_names(["format", "accuracy", "reasoning"], use_semantic=True) == \
        ["format", "accuracy", "reasoning_semantic"]


def test_resolve_reward_names_default():
    assert rg.resolve_reward_names(["format", "accuracy", "reasoning"]) == \
        ["format", "accuracy", "reasoning"]
