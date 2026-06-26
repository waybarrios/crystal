"""
Parity: crystal_metrics.rewards.causal_process_reward must match the original
VLM-R1 causal_reward.causal_intervention_reward exactly.

Skipped if the original VLM-R1 checkout is absent (set CRYSTAL_VLMR1_PATH).
"""

import importlib
import os
import sys

import pytest

from crystal_metrics import rewards

VLMR1_PATH = os.environ.get("CRYSTAL_VLMR1_PATH", "/gpudata3/Wayner/VLM-R1")
MLLM_EVAL = os.path.join(VLMR1_PATH, "mllm_evaluator")


def _import_original_cpr():
    if not os.path.isfile(os.path.join(MLLM_EVAL, "causal_reward.py")):
        pytest.skip(f"Original VLM-R1 causal_reward not found at {MLLM_EVAL}")
    # causal_reward.py does `from simple_similarity import best_match_f1`, so the
    # mllm_evaluator dir itself must be importable as a top-level path.
    if MLLM_EVAL not in sys.path:
        sys.path.insert(0, MLLM_EVAL)
    try:
        return importlib.import_module("causal_reward")
    except Exception as e:  # pragma: no cover
        pytest.skip(f"Could not import original causal_reward: {e}")


def _c(text):
    return [{"content": text}]


# Cases: correct+good, correct+weak, wrong+good, wrong+weak, XML format, empty.
COMPLETIONS = [
    _c('{"reasoning_steps": ["The traffic light is green", "Green means go"], "answer": "B"}'),
    _c('{"reasoning_steps": ["Something unrelated entirely"], "answer": "B"}'),
    _c('{"reasoning_steps": ["The traffic light is green", "Green means go"], "answer": "C"}'),
    _c('{"reasoning_steps": ["Random words here"], "answer": "C"}'),
    _c("<think>Step 1: The light is green. Step 2: Green means go.</think><answer>B</answer>"),
    _c("no structure at all"),
]
GROUND_TRUTHS = ["B", "B", "B", "B", "B", "B"]
REFERENCE_STEPS = [
    ["Observe the traffic light", "The light is green", "Green means go"],
    ["Observe the traffic light", "The light is green", "Green means go"],
    ["Observe the traffic light", "The light is green", "Green means go"],
    ["Observe the traffic light", "The light is green", "Green means go"],
    ["Observe the traffic light", "The light is green", "Green means go"],
    ["Observe the traffic light", "The light is green", "Green means go"],
]


@pytest.mark.parametrize("answer_weight,step_weight", [(0.6, 0.4), (0.65, 0.35), (0.5, 0.5)])
def test_cpr_parity(answer_weight, step_weight):
    orig = _import_original_cpr()

    new_rewards = rewards.causal_process_reward(
        COMPLETIONS, ground_truths=GROUND_TRUTHS, reference_steps=REFERENCE_STEPS,
        answer_weight=answer_weight, step_weight=step_weight,
    )
    old_rewards = orig.causal_intervention_reward(
        completions=COMPLETIONS, ground_truths=GROUND_TRUTHS, reference_steps=REFERENCE_STEPS,
        answer_weight=answer_weight, step_weight=step_weight,
    )

    assert len(new_rewards) == len(old_rewards) == len(COMPLETIONS)
    for i, (nv, ov) in enumerate(zip(new_rewards, old_rewards)):
        assert nv == pytest.approx(ov, rel=1e-9, abs=1e-12), f"completion {i}: {nv} != {ov}"
