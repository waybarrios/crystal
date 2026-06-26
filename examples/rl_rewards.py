#!/usr/bin/env python3
"""
Example: CRYSTAL RL reward functions (CPR / SPR / format / accuracy).

Run:
    python examples/rl_rewards.py
"""

from crystal_metrics import (
    causal_process_reward,
    format_reward,
    select_reward_func,
    word_overlap_reasoning_reward,
)

# Completions are in the GRPO trainer format: [[{"content": "..."}], ...].
completions = [
    [{"content": '{"reasoning_steps": ["The light is green", "Green means go"], "answer": "B"}'}],
    [{"content": '{"reasoning_steps": ["Totally unrelated text"], "answer": "B"}'}],
    [{"content": "not valid json"}],
]
ground_truths = ["B", "B", "B"]
reference_steps = [
    ["Observe the traffic light", "The light is green", "Green means go"],
    ["Observe the traffic light", "The light is green", "Green means go"],
    ["Observe the traffic light", "The light is green", "Green means go"],
]

print("format       :", format_reward(completions))
print("reasoning    :", word_overlap_reasoning_reward(completions, reference_steps=reference_steps))
print("CPR          :", causal_process_reward(completions, ground_truths=ground_truths,
                                              reference_steps=reference_steps))

# Same thing via the registry (how the trainer selects rewards by name):
cpr = select_reward_func("reasoning_causal")
print("CPR via name :", cpr(completions, ground_truths=ground_truths, reference_steps=reference_steps))
