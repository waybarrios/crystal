#!/usr/bin/env python3
"""
Model-agnostic GRPO reward adapter for CRYSTAL.

The CRYSTAL reward logic (CPR / SPR / word-overlap / format / accuracy) lives in
the pip-installable ``crystal_metrics.rewards`` module. This adapter exposes those
rewards in the shape a TRL-style ``GRPOTrainer`` expects — a list of callables,
each ``(completions, **kwargs) -> list[float]`` — and is **model-agnostic**: it
only reads the generated text and the dataset fields (``reference_steps``,
``solution``), so the same rewards train Qwen2.5-VL, InternVL, GLM-4V, or any
other VLM the base trainer supports.

Usage in a training entry point (e.g. open-r1-multimodal's ``grpo_rec.py``)::

    from rewards_grpo import build_reward_funcs
    reward_funcs = build_reward_funcs(
        names=["format", "accuracy", "reasoning_causal"],
        causal_answer_weight=0.6, causal_step_weight=0.4,
    )
    trainer = VLMGRPOTrainer(..., reward_funcs=reward_funcs, ...)

Requires: ``pip install crystal-metrics`` (and ``crystal-metrics[judge]`` only if
you enable the LLM judge for the accuracy reward).
"""

from functools import partial
from typing import Callable, List, Optional

from crystal_metrics.rewards import (
    accuracy_reward,
    causal_process_reward,
    format_reward,
    semantic_reasoning_reward,
    word_overlap_reasoning_reward,
)


def _accuracy(completions, **kwargs):
    # GRPOTrainer passes the ground-truth answers as `solution`.
    solution = kwargs.pop("solution", kwargs.pop("answer", []))
    return accuracy_reward(completions, solution=solution, **kwargs)


def build_reward_funcs(
    names: List[str],
    causal_answer_weight: float = 0.6,
    causal_step_weight: float = 0.4,
    semantic_threshold: float = 0.70,
    word_overlap_threshold: float = 0.45,
    use_llm_judge: bool = False,
) -> List[Callable]:
    """
    Build the list of reward callables for the GRPO trainer.

    Args:
        names: reward names in order, any of
            "format", "accuracy", "reasoning" (word overlap),
            "reasoning_semantic" (SPR), "reasoning_causal" (CPR).
        causal_answer_weight, causal_step_weight: CPR weights (paper 0.6 / 0.4).
        semantic_threshold: cosine threshold for SPR (paper 0.70).
        word_overlap_threshold: threshold for the word-overlap reasoning reward.
        use_llm_judge: enable the LLM judge inside the accuracy reward
            (needs crystal-metrics[judge]).

    Returns:
        List of callables aligned with ``names``.
    """
    registry = {
        "format": format_reward,
        "accuracy": partial(_accuracy, use_llm_grader=use_llm_judge),
        "reasoning": partial(word_overlap_reasoning_reward, threshold=word_overlap_threshold),
        "reasoning_semantic": partial(semantic_reasoning_reward, threshold=semantic_threshold),
        "reasoning_causal": partial(
            causal_process_reward,
            answer_weight=causal_answer_weight, step_weight=causal_step_weight,
        ),
    }
    funcs = []
    for name in names:
        if name not in registry:
            raise ValueError(f"Unknown reward {name!r}. Available: {sorted(registry)}")
        fn = registry[name]
        # Give each partial a readable __name__ for trainer logging.
        try:
            fn.__name__ = f"crystal_{name}_reward"
        except (AttributeError, TypeError):
            pass
        funcs.append(fn)
    return funcs


def resolve_reward_names(
    base_names: List[str],
    use_causal: bool = False,
    use_semantic: bool = False,
) -> List[str]:
    """
    Map a base reward list to concrete names, swapping the generic "reasoning"
    entry for CPR or SPR — mirroring the training scripts' flags so a launch
    script can pass ``--use_causal_reasoning_reward`` and get CPR.
    """
    replacement = "reasoning"
    if use_causal:
        replacement = "reasoning_causal"
    elif use_semantic:
        replacement = "reasoning_semantic"
    return [replacement if n == "reasoning" else n for n in base_names]
