"""
crystal-metrics: transparent multimodal reasoning metrics from the CRYSTAL benchmark.

Core metrics (no LLM required):
- Match F1, Precision, Recall      -> MLLMReasoningEvaluator
- Ordered Match F1 (Kendall / LIS) -> MLLMReasoningEvaluator(..., alpha=...)
- Multi-format Accuracy            -> AccuracyCalculator

The optional LLM judge lives in ``crystal_metrics.judge`` and needs the
``[judge]`` extra: ``pip install crystal-metrics[judge]``. It is intentionally
NOT imported here so the core package stays free of the ``openai`` dependency.
"""

from .accuracy import AccuracyCalculator, AccuracyResult, AnswerNormalizer
from .reasoning import (
    EvaluationMetrics,
    MLLMReasoningEvaluator,
    load_json_data,
    save_results,
)
from .rewards import (
    accuracy_reward,
    causal_process_reward,
    format_reward,
    parse_reasoning_steps,
    select_reward_func,
    semantic_reasoning_reward,
    word_overlap_reasoning_reward,
)
from .similarity import (
    best_match_f1,
    jaccard_similarity,
    semantic_match_f1,
    word_overlap_similarity,
)

__version__ = "0.2.0"

__all__ = [
    "MLLMReasoningEvaluator",
    "EvaluationMetrics",
    "AccuracyCalculator",
    "AccuracyResult",
    "AnswerNormalizer",
    "best_match_f1",
    "semantic_match_f1",
    "jaccard_similarity",
    "word_overlap_similarity",
    "load_json_data",
    "save_results",
    "format_reward",
    "accuracy_reward",
    "word_overlap_reasoning_reward",
    "semantic_reasoning_reward",
    "causal_process_reward",
    "parse_reasoning_steps",
    "select_reward_func",
    "__version__",
]
