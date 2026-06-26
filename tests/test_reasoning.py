"""Unit tests for Match F1 / Ordered Match F1 logic (no model download)."""

import numpy as np
import pytest

from crystal_metrics.reasoning import MLLMReasoningEvaluator
from conftest import make_model_free_evaluator


# ---------------------------------------------------------------------------
# Controlled-embedding helper: each label maps to one axis of a one-hot space,
# so identical labels have cosine 1.0 and different labels have cosine 0.0.
# This makes Match F1 / Precision / Recall exactly predictable.
# ---------------------------------------------------------------------------
def onehot_embed(label_to_axis, dim):
    def _embed(texts):
        if not texts:
            return np.array([])
        out = np.zeros((len(texts), dim))
        for i, t in enumerate(texts):
            out[i, label_to_axis[t]] = 1.0
        return out
    return _embed


def test_perfect_match():
    labels = {"a": 0, "b": 1, "c": 2}
    ev = make_model_free_evaluator(threshold=0.5, embed_fn=onehot_embed(labels, 3))
    m = ev.evaluate_single(["a", "b", "c"], ["a", "b", "c"])
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.match_f1 == 1.0
    assert m.num_matched_predictions == 3


def test_partial_match_precision_recall():
    # 3 predicted, 4 reference; only a,b,c overlap -> 3 matches.
    labels = {"a": 0, "b": 1, "c": 2, "d": 3, "x": 4, "y": 5}
    ev = make_model_free_evaluator(threshold=0.5, embed_fn=onehot_embed(labels, 6))
    m = ev.evaluate_single(["a", "b", "x"], ["a", "b", "c", "d"])
    # matched preds: a,b -> 2/3 precision ; matched refs: a,b -> 2/4 recall
    assert m.precision == pytest.approx(2 / 3)
    assert m.recall == pytest.approx(2 / 4)
    assert m.match_f1 == pytest.approx(2 * (2 / 3 * 2 / 4) / (2 / 3 + 2 / 4))


def test_no_overlap_zero_f1():
    labels = {"a": 0, "b": 1, "x": 2, "y": 3}
    ev = make_model_free_evaluator(threshold=0.5, embed_fn=onehot_embed(labels, 4))
    m = ev.evaluate_single(["a", "b"], ["x", "y"])
    assert m.match_f1 == 0.0
    assert m.precision == 0.0
    assert m.recall == 0.0


def test_empty_predicted_returns_zeros():
    ev = make_model_free_evaluator(threshold=0.5)
    m = ev.evaluate_single([], ["a", "b"])
    assert m.match_f1 == 0.0
    assert m.num_reference_steps == 2
    assert m.num_predicted_steps == 0


def test_empty_reference_raises():
    ev = make_model_free_evaluator(threshold=0.5)
    with pytest.raises(ValueError):
        ev.evaluate_single(["a"], [])


# ---------------------------------------------------------------------------
# Ordering: Kendall's Tau and LIS via the static methods (no model at all).
# ---------------------------------------------------------------------------
def test_kendall_tau_perfect_and_reversed():
    perfect = [(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0)]
    reversed_ = [(2, 0, 1.0), (1, 1, 1.0), (0, 2, 1.0)]
    assert MLLMReasoningEvaluator._compute_kendall_tau(perfect) == 1.0
    assert MLLMReasoningEvaluator._compute_kendall_tau(reversed_) == -1.0


def test_lis_ratio():
    perfect = [(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0)]
    reversed_ = [(2, 0, 1.0), (1, 1, 1.0), (0, 2, 1.0)]
    assert MLLMReasoningEvaluator._compute_lis_ratio(perfect) == 1.0
    assert MLLMReasoningEvaluator._compute_lis_ratio(reversed_) == pytest.approx(1 / 3)


def test_fewer_than_two_matches_order_undefined():
    assert MLLMReasoningEvaluator._compute_kendall_tau([(0, 0, 1.0)]) == 1.0
    assert MLLMReasoningEvaluator._compute_lis_ratio([]) == 1.0


# ---------------------------------------------------------------------------
# Ordered Match F1 behavior.
# ---------------------------------------------------------------------------
def test_alpha_zero_ordered_equals_match_f1():
    labels = {"a": 0, "b": 1, "c": 2}
    ev = make_model_free_evaluator(threshold=0.5, embed_fn=onehot_embed(labels, 3))
    m = ev.evaluate_single(["a", "b", "c"], ["a", "b", "c"], alpha=0.0)
    assert m.ordered_match_f1 == m.match_f1


def test_disordered_chain_penalized():
    labels = {"a": 0, "b": 1, "c": 2}
    ev = make_model_free_evaluator(threshold=0.5, embed_fn=onehot_embed(labels, 3))
    ordered = ev.evaluate_single(["a", "b", "c"], ["a", "b", "c"], alpha=0.3)
    shuffled = ev.evaluate_single(["c", "b", "a"], ["a", "b", "c"], alpha=0.3)
    # Same Match F1 (same set matched) but reversed order -> lower Ordered F1.
    assert ordered.match_f1 == shuffled.match_f1 == 1.0
    assert shuffled.ordered_match_f1 < ordered.ordered_match_f1
    assert ordered.ordered_match_f1 == pytest.approx(1.0)


def test_lis_order_metric_selected():
    labels = {"a": 0, "b": 1, "c": 2}
    ev = make_model_free_evaluator(threshold=0.5, embed_fn=onehot_embed(labels, 3))
    m = ev.evaluate_single(["c", "b", "a"], ["a", "b", "c"], alpha=0.3, order_metric="lis")
    assert m.order_metric_used == "lis"
    assert m.order_score == pytest.approx(1 / 3)
