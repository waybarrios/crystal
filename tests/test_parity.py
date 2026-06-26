"""
Parity tests: the ported `crystal_metrics` must produce EXACTLY the same numbers
as the original VLM-R1 `mllm_evaluator`.

Strategy:
- Reasoning: feed BOTH implementations identical deterministic embeddings (no
  model download), so any divergence is a logic difference, not embedding noise.
  Compare every field of EvaluationMetrics across many inputs and settings.
- Accuracy: compare the rule-based judgments case by case.
- A real-model end-to-end parity check runs too, skipped if the model can't load.

Point the suite at a checkout of VLM-R1 via CRYSTAL_VLMR1_PATH
(default: /gpudata3/Wayner/VLM-R1). The whole module is skipped if it is absent.
"""

import dataclasses
import importlib
import os
import sys

import numpy as np
import pytest

from conftest import deterministic_embeddings, make_model_free_evaluator

VLMR1_PATH = os.environ.get("CRYSTAL_VLMR1_PATH", "/gpudata3/Wayner/VLM-R1")


def _import_original():
    """Import the original VLM-R1 mllm_evaluator package, or skip the module."""
    if not os.path.isdir(os.path.join(VLMR1_PATH, "mllm_evaluator")):
        pytest.skip(f"Original VLM-R1 not found at {VLMR1_PATH} (set CRYSTAL_VLMR1_PATH)")
    if VLMR1_PATH not in sys.path:
        sys.path.insert(0, VLMR1_PATH)
    try:
        return importlib.import_module("mllm_evaluator.mllm_evaluator")
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"Could not import original mllm_evaluator: {e}")


# Test inputs covering: perfect match, partial, no overlap, ordered, reversed,
# duplicates, and size asymmetry.
STEP_CASES = [
    (["a", "b", "c"], ["a", "b", "c"]),
    (["a", "b", "x"], ["a", "b", "c", "d"]),
    (["c", "b", "a"], ["a", "b", "c"]),
    (["a"], ["a", "b", "c", "d", "e"]),
    (["a", "b", "c", "d", "e"], ["a"]),
    (["a", "a", "b"], ["a", "b", "c"]),
    (["p", "q", "r"], ["a", "b", "c"]),
    (["a", "c", "b", "d"], ["a", "b", "c", "d"]),
]

SETTINGS = [
    {"alpha": 0.0, "order_metric": "kendall_tau"},
    {"alpha": 0.3, "order_metric": "kendall_tau"},
    {"alpha": 0.3, "order_metric": "lis"},
    {"alpha": 0.5, "order_metric": "lis"},
]


def _make_original_model_free(orig_module, threshold, embed_fn):
    cls = orig_module.MLLMReasoningEvaluator
    ev = cls.__new__(cls)
    ev.similarity_threshold = threshold
    ev.debug_mode = False
    ev.model_name = "test-deterministic"
    ev._compute_embeddings = embed_fn
    return ev


@pytest.mark.parametrize("pred,ref", STEP_CASES)
@pytest.mark.parametrize("settings", SETTINGS)
def test_reasoning_parity_deterministic(pred, ref, settings):
    orig_module = _import_original()

    embed_fn = lambda texts: deterministic_embeddings(texts)
    new_ev = make_model_free_evaluator(threshold=0.35, embed_fn=embed_fn)
    old_ev = _make_original_model_free(orig_module, 0.35, embed_fn)

    new_m = new_ev.evaluate_single(pred, ref, **settings)
    old_m = old_ev.evaluate_single(pred, ref, **settings)

    fields = [f.name for f in dataclasses.fields(new_m)]
    for name in fields:
        nv = getattr(new_m, name)
        ov = getattr(old_m, name)
        if isinstance(nv, float):
            assert nv == pytest.approx(ov, rel=1e-9, abs=1e-12), f"field {name}: {nv} != {ov}"
        else:
            assert nv == ov, f"field {name}: {nv} != {ov}"


ACCURACY_CASES = [
    ("How many?", "5", "5.0"),
    ("Value?", "0.67", "0.67234572345763"),
    ("How many?", "5", "42"),
    ("Is it green?", "Yes", "yes"),
    ("Is it green?", "Yes", "No"),
    ("What color?", "A", "A"),
    ("What color?", "(A)", "A"),
    ("Capital?\nA) London\nB) Paris", "b) Paris", "B"),
    ("Q", "Paris", "paris"),
    ("Q", "London", "Paris"),
    ("Q", "the capital of france is paris", "capital of france is paris"),
]


@pytest.mark.parametrize("question,pred,gt", ACCURACY_CASES)
def test_accuracy_parity(question, pred, gt):
    if not os.path.isdir(os.path.join(VLMR1_PATH, "mllm_evaluator")):
        pytest.skip(f"Original VLM-R1 not found at {VLMR1_PATH}")
    if VLMR1_PATH not in sys.path:
        sys.path.insert(0, VLMR1_PATH)
    try:
        orig_acc = importlib.import_module("mllm_evaluator.accuracy_calculator")
    except Exception as e:  # pragma: no cover
        pytest.skip(f"Could not import original accuracy_calculator: {e}")

    from crystal_metrics.accuracy import AccuracyCalculator as NewCalc

    new_calc = NewCalc(use_llm_grader=False)
    old_calc = orig_acc.AccuracyCalculator(use_llm_grader=False)

    new_r = new_calc.evaluate_single(question, pred, gt)
    old_r = old_calc.evaluate_single(question, pred, gt)

    assert new_r.is_correct == old_r.is_correct
    assert new_r.match_type == old_r.match_type
    assert new_r.confidence == pytest.approx(old_r.confidence)
    assert new_r.normalized_prediction == old_r.normalized_prediction
    assert new_r.normalized_ground_truth == old_r.normalized_ground_truth


def test_reasoning_parity_real_model():
    """End-to-end parity using the real ablation-validated embedding model."""
    orig_module = _import_original()
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-distilroberta-v1", device="cpu")
        model.eval()
    except Exception as e:  # offline / no weights
        pytest.skip(f"Could not load all-distilroberta-v1: {e}")

    def real_embed(texts):
        if not texts:
            return np.array([])
        return model.encode(texts, convert_to_tensor=False, show_progress_bar=False)

    new_ev = make_model_free_evaluator(threshold=0.35, embed_fn=real_embed)
    old_ev = _make_original_model_free(orig_module, 0.35, real_embed)

    pred = ["Three objects are on a table", "The middle one is the smallest", "The answer is C"]
    ref = ["There are three objects in the image", "Compare the sizes", "The middle object is smallest", "Select option C"]

    new_m = new_ev.evaluate_single(pred, ref, alpha=0.3)
    old_m = old_ev.evaluate_single(pred, ref, alpha=0.3)

    assert new_m.match_f1 == pytest.approx(old_m.match_f1, rel=1e-9, abs=1e-9)
    assert new_m.precision == pytest.approx(old_m.precision)
    assert new_m.recall == pytest.approx(old_m.recall)
    assert new_m.ordered_match_f1 == pytest.approx(old_m.ordered_match_f1, rel=1e-9, abs=1e-9)
