"""Shared test helpers for crystal-metrics."""

import hashlib

import numpy as np

from crystal_metrics.reasoning import MLLMReasoningEvaluator


def deterministic_embeddings(texts, dim: int = 32):
    """
    Map texts to fixed unit vectors deterministically (no model needed).

    Identical strings -> identical vectors (cosine 1.0). Uses SHA-256 (not the
    salted built-in hash) so results are stable across processes and runs.
    """
    if not texts:
        return np.array([])
    vectors = []
    for t in texts:
        seed = int.from_bytes(hashlib.sha256(t.encode("utf-8")).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(dim)
        vectors.append(v / (np.linalg.norm(v) + 1e-12))
    return np.vstack(vectors)


def make_model_free_evaluator(threshold: float = 0.35, embed_fn=None):
    """
    Build an MLLMReasoningEvaluator WITHOUT loading a SentenceTransformer.

    Bypasses __init__ (which would download a model) and patches
    ``_compute_embeddings`` with a deterministic function. The matching, F1, and
    ordering logic under test are untouched.
    """
    ev = MLLMReasoningEvaluator.__new__(MLLMReasoningEvaluator)
    ev.similarity_threshold = threshold
    ev.debug_mode = False
    ev.model_name = "test-deterministic"
    ev._compute_embeddings = embed_fn or (lambda texts: deterministic_embeddings(texts))
    return ev
