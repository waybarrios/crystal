"""
Lightweight text-similarity helpers (pure Python, no heavy deps).

These power the deterministic, GPU-free matching fallbacks and are useful for
quick experiments. The semantic variant lazily loads a SentenceTransformer with
a process-local cache for multi-process (e.g. DeepSpeed) safety.
"""

import re
from collections import Counter
from typing import Any, List, Tuple

import numpy as np


def tokenize(text: str) -> List[str]:
    """Simple word tokenization."""
    return re.findall(r"\w+", text.lower())


def jaccard_similarity(text1: str, text2: str) -> float:
    """Jaccard similarity: |intersection| / |union|. Deterministic."""
    tokens1 = set(tokenize(text1))
    tokens2 = set(tokenize(text2))
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return intersection / union if union > 0 else 0.0


def word_overlap_similarity(text1: str, text2: str) -> float:
    """Common-word count divided by the geometric mean of token lengths."""
    tokens1 = tokenize(text1)
    tokens2 = tokenize(text2)
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    counter1 = Counter(tokens1)
    counter2 = Counter(tokens2)
    common = sum((counter1 & counter2).values())
    denom = (len(tokens1) * len(tokens2)) ** 0.5
    return common / denom if denom > 0 else 0.0


# Process-local SentenceTransformer cache (avoids multi-process tensor issues).
_semantic_model_cache = {}


def _get_semantic_model():
    """Lazily build a process-local CPU SentenceTransformer."""
    import os

    pid = os.getpid()
    if pid not in _semantic_model_cache:
        try:
            from sentence_transformers import SentenceTransformer

            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            model = SentenceTransformer("all-distilroberta-v1", device="cpu")
            model.eval()
            for param in model.parameters():
                if param.device.type != "cpu":
                    param.data = param.data.cpu()
            _semantic_model_cache[pid] = model
        except Exception as e:
            print(f"semantic_match_f1: model init failed for PID {pid}: {e}")
            _semantic_model_cache[pid] = None
    return _semantic_model_cache.get(pid)


def semantic_match_f1(
    predicted_steps: List[str],
    reference_steps: List[str],
    model: Any = None,
    threshold: float = 0.70,
) -> Tuple[float, int, int]:
    """
    F1 of step matching using semantic (SentenceTransformer + cosine) similarity.

    Returns (f1_score, matched_predictions, matched_references).
    """
    if not predicted_steps or not reference_steps:
        return 0.0, 0, 0

    if model is None:
        model = _get_semantic_model()
    if model is None:
        raise RuntimeError("SentenceTransformer not available")

    try:
        pred_embeddings = model.encode(predicted_steps, convert_to_tensor=False, show_progress_bar=False)
        ref_embeddings = model.encode(reference_steps, convert_to_tensor=False, show_progress_bar=False)
    except Exception as e:
        import os

        pid = os.getpid()
        _semantic_model_cache.pop(pid, None)
        model = _get_semantic_model()
        if model is None:
            raise RuntimeError(f"SentenceTransformer encoding failed: {e}")
        pred_embeddings = model.encode(predicted_steps, convert_to_tensor=False, show_progress_bar=False)
        ref_embeddings = model.encode(reference_steps, convert_to_tensor=False, show_progress_bar=False)

    pred_norm = pred_embeddings / (np.linalg.norm(pred_embeddings, axis=1, keepdims=True) + 1e-8)
    ref_norm = ref_embeddings / (np.linalg.norm(ref_embeddings, axis=1, keepdims=True) + 1e-8)
    similarity_matrix = np.dot(pred_norm, ref_norm.T)

    return _greedy_f1(similarity_matrix, len(predicted_steps), len(reference_steps), threshold)


def best_match_f1(
    predicted_steps: List[str],
    reference_steps: List[str],
    threshold: float = 0.3,
) -> Tuple[float, int, int]:
    """F1 of step matching using word-overlap similarity (no model needed)."""
    if not predicted_steps or not reference_steps:
        return 0.0, 0, 0

    n_pred = len(predicted_steps)
    n_ref = len(reference_steps)
    similarity_matrix = np.zeros((n_pred, n_ref))
    for i, pred in enumerate(predicted_steps):
        for j, ref in enumerate(reference_steps):
            similarity_matrix[i, j] = word_overlap_similarity(pred, ref)

    return _greedy_f1(similarity_matrix, n_pred, n_ref, threshold)


def _greedy_f1(
    similarity_matrix: np.ndarray, n_pred: int, n_ref: int, threshold: float
) -> Tuple[float, int, int]:
    """Greedy 1:1 matching above threshold, returning (f1, n_matched_pred, n_matched_ref)."""
    similarities = []
    for i in range(n_pred):
        for j in range(n_ref):
            if similarity_matrix[i, j] > threshold:
                similarities.append((similarity_matrix[i, j], i, j))
    similarities.sort(reverse=True)

    matched_preds = set()
    matched_refs = set()
    for _, pred_idx, ref_idx in similarities:
        if pred_idx not in matched_preds and ref_idx not in matched_refs:
            matched_preds.add(pred_idx)
            matched_refs.add(ref_idx)

    n_matched_pred = len(matched_preds)
    n_matched_ref = len(matched_refs)
    precision = n_matched_pred / n_pred if n_pred > 0 else 0.0
    recall = n_matched_ref / n_ref if n_ref > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1, n_matched_pred, n_matched_ref
