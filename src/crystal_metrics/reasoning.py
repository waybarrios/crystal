#!/usr/bin/env python3
"""
CRYSTAL reasoning metrics.

Match F1 measures the quality of step matching between predicted and reference
reasoning chains via semantic similarity:

- Precision = |matched_predictions| / |total_predictions|
- Recall    = |matched_references|  / |total_references|
- Match F1  = 2 * (P * R) / (P + R)

Ordered Match F1 additionally penalizes reasoning chains whose matched steps are
out of order, using either Kendall's Tau or the LIS (Longest Increasing
Subsequence) ratio, blended with `alpha`:

    Ordered_F1 = F1 * ((1 - alpha) + alpha * order_score)

References:
- Rajpurkar et al. (2016): SQuAD uses F1 for token-level answer matching
- Reimers & Gurevych (2019): Sentence-BERT for semantic similarity
"""

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


@dataclass
class EvaluationMetrics:
    """Container for per-sample reasoning metrics."""
    match_f1: float
    precision: float
    recall: float
    num_predicted_steps: int
    num_reference_steps: int
    num_matched_predictions: int
    num_matched_references: int
    avg_similarity: float
    max_similarity: float
    threshold_used: float
    # Ordered Match F1 fields (populated when alpha > 0)
    kendall_tau: float = 1.0
    tau_normalized: float = 1.0
    lis_ratio: float = 1.0
    order_score: float = 1.0  # normalized score from chosen order metric
    ordered_match_f1: float = 0.0
    alpha_used: float = 0.0
    order_metric_used: str = "none"


class MLLMReasoningEvaluator:
    """
    Evaluator for MLLM reasoning chains using the CRYSTAL Match F1 metric.

    The default embedding model and similarity threshold are the
    ablation-validated values from the CRYSTAL paper (Section 4.3):
    ``all-distilroberta-v1`` with threshold tau = 0.35.
    """

    def __init__(
        self,
        model_name: str = "all-distilroberta-v1",
        similarity_threshold: Optional[float] = None,
        device: Optional[str] = None,
        debug_mode: bool = False,
    ):
        """
        Args:
            model_name: Sentence-transformer model to use for step embeddings.
            similarity_threshold: Custom cosine threshold (None = model default).
            device: 'auto', 'cuda', or 'cpu'. None/'auto' picks CUDA if available.
            debug_mode: Print per-match debug information.
        """
        import torch
        from sentence_transformers import SentenceTransformer

        if device is None or device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if debug_mode:
            print(f"Initializing CRYSTAL reasoning evaluator on device: {self.device}")

        import os
        cache_folder = os.path.expanduser("~/.cache/sentence_transformers")

        self.model = SentenceTransformer(
            model_name,
            device=self.device,
            cache_folder=cache_folder,
        )
        self.model.eval()
        if hasattr(self.model, "_first_module"):
            self.model._first_module().to(self.device)
        self.model_name = model_name
        self.debug_mode = debug_mode

        # Model-specific optimized thresholds (empirically determined).
        self.model_thresholds = {
            "all-MiniLM-L6-v2": 0.35,
            "all-MiniLM-L12-v2": 0.37,
            "all-mpnet-base-v2": 0.38,
            "all-distilroberta-v1": 0.35,  # Ablation-validated (tau=0.35, Paper Section 4.3)
            "paraphrase-multilingual-MiniLM-L12-v2": 0.33,
            "paraphrase-multilingual-mpnet-base-v2": 0.35,
        }

        if similarity_threshold is not None:
            self.similarity_threshold = similarity_threshold
        else:
            self.similarity_threshold = self.model_thresholds.get(model_name, 0.45)

        if debug_mode:
            print(f"Model: {model_name}")
            print(f"Similarity threshold: {self.similarity_threshold}")

    def _compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """Compute sentence embeddings for a list of texts."""
        if not texts:
            return np.array([])
        return self.model.encode(texts, convert_to_tensor=False, show_progress_bar=False)

    def _compute_similarity_matrix(
        self, embeddings1: np.ndarray, embeddings2: np.ndarray
    ) -> np.ndarray:
        """Cosine similarity matrix: sim(A, B) = (A . B) / (||A|| ||B||)."""
        if embeddings1.size == 0 or embeddings2.size == 0:
            return np.array([[]])

        embeddings1_norm = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
        embeddings2_norm = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
        return np.dot(embeddings1_norm, embeddings2_norm.T)

    def _find_matches(
        self, similarity_matrix: np.ndarray, threshold: float
    ) -> Tuple[set, set, List[Tuple[int, int, float]]]:
        """
        Find a greedy 1:1 matching between predicted and reference steps.

        1. Collect all pairs with similarity > threshold.
        2. Sort by descending similarity.
        3. Assign greedily, never double-assigning a step.
        """
        matched_refs = set()
        matched_preds = set()
        match_pairs = []

        similarities = []
        for i in range(similarity_matrix.shape[0]):
            for j in range(similarity_matrix.shape[1]):
                if similarity_matrix[i, j] > threshold:
                    similarities.append((similarity_matrix[i, j], i, j))

        similarities.sort(reverse=True)

        for sim, pred_idx, ref_idx in similarities:
            if pred_idx not in matched_preds and ref_idx not in matched_refs:
                matched_preds.add(pred_idx)
                matched_refs.add(ref_idx)
                match_pairs.append((pred_idx, ref_idx, sim))
                if self.debug_mode:
                    print(f"Match: P{pred_idx} <-> R{ref_idx} (sim: {sim:.3f})")

        return matched_preds, matched_refs, match_pairs

    @staticmethod
    def _compute_kendall_tau(match_pairs: List[Tuple[int, int, float]]) -> float:
        """
        Kendall's Tau over matched pairs, measuring order preservation.

        Returns tau in [-1, 1]: +1 perfect order, 0 random, -1 reversed.
        Returns 1.0 if fewer than 2 matches (order undefined).
        """
        if len(match_pairs) < 2:
            return 1.0

        sorted_by_ref = sorted(match_pairs, key=lambda x: x[1])
        pred_indices = [p[0] for p in sorted_by_ref]

        k = len(pred_indices)
        concordant = 0
        discordant = 0
        for i in range(k):
            for j in range(i + 1, k):
                if pred_indices[i] < pred_indices[j]:
                    concordant += 1
                elif pred_indices[i] > pred_indices[j]:
                    discordant += 1

        total_pairs = k * (k - 1) / 2
        if total_pairs == 0:
            return 1.0
        return (concordant - discordant) / total_pairs

    @staticmethod
    def _compute_lis_ratio(match_pairs: List[Tuple[int, int, float]]) -> float:
        """
        LIS (Longest Increasing Subsequence) ratio over matched pairs.

        Fraction of matched steps that lie in the correct relative order.
        Returns ratio in [0, 1]; 1.0 if fewer than 2 matches.
        """
        if len(match_pairs) < 2:
            return 1.0

        sorted_by_ref = sorted(match_pairs, key=lambda x: x[1])
        pred_indices = [p[0] for p in sorted_by_ref]

        from bisect import bisect_left
        tails = []
        for val in pred_indices:
            pos = bisect_left(tails, val)
            if pos == len(tails):
                tails.append(val)
            else:
                tails[pos] = val

        return len(tails) / len(pred_indices)

    def evaluate_single(
        self,
        predicted_steps: List[str],
        reference_steps: List[str],
        verbose: bool = None,
        alpha: float = 0.0,
        order_metric: str = "kendall_tau",
    ) -> EvaluationMetrics:
        """
        Evaluate one sample.

        Args:
            predicted_steps: Predicted reasoning steps.
            reference_steps: Reference / ground-truth reasoning steps.
            verbose: Print debug info (defaults to debug_mode).
            alpha: Order sensitivity in [0, 1]. 0 = ignore order; 0.3 recommended.
            order_metric: "kendall_tau" or "lis".

        Returns:
            EvaluationMetrics with Match F1 and (when alpha > 0) Ordered Match F1.
        """
        if verbose is None:
            verbose = self.debug_mode

        if not reference_steps:
            raise ValueError("Reference steps cannot be empty")

        if not predicted_steps:
            return EvaluationMetrics(
                match_f1=0.0,
                precision=0.0,
                recall=0.0,
                num_predicted_steps=0,
                num_reference_steps=len(reference_steps),
                num_matched_predictions=0,
                num_matched_references=0,
                avg_similarity=0.0,
                max_similarity=0.0,
                threshold_used=self.similarity_threshold,
                alpha_used=alpha,
                order_metric_used=order_metric,
            )

        pred_embeddings = self._compute_embeddings(predicted_steps)
        ref_embeddings = self._compute_embeddings(reference_steps)

        similarity_matrix = self._compute_similarity_matrix(pred_embeddings, ref_embeddings)

        matched_preds, matched_refs, match_pairs = self._find_matches(
            similarity_matrix, self.similarity_threshold
        )

        n_predicted = len(predicted_steps)
        n_reference = len(reference_steps)
        n_matched_preds = len(matched_preds)
        n_matched_refs = len(matched_refs)

        precision = n_matched_preds / n_predicted if n_predicted > 0 else 0.0
        recall = n_matched_refs / n_reference if n_reference > 0 else 0.0
        match_f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        avg_similarity = np.mean(similarity_matrix) if similarity_matrix.size > 0 else 0.0
        max_similarity = np.max(similarity_matrix) if similarity_matrix.size > 0 else 0.0

        tau = self._compute_kendall_tau(match_pairs)
        tau_norm = (tau + 1.0) / 2.0
        lis = self._compute_lis_ratio(match_pairs)

        if order_metric == "lis":
            order_score = lis
        else:  # kendall_tau
            order_score = tau_norm

        ordered_f1 = (
            match_f1 * ((1.0 - alpha) + alpha * order_score) if alpha > 0 else match_f1
        )

        if verbose:
            print("\nEvaluation Results:")
            print(f"  Precision: {precision:.3f} ({n_matched_preds}/{n_predicted})")
            print(f"  Recall: {recall:.3f} ({n_matched_refs}/{n_reference})")
            print(f"  Match F1: {match_f1:.3f}")
            if alpha > 0:
                print(f"  Kendall's Tau: {tau:.3f} (normalized: {tau_norm:.3f})")
                print(f"  LIS ratio: {lis:.3f}")
                print(f"  Order metric: {order_metric} (score: {order_score:.3f})")
                print(f"  Ordered Match F1 (alpha={alpha}): {ordered_f1:.3f}")

        return EvaluationMetrics(
            match_f1=match_f1,
            precision=precision,
            recall=recall,
            num_predicted_steps=n_predicted,
            num_reference_steps=n_reference,
            num_matched_predictions=n_matched_preds,
            num_matched_references=n_matched_refs,
            avg_similarity=avg_similarity,
            max_similarity=max_similarity,
            threshold_used=self.similarity_threshold,
            kendall_tau=tau,
            tau_normalized=tau_norm,
            lis_ratio=lis,
            order_score=order_score,
            ordered_match_f1=ordered_f1,
            alpha_used=alpha,
            order_metric_used=order_metric,
        )

    def evaluate_dataset(
        self,
        predictions: Dict[int, Dict],
        ground_truth: Dict[int, Dict],
        verbose: bool = False,
        alpha: float = 0.0,
        order_metric: str = "kendall_tau",
    ) -> pd.DataFrame:
        """
        Evaluate a full dataset.

        Args:
            predictions: {id: {"reasoning_steps": [...], "answer": "..."}}
            ground_truth: {id: {"reference_steps": [...]}}
            verbose: Show a progress bar.
            alpha: Order sensitivity passed through to evaluate_single.
            order_metric: "kendall_tau" or "lis".

        Returns:
            DataFrame with one row per evaluated sample.
        """
        pred_indices = set(predictions.keys())
        gt_indices = set(ground_truth.keys())

        if pred_indices != gt_indices:
            missing_pred = gt_indices - pred_indices
            missing_gt = pred_indices - gt_indices
            if missing_pred:
                print(f"Warning: Missing predictions for indices: {missing_pred}")
            if missing_gt:
                print(f"Warning: Missing ground truth for indices: {missing_gt}")

        common_indices = pred_indices.intersection(gt_indices)
        print(f"Evaluating {len(common_indices)} samples...")

        results = []
        for idx in tqdm(common_indices, disable=not verbose):
            try:
                pred_steps = predictions[idx]["reasoning_steps"]
                ref_steps = ground_truth[idx]["reference_steps"]
                answer = predictions[idx].get("answer", "")

                metrics = self.evaluate_single(
                    pred_steps, ref_steps, verbose=False,
                    alpha=alpha, order_metric=order_metric,
                )

                results.append({
                    "sample_idx": idx,
                    "answer": answer,
                    "match_f1": metrics.match_f1,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "ordered_match_f1": metrics.ordered_match_f1,
                    "kendall_tau": metrics.kendall_tau,
                    "lis_ratio": metrics.lis_ratio,
                    "num_predicted_steps": metrics.num_predicted_steps,
                    "num_reference_steps": metrics.num_reference_steps,
                    "num_matched_predictions": metrics.num_matched_predictions,
                    "num_matched_references": metrics.num_matched_references,
                    "avg_similarity": metrics.avg_similarity,
                    "max_similarity": metrics.max_similarity,
                    "threshold_used": metrics.threshold_used,
                })
            except Exception as e:
                print(f"Error evaluating sample {idx}: {e}")
                continue

        df = pd.DataFrame(results)

        if len(df) > 0:
            print("\n=== Evaluation Summary ===")
            print(f"Samples evaluated: {len(df)}")
            print(f"Model: {self.model_name} (threshold: {self.similarity_threshold:.3f})")
            print(f"Average Match F1: {df['match_f1'].mean():.3f} (+/-{df['match_f1'].std():.3f})")
            print(f"Average Precision: {df['precision'].mean():.3f}")
            print(f"Average Recall: {df['recall'].mean():.3f}")
            if alpha > 0:
                print(f"Average Ordered Match F1 (alpha={alpha}): {df['ordered_match_f1'].mean():.3f}")

        return df

    def generate_summary(self, results_df: pd.DataFrame) -> Dict:
        """Statistical summary of an evaluation DataFrame."""
        if len(results_df) == 0:
            return {}

        return {
            "total_samples": len(results_df),
            "model_name": self.model_name,
            "similarity_threshold": self.similarity_threshold,
            "avg_match_f1": results_df["match_f1"].mean(),
            "std_match_f1": results_df["match_f1"].std(),
            "median_match_f1": results_df["match_f1"].median(),
            "avg_precision": results_df["precision"].mean(),
            "avg_recall": results_df["recall"].mean(),
            "top_10_percent_threshold": results_df["match_f1"].quantile(0.9),
            "bottom_10_percent_threshold": results_df["match_f1"].quantile(0.1),
        }


def load_json_data(file_path: str) -> Dict:
    """Load data from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_results(results_df: pd.DataFrame, output_path: str):
    """Save evaluation results to a CSV file."""
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")
