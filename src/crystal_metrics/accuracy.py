#!/usr/bin/env python3
"""
Accuracy calculator for MLLM answers.

Handles multiple answer formats — yes/no, numeric (with tolerance), multiple
choice, and free-form text — using rule-based matching. Free-form text answers
optionally fall back to an LLM judge (see ``crystal_metrics.judge``), which is
imported lazily so the core install needs no LLM dependency.
"""

import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from tqdm import tqdm


@dataclass
class AccuracyResult:
    """Container for a single accuracy judgment."""
    is_correct: bool
    predicted_answer: str
    ground_truth_answer: str
    normalized_prediction: str
    normalized_ground_truth: str
    match_type: str  # 'exact', 'numeric_*', 'choice', 'yes_no', 'llm_verified'
    confidence: float = 1.0


class AnswerNormalizer:
    """Normalize and compare answers across formats."""

    @staticmethod
    def normalize_text(text: str) -> str:
        """Lowercase, collapse whitespace, normalize quotes, strip punctuation."""
        if not isinstance(text, str):
            text = str(text)
        text = text.strip().lower()

        # Normalize smart quotes/apostrophes to ASCII.
        text = text.replace("‘", "'").replace("’", "'")
        text = text.replace("“", '"').replace("”", '"')
        text = text.replace("`", "'")

        # Remove periods that are not decimal points, then other punctuation.
        text = re.sub(r"(?<!\d)\.(?!\d)", "", text)
        text = re.sub(r"""[,()"\[\]{}']""", "", text)
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def extract_number(text: str) -> Optional[float]:
        """Extract the first numeric value from text, if any."""
        if not isinstance(text, str):
            text = str(text)
        text = text.strip().lower()

        patterns = [
            r"the answer is:?\s*",
            r"the result is:?\s*",
            r"answer:?\s*",
            r"result:?\s*",
            r"approximately:?\s*",
            r"about:?\s*",
        ]
        for pattern in patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        matches = re.findall(r"-?\d+\.?\d*", text)
        if matches:
            try:
                return float(matches[0])
            except ValueError:
                pass
        return None

    @staticmethod
    def compare_numbers(
        predicted: float,
        ground_truth: float,
        relative_tolerance: float = 0.1,
        absolute_tolerance: float = 0.05,
        delta: float = 1e-10,
    ) -> Tuple[bool, str, float]:
        """
        Compare two numbers with flexible tolerance for rounding.

        Tolerance criterion (paper Eq. 2):
            |pred - gt| <= eps_abs  OR  |pred - gt| / max(|gt|, delta) <= eps_rel

        Returns:
            (is_correct, match_type, confidence)
        """
        if predicted == ground_truth:
            return True, "numeric_exact", 1.0

        abs_diff = abs(predicted - ground_truth)

        if abs_diff <= absolute_tolerance:
            return True, "numeric_rounded", 0.95

        denominator = max(abs(ground_truth), delta)
        rel_diff = abs_diff / denominator

        if rel_diff <= relative_tolerance:
            return True, "numeric_rounded", 0.9

        if abs_diff < 0.01:
            return True, "numeric_rounded", 1.0

        if ground_truth != 0:
            pred_str = str(predicted).rstrip("0").rstrip(".")
            if "." in pred_str:
                pred_decimals = len(pred_str.split(".")[1])
            else:
                pred_decimals = 0

            if pred_decimals <= 2:
                if rel_diff < 0.15:
                    return True, "numeric_rounded", 0.85
                rounded_gt = round(ground_truth, pred_decimals)
                if abs(predicted - rounded_gt) < 1e-6:
                    return True, "numeric_rounded", 0.95

        return False, "numeric_mismatch", 0.0

    @staticmethod
    def extract_choice(text: str, problem_text: Optional[str] = None) -> Optional[str]:
        """Extract a multiple-choice letter (A, B, C, ...) from an answer."""
        text_normalized = AnswerNormalizer.normalize_text(text)

        # Case 1: very short answer that is exactly a single letter.
        if len(text_normalized) <= 3:
            match = re.search(r"^([a-z])$", text_normalized)
            if match:
                return match.group(1).upper()

        # Case 2: extract the letter from common patterns.
        patterns = [
            r"^([a-z])\)",
            r"^\(([a-z])\)",
            r"^([a-z])\.",
            r"option\s+([a-z])",
            r"choice\s+([a-z])",
            r"answer\s+is\s+([a-z])",
            r"\(([a-z])\)",
            r"\b([a-z])\)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text_normalized)
            if match:
                return match.group(1).upper()

        # Case 3: match full answer text against the options in the problem.
        if problem_text:
            return AnswerNormalizer.match_answer_to_option(text, problem_text)

        return None

    @staticmethod
    def match_answer_to_option(answer_text: str, problem_text: str) -> Optional[str]:
        """Match a full answer text to its option letter using the problem text."""
        answer_normalized = AnswerNormalizer.normalize_text(answer_text)

        option_pattern = r"([A-Z])\)\s*([^\n]+)"
        matches = re.findall(option_pattern, problem_text)
        if not matches:
            return None

        for letter, option_text in matches:
            option_normalized = AnswerNormalizer.normalize_text(option_text)
            if answer_normalized == option_normalized:
                return letter
            if answer_normalized in option_normalized or option_normalized in answer_normalized:
                shorter_len = min(len(answer_normalized), len(option_normalized))
                longer_len = max(len(answer_normalized), len(option_normalized))
                if longer_len > 0 and shorter_len / longer_len >= 0.8:
                    return letter

        return None

    @staticmethod
    def is_yes_no_question(text: str) -> bool:
        """Whether the (normalized) answer is a yes/no/true/false token."""
        text = AnswerNormalizer.normalize_text(text)
        return text in ["yes", "no", "true", "false"]

    @staticmethod
    def normalize_yes_no(text: str) -> str:
        """Normalize yes/no variants to 'yes' or 'no'."""
        text = AnswerNormalizer.normalize_text(text)
        yes_patterns = ["yes", "true", "correct", "affirmative"]
        no_patterns = ["no", "false", "incorrect", "negative"]
        for pattern in yes_patterns:
            if pattern in text:
                return "yes"
        for pattern in no_patterns:
            if pattern in text:
                return "no"
        return text


class AccuracyCalculator:
    """Calculate multi-format answer accuracy for MLLM predictions."""

    def __init__(
        self,
        use_llm_grader: bool = False,
        llm_model: str = "llama3.2",
        base_url: str = "http://localhost:11434/v1",
        numeric_relative_tolerance: float = 0.1,
        numeric_absolute_tolerance: float = 0.05,
        delta: float = 1e-10,
    ):
        """
        Args:
            use_llm_grader: Use an LLM judge for long free-form text answers.
                Requires the ``[judge]`` extra. Defaults to False (rule-based only).
            llm_model: Model name for the judge.
            base_url: OpenAI-compatible endpoint for the judge.
            numeric_relative_tolerance: Relative tolerance for numeric matching.
            numeric_absolute_tolerance: Absolute tolerance for numeric matching.
            delta: Small value to avoid division by zero.
        """
        self.normalizer = AnswerNormalizer()
        self.use_llm_grader = use_llm_grader
        if use_llm_grader:
            from .judge import LLMGrader  # lazy: only needs openai when enabled
            self.llm_grader = LLMGrader(model=llm_model, base_url=base_url)
        else:
            self.llm_grader = None
        self.numeric_relative_tolerance = numeric_relative_tolerance
        self.numeric_absolute_tolerance = numeric_absolute_tolerance
        self.delta = delta

    def evaluate_single(
        self, question: str, predicted_answer: str, ground_truth_answer: str
    ) -> AccuracyResult:
        """
        Evaluate one prediction against ground truth.

        Resolution order: numeric (with tolerance) -> yes/no -> multiple choice
        -> exact text -> LLM judge (if enabled) -> substring fallback.
        """
        pred_norm = self.normalizer.normalize_text(predicted_answer)
        gt_norm = self.normalizer.normalize_text(ground_truth_answer)

        # 1. Numeric matching with tolerance.
        pred_num = self.normalizer.extract_number(predicted_answer)
        gt_num = self.normalizer.extract_number(ground_truth_answer)
        if pred_num is not None and gt_num is not None:
            is_correct, match_type, confidence = self.normalizer.compare_numbers(
                pred_num, gt_num,
                relative_tolerance=self.numeric_relative_tolerance,
                absolute_tolerance=self.numeric_absolute_tolerance,
                delta=self.delta,
            )
            return AccuracyResult(
                is_correct=is_correct,
                predicted_answer=predicted_answer,
                ground_truth_answer=ground_truth_answer,
                normalized_prediction=str(pred_num),
                normalized_ground_truth=str(gt_num),
                match_type=match_type,
                confidence=confidence,
            )

        # 2. Yes/no (before choice, so "Yes" is not read as letter "Y").
        if self.normalizer.is_yes_no_question(gt_norm):
            pred_yn = self.normalizer.normalize_yes_no(predicted_answer)
            gt_yn = self.normalizer.normalize_yes_no(ground_truth_answer)
            return AccuracyResult(
                is_correct=pred_yn == gt_yn,
                predicted_answer=predicted_answer,
                ground_truth_answer=ground_truth_answer,
                normalized_prediction=pred_yn,
                normalized_ground_truth=gt_yn,
                match_type="yes_no",
                confidence=1.0,
            )

        # 3. Multiple choice.
        pred_choice = self.normalizer.extract_choice(predicted_answer, question)
        gt_choice = self.normalizer.extract_choice(ground_truth_answer, question)
        if pred_choice is not None or gt_choice is not None:
            if pred_choice is None and gt_choice is not None:
                pred_choice = self.normalizer.match_answer_to_option(predicted_answer, question)
            if gt_choice is None and pred_choice is not None:
                gt_choice = self.normalizer.match_answer_to_option(ground_truth_answer, question)
            if pred_choice is not None and gt_choice is not None:
                return AccuracyResult(
                    is_correct=pred_choice == gt_choice,
                    predicted_answer=predicted_answer,
                    ground_truth_answer=ground_truth_answer,
                    normalized_prediction=pred_choice,
                    normalized_ground_truth=gt_choice,
                    match_type="choice",
                    confidence=1.0,
                )

        # 4. Exact text match (after normalization).
        if pred_norm == gt_norm:
            return AccuracyResult(
                is_correct=True,
                predicted_answer=predicted_answer,
                ground_truth_answer=ground_truth_answer,
                normalized_prediction=pred_norm,
                normalized_ground_truth=gt_norm,
                match_type="exact",
                confidence=1.0,
            )

        # 5. Single-word mismatch -> incorrect, no LLM needed.
        gt_words = gt_norm.split()
        pred_words = pred_norm.split()
        if len(gt_words) == 1 and len(pred_words) == 1:
            return AccuracyResult(
                is_correct=False,
                predicted_answer=predicted_answer,
                ground_truth_answer=ground_truth_answer,
                normalized_prediction=pred_norm,
                normalized_ground_truth=gt_norm,
                match_type="exact",
                confidence=1.0,
            )

        # 6. Longer text -> LLM judge if enabled.
        if self.use_llm_grader and self.llm_grader:
            is_correct, confidence = self.llm_grader.verify_answer(
                question, predicted_answer, ground_truth_answer
            )
            return AccuracyResult(
                is_correct=is_correct,
                predicted_answer=predicted_answer,
                ground_truth_answer=ground_truth_answer,
                normalized_prediction=pred_norm,
                normalized_ground_truth=gt_norm,
                match_type="llm_verified",
                confidence=confidence,
            )

        # 7. Substring fallback.
        is_correct = pred_norm in gt_norm or gt_norm in pred_norm
        return AccuracyResult(
            is_correct=is_correct,
            predicted_answer=predicted_answer,
            ground_truth_answer=ground_truth_answer,
            normalized_prediction=pred_norm,
            normalized_ground_truth=gt_norm,
            match_type="exact",
            confidence=0.7 if is_correct else 0.3,
        )

    def evaluate_dataset(
        self, predictions: Dict[int, Dict], ground_truth: Dict[int, Dict]
    ) -> Dict:
        """
        Evaluate a full dataset. Accuracy = (1/N) * sum(C(pred_i, gt_i)).

        Args:
            predictions: {id: {"answer": "...", "question": "..."}}
            ground_truth: {id: {"answer": "..."}}

        Returns:
            Dict with overall/weighted accuracy, per-type stats, and detailed rows.
        """
        results = []
        for idx in tqdm(predictions.keys(), desc="Evaluating accuracy", unit="sample"):
            if idx not in ground_truth:
                continue
            question = predictions[idx].get("question", "")
            pred_answer = predictions[idx].get("answer", "")
            gt_answer = ground_truth[idx].get("answer", "")

            result = self.evaluate_single(question, pred_answer, gt_answer)
            results.append({
                "sample_idx": idx,
                "is_correct": result.is_correct,
                "predicted": result.predicted_answer,
                "ground_truth": result.ground_truth_answer,
                "normalized_pred": result.normalized_prediction,
                "normalized_gt": result.normalized_ground_truth,
                "match_type": result.match_type,
                "confidence": result.confidence,
            })

        total = len(results)
        correct = sum(1 for r in results if r["is_correct"])
        accuracy = correct / total if total > 0 else 0.0

        weighted_correct = sum(r["confidence"] for r in results if r["is_correct"])
        weighted_accuracy = weighted_correct / total if total > 0 else 0.0

        type_stats = {}
        for match_type in set(r["match_type"] for r in results):
            type_results = [r for r in results if r["match_type"] == match_type]
            if type_results:
                type_correct = sum(1 for r in type_results if r["is_correct"])
                avg_confidence = sum(r["confidence"] for r in type_results) / len(type_results)
                type_stats[match_type] = {
                    "count": len(type_results),
                    "correct": type_correct,
                    "accuracy": type_correct / len(type_results),
                    "avg_confidence": avg_confidence,
                }

        return {
            "overall_accuracy": accuracy,
            "weighted_accuracy": weighted_accuracy,
            "total_samples": total,
            "correct_samples": correct,
            "type_statistics": type_stats,
            "detailed_results": results,
        }
