#!/usr/bin/env python3
"""
Reward functions for RL training (GRPO) on CRYSTAL.

These are the rewards used to train models with the Causal Process Reward (CPR)
and Semantic Process Reward (SPR) described in the paper. They are **pure Python**
(regex + the lightweight matching in ``crystal_metrics.similarity`` /
``crystal_metrics.accuracy``) — no torch, no trainer dependency — so the heavy
GRPO training stack simply imports them.

Every reward takes a list of ``completions`` in the trainer's format
``[[{"content": "<model output>"}], ...]`` and returns a list of floats, one per
completion. They are model-agnostic: they only look at the generated text.

Reward registry (``select_reward_func``):
    "format"            -> format_reward            (valid JSON schema)
    "accuracy"          -> accuracy_reward          (final-answer correctness)
    "reasoning"         -> word_overlap_reasoning_reward
    "reasoning_semantic"-> semantic_reasoning_reward (SPR)
    "reasoning_causal"  -> causal_process_reward     (CPR)
"""

import json
import re
from typing import Callable, Dict, List, Optional, Tuple

from .similarity import best_match_f1

__all__ = [
    "parse_reasoning_steps",
    "check_answer_match",
    "format_reward",
    "accuracy_reward",
    "word_overlap_reasoning_reward",
    "semantic_reasoning_reward",
    "causal_process_reward",
    "select_reward_func",
]


# ---------------------------------------------------------------------------
# Completion / JSON helpers
# ---------------------------------------------------------------------------
def _content_of(completion) -> str:
    """Extract the text of a single completion ([{"content": ...}] or str)."""
    try:
        if isinstance(completion, str):
            return completion
        return completion[0]["content"]
    except (KeyError, IndexError, TypeError):
        return str(completion)


def _extract_json_object(content: str) -> Optional[dict]:
    """
    Parse a JSON object from model output, mirroring the training-time parser:
    strip code fences, try whole-string parse, else extract the outermost
    balanced ``{...}`` and repair smart quotes / trailing commas.
    """
    content_cleaned = re.sub(r"```json\s*|\s*```", "", content).strip()

    # Method 1: parse the whole thing.
    try:
        return json.loads(content_cleaned)
    except (json.JSONDecodeError, Exception):
        pass

    # Method 2: outermost balanced brace span.
    first_brace = content_cleaned.find("{")
    if first_brace == -1:
        return None
    brace_count = 0
    json_str = None
    for idx in range(first_brace, len(content_cleaned)):
        if content_cleaned[idx] == "{":
            brace_count += 1
        elif content_cleaned[idx] == "}":
            brace_count -= 1
            if brace_count == 0:
                json_str = content_cleaned[first_brace:idx + 1]
                break
    if json_str is None:
        return None

    # Repairs: smart quotes, single->double quotes, missing commas, trailing commas.
    json_str = json_str.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    json_str = re.sub(r"""(?<=[:,\[])\s*'([^']*)'(?=\s*[,\]\}])""", r' "\1"', json_str)
    json_str = re.sub(r'"\s*\n\s*"', '",\n    "', json_str)
    json_str = re.sub(r'"\s+(?=")', '", ', json_str)
    json_str = re.sub(r",(\s*[\]}])", r"\1", json_str)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


def _validate_schema(parsed: Optional[dict]) -> Tuple[Optional[dict], bool]:
    """Return (cleaned, is_valid) for the {reasoning_steps: [str], answer: str} schema."""
    if not isinstance(parsed, dict):
        return None, False
    if "reasoning_steps" not in parsed or "answer" not in parsed:
        return None, False
    if not isinstance(parsed["reasoning_steps"], list) or not isinstance(parsed["answer"], str):
        return None, False
    if not all(isinstance(s, str) for s in parsed["reasoning_steps"]):
        return None, False
    steps = [s.strip() for s in parsed["reasoning_steps"] if s and s.strip()]
    if not steps:
        return None, False
    return {"reasoning_steps": steps, "answer": parsed["answer"].strip()}, True


def parse_reasoning_steps(response: str) -> Tuple[List[str], str]:
    """
    Parse a completion into (reasoning_steps, answer).

    Supports the CRYSTAL JSON schema and an XML ``<think>...</think><answer>...</answer>``
    fallback with several step-delimiter heuristics. Verbatim port of the
    training-time parser so CPR rewards match exactly.
    """
    # JSON format first.
    try:
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'\{[^{}]*"reasoning_steps"[^{}]*\}', response, re.DOTALL)
            json_str = json_match.group(0) if json_match else None
        if json_str:
            data = json.loads(json_str)
            steps = data.get("reasoning_steps", [])
            if isinstance(steps, str):
                steps = [steps] if steps else []
            answer = data.get("answer", "")
            if steps or answer:
                return steps, str(answer)
    except (json.JSONDecodeError, TypeError):
        pass

    # XML / tag format.
    think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    if not think_match:
        think_match = re.search(r"<reasoning>(.*?)</reasoning>", response, re.DOTALL)
    if not think_match:
        return [], response.strip()

    think_content = think_match.group(1).strip()
    steps: List[str] = []

    matches = re.findall(r"Step\s*(\d+)[:.]\s*(.*?)(?=Step\s*\d+[:.:]|\Z)",
                         think_content, re.DOTALL | re.IGNORECASE)
    if matches:
        steps = [m[1].strip() for m in matches if m[1].strip()]
    if not steps:
        matches = re.findall(r"^\s*(\d+)[.)]\s*(.*?)(?=^\s*\d+[.)]|\Z)",
                             think_content, re.DOTALL | re.MULTILINE)
        if matches:
            steps = [m[1].strip() for m in matches if m[1].strip()]
    if not steps:
        matches = re.findall(r"[-•*]\s*(.*?)(?=[-•*]|\Z)", think_content, re.DOTALL)
        if matches:
            steps = [m.strip() for m in matches if m.strip()]
    if not steps:
        lines = [s.strip() for s in think_content.split("\n") if s.strip()]
        steps = [l for l in lines if len(l) > 10]

    answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if not answer_match:
        answer_match = re.search(r"<final_answer>(.*?)</final_answer>", response, re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else ""
    return steps, answer


def check_answer_match(predicted: str, ground_truth: str) -> bool:
    """Multi-format answer match: exact, MCQ letter, yes/no, numeric tolerance, substring."""
    if not predicted or not ground_truth:
        return False
    pred_clean = predicted.strip().lower()
    gt_clean = ground_truth.strip().lower()
    if pred_clean == gt_clean:
        return True

    pred_option = re.search(r"^([a-d])\b", pred_clean)
    gt_option = re.search(r"^([a-d])\b", gt_clean)
    if pred_option and gt_option:
        return pred_option.group(1) == gt_option.group(1)

    yes_variants = {"yes", "true", "correct", "1"}
    no_variants = {"no", "false", "incorrect", "0"}
    if (pred_clean in yes_variants or pred_clean in no_variants) and \
       (gt_clean in yes_variants or gt_clean in no_variants):
        return (pred_clean in yes_variants) == (gt_clean in yes_variants)

    pred_nums = re.findall(r"[-+]?\d*\.?\d+", pred_clean)
    gt_nums = re.findall(r"[-+]?\d*\.?\d+", gt_clean)
    if pred_nums and gt_nums:
        try:
            pred_val = float(pred_nums[0])
            gt_val = float(gt_nums[0])
            if abs(gt_val) > 1:
                return abs(pred_val - gt_val) / abs(gt_val) < 0.01
            return abs(pred_val - gt_val) < 0.01
        except ValueError:
            pass

    if len(gt_clean) > 10:
        return gt_clean in pred_clean or pred_clean in gt_clean
    return False


# ---------------------------------------------------------------------------
# Reward functions (each: completions -> list[float])
# ---------------------------------------------------------------------------
def format_reward(completions, **kwargs) -> List[float]:
    """1.0 if the completion is valid JSON with a non-empty string reasoning_steps + answer."""
    rewards = []
    for completion in completions:
        parsed = _extract_json_object(_content_of(completion))
        _, is_valid = _validate_schema(parsed)
        rewards.append(1.0 if is_valid else 0.0)
    return rewards


def accuracy_reward(completions, solution, use_llm_grader: bool = False,
                    llm_model: str = "llama3.2",
                    base_url: str = "http://localhost:11434/v1", **kwargs) -> List[float]:
    """
    Final-answer correctness in [0, 1] (the AccuracyCalculator confidence when correct,
    else 0). ``solution`` is the list of ground-truth answers, one per completion.
    """
    from .accuracy import AccuracyCalculator

    calc = AccuracyCalculator(use_llm_grader=use_llm_grader, llm_model=llm_model, base_url=base_url)
    problems = kwargs.get("problem", [""] * len(completions))
    rewards = []
    for i, completion in enumerate(completions):
        reward = 0.0
        parsed = _extract_json_object(_content_of(completion))
        predicted_answer = parsed.get("answer", "") if isinstance(parsed, dict) else ""
        if predicted_answer:
            sol = solution[i] if i < len(solution) else ""
            problem = problems[i] if i < len(problems) else ""
            result = calc.evaluate_single(problem, predicted_answer, sol)
            if result.is_correct:
                reward = result.confidence
        rewards.append(reward)
    return rewards


def word_overlap_reasoning_reward(completions, reference_steps=None,
                                  threshold: float = 0.45, **kwargs) -> List[float]:
    """Reasoning F1 via word-overlap matching (the default training reasoning reward)."""
    reference_steps = reference_steps if reference_steps is not None else kwargs.get("reference_steps", [])
    rewards = []
    for i, completion in enumerate(completions):
        reward = 0.0
        parsed = _extract_json_object(_content_of(completion))
        cleaned, is_valid = _validate_schema(parsed)
        if is_valid:
            ref = reference_steps[i] if i < len(reference_steps) else []
            ref_cleaned = [str(s).strip() for s in ref if s and str(s).strip()]
            if ref_cleaned:
                f1, _, _ = best_match_f1(cleaned["reasoning_steps"], ref_cleaned, threshold=threshold)
                reward = f1
        rewards.append(reward)
    return rewards


def semantic_reasoning_reward(completions, reference_steps=None,
                              threshold: float = 0.70, model=None, **kwargs) -> List[float]:
    """
    Semantic Process Reward (SPR): reasoning F1 via SentenceTransformer cosine
    similarity. Falls back to word overlap if the embedding model is unavailable.
    """
    from .similarity import semantic_match_f1

    reference_steps = reference_steps if reference_steps is not None else kwargs.get("reference_steps", [])
    rewards = []
    for i, completion in enumerate(completions):
        reward = 0.0
        parsed = _extract_json_object(_content_of(completion))
        cleaned, is_valid = _validate_schema(parsed)
        if is_valid:
            ref = reference_steps[i] if i < len(reference_steps) else []
            ref_cleaned = [str(s).strip() for s in ref if s and str(s).strip()]
            if ref_cleaned:
                try:
                    f1, _, _ = semantic_match_f1(cleaned["reasoning_steps"], ref_cleaned,
                                                 model=model, threshold=threshold)
                except Exception:
                    f1, _, _ = best_match_f1(cleaned["reasoning_steps"], ref_cleaned, threshold=0.45)
                reward = f1
        rewards.append(reward)
    return rewards


def _step_alignment_f1(predicted_steps: List[str], reference_steps: List[str],
                       threshold: float = 0.3) -> float:
    if not predicted_steps or not reference_steps:
        return 0.0
    f1, _, _ = best_match_f1(predicted_steps, reference_steps, threshold=threshold)
    return f1


def _lightweight_causal_reward(predicted_steps, reference_steps, predicted_answer,
                               ground_truth, answer_weight=0.6, step_weight=0.4) -> float:
    """
    CPR proxy: multiplicative interaction of answer correctness and step alignment.

        correct   -> answer_weight * 1 + step_weight * F1_step
        incorrect -> step_weight * F1_step * 0.3
    """
    answer_correct = check_answer_match(predicted_answer, ground_truth)
    step_score = _step_alignment_f1(predicted_steps, reference_steps, threshold=0.3) \
        if (predicted_steps and reference_steps) else 0.0
    if answer_correct:
        return answer_weight * 1.0 + step_weight * step_score
    return step_weight * step_score * 0.3


def causal_process_reward(completions, ground_truths=None, reference_steps=None,
                          answer_weight: float = 0.6, step_weight: float = 0.4,
                          **kwargs) -> List[float]:
    """
    Causal Process Reward (CPR). Rewards reasoning steps by their causal necessity
    for the correct answer via a multiplicative answer x step-alignment interaction.

    Args:
        completions: trainer-format completions.
        ground_truths: list of correct answers (also read from kwargs
            ground_truth/answer/solution for trainer compatibility).
        reference_steps: list of reference step lists.
        answer_weight, step_weight: CPR weights (paper defaults 0.6 / 0.4).
    """
    if ground_truths is None:
        ground_truths = kwargs.get("ground_truth", kwargs.get("answer", kwargs.get("solution", None)))
    if ground_truths is None:
        ground_truths = [""] * len(completions)
    if not isinstance(ground_truths, list):
        ground_truths = [ground_truths] * len(completions)
    ground_truths = [str(gt) if gt is not None else "" for gt in ground_truths]

    if reference_steps is None:
        reference_steps = kwargs.get("reference_steps", None)
    if reference_steps is None or not isinstance(reference_steps, list):
        reference_steps = [[]] * len(completions)

    rewards = []
    for i, completion in enumerate(completions):
        steps, answer = parse_reasoning_steps(_content_of(completion))
        gt = ground_truths[i] if i < len(ground_truths) else ""
        ref = reference_steps[i] if i < len(reference_steps) else []
        if ref and not isinstance(ref[0], str):
            ref = [str(s) for s in ref]
        ref = [s.strip() for s in ref if s and str(s).strip()]
        rewards.append(_lightweight_causal_reward(
            predicted_steps=steps, reference_steps=ref,
            predicted_answer=answer, ground_truth=gt,
            answer_weight=answer_weight, step_weight=step_weight,
        ))
    return rewards


_REWARD_REGISTRY: Dict[str, Callable] = {
    "format": format_reward,
    "accuracy": accuracy_reward,
    "reasoning": word_overlap_reasoning_reward,
    "reasoning_semantic": semantic_reasoning_reward,
    "reasoning_causal": causal_process_reward,
}


def select_reward_func(name: str) -> Callable:
    """Return the reward callable registered under ``name``."""
    if name not in _REWARD_REGISTRY:
        raise ValueError(
            f"Unknown reward function: {name!r}. "
            f"Available: {sorted(_REWARD_REGISTRY)}"
        )
    return _REWARD_REGISTRY[name]
