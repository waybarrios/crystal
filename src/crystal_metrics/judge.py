#!/usr/bin/env python3
"""
Optional LLM judge for semantic answer verification.

This module requires the ``[judge]`` extra::

    pip install crystal-metrics[judge]

It talks to any OpenAI-compatible endpoint (e.g. a local Ollama server) to decide
whether a free-form predicted answer is semantically equivalent to the reference.
``openai`` is imported lazily inside ``LLMGrader.__init__`` so that importing the
core package never requires it.
"""

import json
import re
from typing import Tuple


class LLMGrader:
    """Verify free-form text answers via an OpenAI-compatible LLM endpoint."""

    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434/v1"):
        """
        Args:
            model: Model name served by the endpoint (e.g. "llama3.2", "gpt-oss:120b").
            base_url: OpenAI-compatible base URL (Ollama default shown).
        """
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "The LLM judge requires the 'openai' package. "
                "Install it with: pip install crystal-metrics[judge]"
            ) from e

        self.model = model
        self.client = OpenAI(
            base_url=base_url,
            api_key="ollama",  # required by the client, but unused by Ollama
        )

    def verify_answer(self, question: str, predicted: str, ground_truth: str) -> Tuple[bool, float]:
        """
        Ask the LLM whether predicted and ground_truth are semantically equivalent.

        Returns:
            (is_correct, confidence). Falls back to string matching on any error.
        """
        # Local import keeps the symbol available to the fallback paths.
        from .accuracy import AnswerNormalizer

        system_prompt = """You are an expert evaluator for question answering systems.
Your task is to determine if two answers are semantically equivalent.
Consider synonyms, paraphrases, and different phrasings of the same content.
Respond ONLY with a JSON object containing two fields:
- "correct": boolean (true if answers are equivalent, false otherwise)
- "confidence": float between 0.0 and 1.0 (your confidence in the judgment)

Example responses:
{"correct": true, "confidence": 0.95}
{"correct": false, "confidence": 0.85}"""

        user_prompt = f"""Question: {question}

Ground Truth Answer: {ground_truth}

Predicted Answer: {predicted}

Are these answers semantically equivalent?"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=100,
            )

            response_text = response.choices[0].message.content.strip()

            try:
                response_text = re.sub(r"```json\s*|\s*```", "", response_text)
                parsed = json.loads(response_text)
                is_correct = parsed.get("correct", False)
                confidence = float(parsed.get("confidence", 0.5))
                confidence = max(0.0, min(1.0, confidence))
                return is_correct, confidence
            except json.JSONDecodeError:
                response_lower = response_text.lower()
                if 'true' in response_lower or '"correct": true' in response_lower:
                    return True, 0.7
                elif 'false' in response_lower or '"correct": false' in response_lower:
                    return False, 0.7

                pred_norm = AnswerNormalizer.normalize_text(predicted)
                gt_norm = AnswerNormalizer.normalize_text(ground_truth)
                if pred_norm == gt_norm:
                    return True, 1.0
                elif pred_norm in gt_norm or gt_norm in pred_norm:
                    return True, 0.8
                return False, 0.3

        except Exception as e:
            print(f"LLM grading error: {e}")
            pred_norm = AnswerNormalizer.normalize_text(predicted)
            gt_norm = AnswerNormalizer.normalize_text(ground_truth)
            if pred_norm == gt_norm:
                return True, 1.0
            elif pred_norm in gt_norm or gt_norm in pred_norm:
                return True, 0.8
            return False, 0.3
