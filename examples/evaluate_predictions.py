#!/usr/bin/env python3
"""
Minimal end-to-end example for crystal-metrics.

Run:
    python examples/evaluate_predictions.py
"""

from crystal_metrics import AccuracyCalculator, MLLMReasoningEvaluator

predictions = {
    0: {
        "question": "Which of the 3 objects is the smallest?",
        "reasoning_steps": [
            "Three objects sit on the table",
            "The middle console looks compact",
            "Therefore the answer is C",
        ],
        "answer": "C",
    }
}

references = {
    0: {
        "reference_steps": [
            "There are three objects in the image",
            "Compare the sizes of the three objects",
            "The middle console is the smallest",
            "Select option C",
        ],
        "answer": "C",
    }
}

# Reasoning quality: Match F1 + Ordered Match F1 (alpha enables ordering penalty).
evaluator = MLLMReasoningEvaluator()  # all-distilroberta-v1, tau=0.35
df = evaluator.evaluate_dataset(predictions, references, verbose=True, alpha=0.3)
print(df[["match_f1", "precision", "recall", "ordered_match_f1"]].round(3))

# Final-answer accuracy (rule-based; no LLM needed for a multiple-choice answer).
calc = AccuracyCalculator(use_llm_grader=False)
acc = calc.evaluate_dataset(predictions, references)
print(f"\nAccuracy: {acc['overall_accuracy']:.3f}")
