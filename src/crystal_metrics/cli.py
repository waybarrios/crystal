#!/usr/bin/env python3
"""
Command-line interface for crystal-metrics.

Usage:
    crystal-metrics evaluate predictions.json references.json [options]

Both files are JSON objects keyed by sample id:

    predictions.json
        {"0": {"question": "...", "reasoning_steps": ["..."], "answer": "..."}}
    references.json
        {"0": {"reference_steps": ["..."], "answer": "..."}}
"""

import argparse
import json
import sys
from typing import Dict, Tuple


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def align_predictions_to_references(
    predictions: Dict, references: Dict
) -> Tuple[Dict, Dict, int]:
    """
    Align predictions to references, deciding what to do with references that
    have no matching prediction.

    CRYSTAL paper protocol: an unanswered reference is NOT skipped — it is scored
    as a complete miss (empty prediction => Match F1 / accuracy = 0). Skipping it
    instead would inflate scores by hiding the cases a model failed to answer.

    Returns:
        (aligned_predictions, aligned_references, num_missing) over the union of
        reference ids, where every reference id is present in aligned_predictions
        (missing ones filled with empty placeholders).
    """
    aligned_pred = {}
    aligned_ref = {}
    num_missing = 0
    for idx, ref in references.items():
        aligned_ref[idx] = ref
        if idx in predictions:
            aligned_pred[idx] = predictions[idx]
        else:
            num_missing += 1
            aligned_pred[idx] = {"question": "", "reasoning_steps": [], "answer": ""}
    return aligned_pred, aligned_ref, num_missing


def _cmd_evaluate(args: argparse.Namespace) -> int:
    predictions = _load_json(args.predictions)
    references = _load_json(args.references)

    aligned_pred, aligned_ref, num_missing = align_predictions_to_references(
        predictions, references
    )
    if num_missing:
        print(f"Note: {num_missing} reference(s) had no prediction; scored as 0 (paper protocol).")

    # Reasoning metrics (Match F1 / Ordered Match F1 / Precision / Recall).
    from .reasoning import MLLMReasoningEvaluator

    evaluator = MLLMReasoningEvaluator(
        model_name=args.model, similarity_threshold=args.threshold
    )
    df = evaluator.evaluate_dataset(
        aligned_pred, aligned_ref, verbose=True,
        alpha=args.alpha, order_metric=args.order_metric,
    )

    summary = {
        "samples": int(len(df)),
        "match_f1": float(df["match_f1"].mean()) if len(df) else 0.0,
        "precision": float(df["precision"].mean()) if len(df) else 0.0,
        "recall": float(df["recall"].mean()) if len(df) else 0.0,
    }
    if args.alpha > 0 and len(df):
        summary["ordered_match_f1"] = float(df["ordered_match_f1"].mean())

    # Accuracy (multi-format). LLM judge only if requested.
    # Score only samples that actually have a prediction; missing ones count as
    # incorrect over the full reference total (paper protocol). We deliberately do
    # NOT feed empty placeholder answers through the calculator, because the
    # original substring fallback treats an empty string as a match ("" in gt).
    from .accuracy import AccuracyCalculator

    answered = {i: predictions[i] for i in references if i in predictions}
    calc = AccuracyCalculator(use_llm_grader=args.use_judge, llm_model=args.judge_model)
    acc = calc.evaluate_dataset(answered, references)
    n_total = len(references)
    summary["accuracy"] = float(acc["correct_samples"] / n_total) if n_total else 0.0

    print("\n=== CRYSTAL metrics ===")
    for k, v in summary.items():
        print(f"  {k:18s}: {v:.4f}" if isinstance(v, float) else f"  {k:18s}: {v}")

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nPer-sample results saved to: {args.output}")
        with open(args.output.replace(".csv", "_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crystal-metrics",
        description="Transparent multimodal reasoning metrics from the CRYSTAL benchmark.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ev = sub.add_parser("evaluate", help="Evaluate predictions against references.")
    ev.add_argument("predictions", help="Path to predictions JSON.")
    ev.add_argument("references", help="Path to references JSON.")
    ev.add_argument("--model", default="all-distilroberta-v1",
                    help="Sentence-transformer model (default: ablation-validated).")
    ev.add_argument("--threshold", type=float, default=None,
                    help="Cosine match threshold (default: model-specific, 0.35 for distilroberta).")
    ev.add_argument("--alpha", type=float, default=0.0,
                    help="Order sensitivity for Ordered Match F1 in [0,1] (0 = off, 0.3 recommended).")
    ev.add_argument("--order-metric", default="kendall_tau", choices=["kendall_tau", "lis"],
                    help="Order metric for Ordered Match F1.")
    ev.add_argument("--use-judge", action="store_true",
                    help="Use the LLM judge for free-form answers (needs [judge] extra).")
    ev.add_argument("--judge-model", default="llama3.2", help="LLM judge model name.")
    ev.add_argument("--output", default=None, help="Optional CSV path for per-sample results.")
    ev.set_defaults(func=_cmd_evaluate)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
