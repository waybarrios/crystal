# Quickstart

## Evaluate a single sample

```python
from crystal_metrics import MLLMReasoningEvaluator

evaluator = MLLMReasoningEvaluator()  # all-distilroberta-v1, threshold τ=0.35

m = evaluator.evaluate_single(
    predicted_steps=[
        "Three objects sit on the table",
        "The middle console is the smallest",
        "Therefore the answer is C",
    ],
    reference_steps=[
        "There are three objects in the image",
        "Compare the sizes of the three objects",
        "The middle object is smallest",
        "Select option C",
    ],
    alpha=0.3,              # enable Ordered Match F1 (0 = order-insensitive)
    order_metric="kendall_tau",  # or "lis"
)

print(f"Match F1:        {m.match_f1:.3f}")
print(f"Precision:       {m.precision:.3f}")
print(f"Recall:          {m.recall:.3f}")
print(f"Ordered Match F1:{m.ordered_match_f1:.3f}")
print(f"Kendall's τ:     {m.kendall_tau:.3f}   LIS ratio: {m.lis_ratio:.3f}")
```

## Evaluate a dataset

```python
predictions = {
    0: {"reasoning_steps": ["...", "..."], "answer": "C"},
    1: {"reasoning_steps": ["...", "..."], "answer": "2"},
}
references = {
    0: {"reference_steps": ["...", "...", "..."], "answer": "C"},
    1: {"reference_steps": ["...", "..."], "answer": "2"},
}

df = evaluator.evaluate_dataset(predictions, references, verbose=True, alpha=0.3)
print(df[["match_f1", "precision", "recall", "ordered_match_f1"]].mean())
```

`evaluate_dataset` returns a `pandas.DataFrame` with one row per sample.

## Answer accuracy

```python
from crystal_metrics import AccuracyCalculator

calc = AccuracyCalculator(use_llm_grader=False)   # rule-based, no LLM
acc = calc.evaluate_dataset(predictions, references)
print(f"Accuracy: {acc['overall_accuracy']:.3f}")
print(acc["type_statistics"])    # per-format breakdown
```

To grade free-form text answers semantically, install the `[judge]` extra and
point at any OpenAI-compatible endpoint (e.g. a local Ollama server):

```python
calc = AccuracyCalculator(
    use_llm_grader=True,
    llm_model="gpt-oss:120b",
    base_url="http://localhost:11434/v1",
)
```

## Data format

```jsonc
// predictions
{"<id>": {"question": "...", "reasoning_steps": ["..."], "answer": "..."}}
// references
{"<id>": {"reference_steps": ["..."], "answer": "..."}}
```

See [metrics.md](metrics.md) for what each metric means and [cli.md](cli.md) for
the command-line interface.
