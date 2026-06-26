# crystal-metrics

[![PyPI](https://img.shields.io/pypi/v/crystal-metrics.svg)](https://pypi.org/project/crystal-metrics/)
[![Python](https://img.shields.io/pypi/pyversions/crystal-metrics.svg)](https://pypi.org/project/crystal-metrics/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/waybarrios/crystal/blob/main/README.md#license)
[![arXiv](https://img.shields.io/badge/arXiv-2603.13099-b31b1b.svg)](https://arxiv.org/abs/2603.13099)

Transparent multimodal reasoning metrics from the **CRYSTAL** benchmark.

*Your model gets the right answer. But does it actually **reason**?* Standard
benchmarks only check the final answer, so a lucky guess scores the same as
sound reasoning. `crystal-metrics` scores the **reasoning chain itself** —
step-level precision/recall, ordering, and answer accuracy.

## What it measures

| Metric | Measures |
|--------|----------|
| **Match F1** | Step-level F1 of predicted vs. reference reasoning steps via semantic-similarity matching |
| **Precision** | Fraction of predicted steps that match a reference step (few wrong things) |
| **Recall** | Fraction of reference steps that were covered (completeness — the hard part) |
| **Ordered Match F1** | Match F1 penalized for out-of-order reasoning (Kendall's τ or LIS ratio) |
| **Accuracy** | Multi-format final-answer correctness (yes/no, numeric, multiple choice, free text) |

## Install

```bash
pip install crystal-metrics          # core metrics (no LLM required)
pip install crystal-metrics[judge]   # + optional LLM judge for free-form answers
```

Requires Python 3.8+. The default embedding model `all-distilroberta-v1` is
downloaded and cached on first use.

## Quickstart

```python
from crystal_metrics import MLLMReasoningEvaluator

evaluator = MLLMReasoningEvaluator()  # all-distilroberta-v1, threshold τ=0.35 (paper defaults)

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
    alpha=0.3,  # enable Ordered Match F1 (0 = order-insensitive)
)

print(f"Match F1:         {m.match_f1:.3f}")
print(f"Precision:        {m.precision:.3f}")
print(f"Recall:           {m.recall:.3f}")
print(f"Ordered Match F1: {m.ordered_match_f1:.3f}")
```

### Answer accuracy

```python
from crystal_metrics import AccuracyCalculator

calc = AccuracyCalculator(use_llm_grader=False)          # rule-based, no LLM
acc = calc.evaluate_dataset(predictions, references)
print(acc["overall_accuracy"], acc["type_statistics"])
```

The optional LLM judge (for free-form text) needs the `[judge]` extra and any
OpenAI-compatible endpoint (e.g. a local Ollama server):

```python
calc = AccuracyCalculator(use_llm_grader=True, llm_model="gpt-oss:120b",
                          base_url="http://localhost:11434/v1")
```

## Command line

```bash
crystal-metrics evaluate predictions.json references.json --alpha 0.3
```

```
=== CRYSTAL metrics ===
  samples           : 3
  match_f1          : 0.5524
  precision         : 0.6667
  recall            : 0.4722
  ordered_match_f1  : 0.4952
  accuracy          : 0.6667
```

### Data format

```jsonc
// predictions
{"<id>": {"question": "...", "reasoning_steps": ["..."], "answer": "..."}}
// references
{"<id>": {"reference_steps": ["..."], "answer": "..."}}
```

## Paper defaults

| Setting | Value | Source |
|---------|-------|--------|
| Embedding model | `all-distilroberta-v1` | Paper §4.3 |
| Similarity threshold τ | 0.35 | Ablation-validated |
| Recommended `alpha` | 0.3 | Paper |
| Numeric tolerance | ε_abs = 0.05, ε_rel = 0.10 | Paper Eq. (2) |

## Documentation

- [Installation](https://github.com/waybarrios/crystal/blob/main/docs/installation.md)
- [Quickstart](https://github.com/waybarrios/crystal/blob/main/docs/quickstart.md)
- [Metric definitions](https://github.com/waybarrios/crystal/blob/main/docs/metrics.md)
- [CLI reference](https://github.com/waybarrios/crystal/blob/main/docs/cli.md)

Benchmark: 🤗 [waybarrios/CRYSTAL](https://huggingface.co/datasets/waybarrios/CRYSTAL) · Project: [github.com/waybarrios/crystal](https://github.com/waybarrios/crystal)

## Citation

```bibtex
@misc{barrios2026crystal,
  title   = {Beyond Final Answers: CRYSTAL Benchmark for Transparent
             Multimodal Reasoning Evaluation},
  author  = {Wayner Barrios and SouYoung Jin},
  year    = {2026},
  eprint  = {2603.13099},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url     = {https://arxiv.org/abs/2603.13099}
}
```

## License

MIT — see the [project repository](https://github.com/waybarrios/crystal).
