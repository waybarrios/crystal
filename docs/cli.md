# Command-line interface

Installing the package adds a `crystal-metrics` command.

## evaluate

```bash
crystal-metrics evaluate PREDICTIONS.json REFERENCES.json [options]
```

Computes Match F1, Ordered Match F1, Precision, Recall, and Accuracy and prints a
summary.

### Input files

Both are JSON objects keyed by sample id:

```jsonc
// PREDICTIONS.json
{"0": {"question": "...", "reasoning_steps": ["..."], "answer": "..."}}
// REFERENCES.json
{"0": {"reference_steps": ["..."], "answer": "..."}}
```

### Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--model` | `all-distilroberta-v1` | Sentence-transformer embedding model |
| `--threshold` | model default (0.35) | Cosine match threshold τ |
| `--alpha` | `0.0` | Order sensitivity for Ordered Match F1 (0.3 recommended) |
| `--order-metric` | `kendall_tau` | `kendall_tau` or `lis` |
| `--use-judge` | off | Use the LLM judge for free-form answers (needs `[judge]`) |
| `--judge-model` | `llama3.2` | LLM judge model name |
| `--output` | none | Write per-sample results to a CSV (plus `_summary.json`) |

### Missing predictions

A reference id with no matching prediction is **not skipped**: its reasoning is
scored as a complete miss (Match F1 = 0) and its answer counts as incorrect, over
the full reference total. This matches the CRYSTAL paper protocol — skipping
unanswered samples would inflate scores. The number of missing predictions is
printed at the top of the run.

### Example

```bash
crystal-metrics evaluate preds.json refs.json --alpha 0.3 --output results.csv
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
