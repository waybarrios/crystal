# Metric definitions

CRYSTAL scores *how a model reasons*, not just whether the final answer is right.
All reasoning metrics are built on a semantic matching between predicted and
reference reasoning steps.

## Step matching

1. Embed every predicted step and every reference step with a sentence
   transformer (default `all-distilroberta-v1`).
2. Build the cosine-similarity matrix between the two sets.
3. Keep pairs with similarity **> threshold** (default τ = 0.35, ablation-validated).
4. Assign matches **greedily, 1:1** — highest similarity first, no step matched twice.

The match set drives every metric below.

## Precision

Fraction of *predicted* steps that matched a reference step.

```
Precision = |matched_predictions| / |total_predicted_steps|
```

High precision = the model says few wrong things.

## Recall

Fraction of *reference* steps that were covered by a prediction.

```
Recall = |matched_references| / |total_reference_steps|
```

High recall = the model covers most of the required reasoning. In CRYSTAL,
**recall is the hard part** — most models are high-precision, low-recall.

## Match F1

Harmonic mean of precision and recall — the headline reasoning-quality metric.

```
Match F1 = 2 · (Precision · Recall) / (Precision + Recall)
```

A lucky-guess answer with incoherent reasoning gets a low Match F1 even when the
final answer is correct.

## Ordered Match F1

Match F1 penalized when matched steps appear **out of order**. Controlled by
`alpha ∈ [0, 1]` (0 = ignore order; 0.3 recommended):

```
Ordered Match F1 = Match F1 · ((1 − alpha) + alpha · order_score)
```

`order_score ∈ [0, 1]` comes from one of two order metrics:

- **Kendall's τ** (normalized to [0, 1]) — concordant minus discordant matched pairs.
  +1 perfect order, 0 random, −1 reversed.
- **LIS ratio** — length of the Longest Increasing Subsequence of matched
  prediction indices divided by the number of matches. Fraction of matched steps
  in correct relative order.

Choose with `order_metric="kendall_tau"` (default) or `order_metric="lis"`.
With fewer than two matches, order is undefined and the order score is 1.0.

## Accuracy

Final-answer correctness, resolved in this order:

1. **Numeric** — extract numbers; correct within tolerance
   (`|p−g| ≤ ε_abs` or `|p−g| / max(|g|, δ) ≤ ε_rel`; defaults ε_abs=0.05, ε_rel=0.10).
2. **Yes/No** — normalized boolean match.
3. **Multiple choice** — extract the letter (A/B/C/…), or match the answer text to
   an option in the question.
4. **Exact text** — after normalization (lowercasing, punctuation/quote cleanup).
5. **Free-form text** — optional LLM judge for semantic equivalence (`[judge]` extra).
6. **Substring fallback** — when the judge is disabled.

`evaluate_dataset` reports overall accuracy, confidence-weighted accuracy, and a
per-`match_type` breakdown.

## Defaults (from the paper)

| Setting | Value | Source |
|---------|-------|--------|
| Embedding model | `all-distilroberta-v1` | Paper §4.3 |
| Similarity threshold τ | 0.35 | Ablation-validated |
| Recommended `alpha` | 0.3 | Paper |
| Numeric tolerance | ε_abs=0.05, ε_rel=0.10 | Paper Eq. (2) |
