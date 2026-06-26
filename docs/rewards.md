# RL reward functions

`crystal_metrics.rewards` provides the reward functions used to train models with
GRPO on CRYSTAL — including the **Causal Process Reward (CPR)** and **Semantic
Process Reward (SPR)** from the paper. They are pure Python (no torch, no trainer
dependency) and **model-agnostic**: each takes the generated text and returns a
float per completion.

## Reward registry

| Name | Function | What it rewards |
|------|----------|-----------------|
| `format` | `format_reward` | 1.0 if output is valid JSON `{reasoning_steps:[str], answer:str}` |
| `accuracy` | `accuracy_reward` | Final-answer correctness in [0, 1] |
| `reasoning` | `word_overlap_reasoning_reward` | Step F1 via word overlap (τ=0.45) |
| `reasoning_semantic` | `semantic_reasoning_reward` | **SPR**: step F1 via embeddings (τ=0.70) |
| `reasoning_causal` | `causal_process_reward` | **CPR**: answer × step-alignment interaction |

```python
from crystal_metrics import select_reward_func
reward_fn = select_reward_func("reasoning_causal")
```

## Completion format

Every reward takes completions in the GRPO trainer format and returns one float
per completion:

```python
completions = [[{"content": '{"reasoning_steps": ["..."], "answer": "B"}'}], ...]
```

## Causal Process Reward (CPR)

```python
from crystal_metrics import causal_process_reward

rewards = causal_process_reward(
    completions,
    ground_truths=["B", "C", ...],          # correct answers
    reference_steps=[["step a", "step b"], ...],
    answer_weight=0.6,                       # paper default
    step_weight=0.4,                         # paper default
)
```

CPR formula (per completion):

```
correct answer   ->  answer_weight · 1 + step_weight · F1_step
incorrect answer ->  step_weight · F1_step · 0.3
```

So a correct answer with faithful reasoning scores highest; a correct answer with
unrelated reasoning is penalized (the "right answer, wrong reasons" case), and a
wrong answer gets at most partial credit for on-track steps.

## Semantic Process Reward (SPR) — experimental

> SPR is an **experimental** reasoning reward, not the method reported in the
> paper (the CRYSTAL results use CPR / CPR-Curriculum). It is provided for
> reference and ablation.

```python
from crystal_metrics import semantic_reasoning_reward

rewards = semantic_reasoning_reward(
    completions, reference_steps=[["..."], ...], threshold=0.70,
)
```

SPR matches steps by SentenceTransformer cosine similarity (captures
paraphrases), falling back to word overlap if the embedding model is
unavailable.

## Using rewards with a GRPO trainer

These functions plug directly into a TRL-style `GRPOTrainer`'s `reward_funcs`
list. The heavy training stack (sub-project `training/`) imports them from here,
so the reward logic lives in one place and is unit- and parity-tested
independently of the trainer.
