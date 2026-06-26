# Training with Causal Process Reward (CPR)

This directory holds the CRYSTAL **RL training** layer: the reward adapter, launch
scripts, DeepSpeed configs, and the Qwen2.5-VL / InternVL model modules used to
train models with GRPO on CRYSTAL.

The reward logic itself (CPR, SPR, format, accuracy) lives in the pip package
[`crystal_metrics.rewards`](../docs/rewards.md); [`rewards_grpo.py`](rewards_grpo.py)
is the **model-agnostic adapter** that plugs those rewards into the trainer.

> **Self-contained.** The GRPO trainer (`open_r1/`) is **bundled** here — vendored
> from [open-r1-multimodal](https://github.com/om-ai-lab/VLM-R1) (Apache-2.0, see
> `open_r1/LICENSE` and `open_r1/NOTICE`) with the CRYSTAL CPR/SPR modifications.
> Its reward logic is rewired to import from the `crystal-metrics` package, so
> there is a single source of truth for the reward functions. No external clone
> needed.

## Contents

```
training/
├── open_r1/               # BUNDLED GRPO trainer (grpo_rec.py, trainer/, vlm_modules/, monkey patches)
│   ├── vlm_modules/        #   Qwen2.5-VL, InternVL, GLM-4V model adapters (rewards -> crystal_metrics)
│   ├── trainer/            #   VLMGRPOTrainer + GRPOConfig
│   ├── utils/              #   PCGrad, callbacks, pycocotools, ...
│   └── LICENSE, NOTICE     #   Apache-2.0 + attribution
├── rewards_grpo.py        # model-agnostic reward adapter (for custom trainers)
├── requirements.txt       # pinned training stack (trl, deepspeed, vllm, ...)
├── configs/               # DeepSpeed ZeRO-3 configs (Qwen + InternVL)
├── scripts/               # launch scripts (CPR, curriculum, SPR, answer-only)
└── docs/                  # VQA training guide, multi-model setup, GRPO tips
```

## Setup

```bash
# Dedicated environment with the pinned training stack (includes crystal-metrics).
python -m venv .venv-train && source .venv-train/bin/activate
pip install -r training/requirements.txt

# (Optional) flash-attn for ~30-50% faster training.
pip install flash-attn>=2.5.0
```

That's it — the launch scripts point `OPENR1_SRC` at the bundled `open_r1/` by
default and put it on `PYTHONPATH`. Override `OPENR1_SRC` only to use a different
trainer checkout.

The dataset defaults to the gated HF benchmark (`waybarrios/CRYSTAL`); log in with
`huggingface-cli login` and request access, or pass a local path via `DATASET`.

## The CPR reward

```
correct answer   ->  answer_weight · 1 + step_weight · F1_step
incorrect answer ->  step_weight · F1_step · 0.3
```

A correct answer with faithful reasoning scores highest; a correct answer with
unrelated reasoning is penalized (the "right answer, wrong reasons" case). Paper
defaults: `answer_weight=0.6`, `step_weight=0.4`.

---

## How to run

Every script is configured by environment variables (sane defaults shown). The
default model is Qwen2.5-VL; override `MODEL` for any VLM.

### 1. CPR (single phase)

Trains reasoning directly from a base model with format + accuracy + CPR rewards.

```bash
MODEL="Qwen/Qwen2.5-VL-3B-Instruct" \
DATASET="waybarrios/CRYSTAL" \
NUM_GPUS=4 \
bash scripts/train_cpr.sh
```

InternVL (uses `--max_anyres_num` and the InternVL DeepSpeed config):

```bash
MODEL="OpenGVLab/InternVL3_5-4B" \
MAX_ANYRES_NUM=12 \
bash scripts/train_cpr_internvl.sh
```

### 2. CPR-Curriculum (two phases) — recommended

The curriculum trains accuracy first, then layers in reasoning. It is the most
stable recipe and generalizes across architectures.

**Phase 1 — answer-only** (format + accuracy, no reasoning reward):

```bash
MODEL="Qwen/Qwen2.5-VL-3B-Instruct" \
OUTPUT="./output/phase1-answer-only" \
bash scripts/train_answer_only.sh
```

Evaluate the Phase-1 checkpoints and pick the one with the **best accuracy**
(see [Evaluating checkpoints](#evaluating-checkpoints)).

**Phase 2 — add CPR** from that checkpoint, with a **lower learning rate** so the
accuracy learned in Phase 1 is preserved:

```bash
MODEL="./output/phase1-answer-only/checkpoint-XXX" \  # best Phase-1 checkpoint
LEARNING_RATE=5e-6 \
CAUSAL_ANSWER_WEIGHT=0.65 CAUSAL_STEP_WEIGHT=0.35 \
OUTPUT="./output/phase2-cpr-curriculum" \
bash scripts/train_cpr.sh
```

For InternVL, run `train_answer_only.sh` then `train_cpr_internvl.sh` the same way
(set `MODEL` to the Phase-1 checkpoint and `LEARNING_RATE=5e-6`).

> **Why two phases?** Training reasoning and accuracy together early on is
> unstable — accuracy drops while reasoning variance is high. Phase 1 stabilizes
> the answer; Phase 2's lower LR + higher `answer_weight` add reasoning without
> regressing accuracy.

### 3. SPR (Semantic Process Reward) — ⚠️ experimental

> **Experimental, not used in the paper.** The reported CRYSTAL results use CPR /
> CPR-Curriculum (see the [results tables](../README.md#training-with-causal-process-reward)).
> SPR is an alternative reasoning reward we explored but did not report; kept here
> for reference.

Same as CPR but the reasoning reward matches steps by embedding cosine similarity
instead of the causal interaction:

```bash
MODEL="Qwen/Qwen2.5-VL-3B-Instruct" \
SEMANTIC_THRESHOLD=0.70 \
bash scripts/train_semantic.sh
```

### 4. Answer-only baseline (ablation)

`scripts/train_answer_only.sh` on its own is the ablation that shows the reasoning
reward is necessary — it optimizes only the final answer.

---

## Evaluating checkpoints

After training, run inference with a checkpoint and score it with the metrics:

```bash
# 1. Inference (see ../docs/inference.md) -> predictions/<ckpt>/*.json
python ../inference/crystal_inference.py \
    --model ./output/phase2-cpr-curriculum/checkpoint-XXX \
    --base-url http://localhost:8000/v1 \
    --output-dir predictions/ckpt-XXX

# 2. Metrics (Match F1, Ordered F1, accuracy)
crystal-metrics evaluate predictions/ckpt-XXX references.json --alpha 0.3
```

For the curriculum, evaluate Phase-1 checkpoints to choose the Phase-2 starting
point (best accuracy), and evaluate Phase-2 checkpoints to choose the final model
(best Match F1 without losing accuracy).

## Key hyperparameters

| Setting | Default | Notes |
|---------|---------|-------|
| `NUM_GENERATIONS` | 5 | GRPO group size (G); 4–8 recommended for VLMs |
| `LEARNING_RATE` | 1e-5 (Phase 1) / 5e-6 (Phase 2) | lower in Phase 2 to preserve accuracy |
| `CAUSAL_ANSWER_WEIGHT` / `CAUSAL_STEP_WEIGHT` | 0.6 / 0.4 (0.65 / 0.35 for InternVL Phase 2) | CPR weights |
| reward weights | 2.0 2.0 2.0 | format / accuracy / reasoning |
| `PER_DEVICE_BATCH` × `GRADIENT_ACCUM` | 5 × 2 | effective batch per GPU |

More tuning guidance is in [`docs/GRPO_TRAINING_RECOMMENDATIONS.md`](docs/GRPO_TRAINING_RECOMMENDATIONS.md).

## Models

- **Qwen2.5-VL** — `open_r1/vlm_modules/qwen_module.py`. Image budget via `--max_pixels`/`--min_pixels`.
- **InternVL** — `open_r1/vlm_modules/internvl_module.py`. Image patches via `--max_anyres_num`.
- **GLM-4V** — `open_r1/vlm_modules/glm_module.py`.

Multi-model setup and transformers-version notes: [`docs/MULTI_MODEL_SETUP.md`](docs/MULTI_MODEL_SETUP.md).
Full VQA training reference: [`docs/VQA_TRAINING.md`](docs/VQA_TRAINING.md).
