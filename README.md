<div align="center">

<img src="assets/eccv2026_logo.png" alt="ECCV 2026 — Malmö, Sept 8–12" width="320"/>

# CRYSTAL: Beyond Final Answers

### Transparent Multimodal Reasoning Evaluation

[![Paper](https://img.shields.io/badge/arXiv-2603.13099-b31b1b.svg)](https://arxiv.org/abs/2603.13099)
[![Dataset](https://img.shields.io/badge/Benchmark-6%2C372_instances-green.svg)](#crystal-benchmark)
[![Models](https://img.shields.io/badge/Evaluated-20_MLLMs-orange.svg)](#key-findings)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97_Dataset-CRYSTAL-yellow.svg)](https://huggingface.co/datasets/waybarrios/CRYSTAL)
[![PyPI](https://img.shields.io/pypi/v/crystal-metrics.svg)](https://pypi.org/project/crystal-metrics/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)

**[Wayner Barrios](https://github.com/waybarrios)** &nbsp;&middot;&nbsp; **[SouYoung Jin](https://souyoungjin.github.io/)**

*Dartmouth College*

[Paper](https://arxiv.org/abs/2603.13099) &nbsp;|&nbsp; [Dataset](https://huggingface.co/datasets/waybarrios/CRYSTAL) &nbsp;|&nbsp; [Results](#key-findings) &nbsp;|&nbsp; [Training](#training-with-causal-process-reward)

---

*Your model gets the right answer. But does it actually **reason**?*

</div>

## The Problem: Right Answers, Wrong Reasons

Current vision-language benchmarks only check final answers. This means a model that **guesses correctly** scores the same as one that **reasons correctly**. CRYSTAL exposes this blind spot.

<div align="center">
<img src="assets/teaser_sample0.jpg" width="280"/>

> **Q:** *"Which of the 3 objects is the smallest?"* &nbsp;&mdash;&nbsp; A model answers correctly (C: middle console), but its reasoning states the middle console is *larger* than the others while claiming it's the smallest. **Traditional benchmarks give full credit for this lucky guess.** CRYSTAL catches it (Match F1: 0.15).
</div>

## What is CRYSTAL?

**C**lear **R**easoning via **Y**ielded **S**teps, **T**raceability and **L**ogic &mdash; a diagnostic benchmark with **6,372 instances** that evaluates multimodal reasoning through **verifiable intermediate steps**.

Instead of just asking *"is the answer correct?"*, CRYSTAL asks:
- Did the model identify the right visual elements?
- Did it apply the correct logical steps?
- Are the steps in a coherent order?

### How It Works

<div align="center">
<img src="assets/workflow_dataset.png" width="700"/>

*Multi-agent reasoning pipeline. Four independent MLLMs generate candidate steps, semantically clustered into consensus sequences, validated by a 5th agent, and verified by human annotators.*
</div>

### Benchmark Statistics

| Statistic | Value |
|:--|:--|
| Total instances | **6,372** |
| Avg. reasoning steps per question | **11.6** |
| Step range | 3 &ndash; 42 |
| Difficulty split | Easy 48.5% / Medium 44.1% / Hard 7.4% |
| Sources | MathVision, ScienceQA, RealWorldQA, MMVP, PlotQA |

### Novel Metrics

| Metric | What it measures |
|:--|:--|
| **Match F1** | Step-level precision & recall via semantic similarity matching |
| **Ordered Match F1** | Penalizes disordered reasoning chains (via LIS ratio) |

## Metrics Package (`crystal-metrics`)

The CRYSTAL metrics ship as a pip-installable package ([on PyPI](https://pypi.org/project/crystal-metrics/)) so anyone can score their own models.

```bash
pip install crystal-metrics          # Match F1, Ordered Match F1, Precision, Recall, Accuracy
pip install crystal-metrics[judge]   # + optional LLM judge for free-form answers
```

```python
from crystal_metrics import MLLMReasoningEvaluator

evaluator = MLLMReasoningEvaluator()  # all-distilroberta-v1, τ=0.35 (paper defaults)
m = evaluator.evaluate_single(predicted_steps, reference_steps, alpha=0.3)
print(m.match_f1, m.precision, m.recall, m.ordered_match_f1)
```

Or from the command line:

```bash
crystal-metrics evaluate predictions.json references.json --alpha 0.3
```

It also ships the **RL reward functions** (Causal & Semantic Process Reward) used to train models on CRYSTAL:

```python
from crystal_metrics import causal_process_reward
rewards = causal_process_reward(completions, ground_truths=gts, reference_steps=refs)
```

**Docs:** [installation](docs/installation.md) &middot; [quickstart](docs/quickstart.md) &middot; [metric definitions](docs/metrics.md) &middot; [CLI](docs/cli.md) &middot; [RL rewards](docs/rewards.md). Package source lives in [`src/crystal_metrics/`](src/crystal_metrics/).

## Running Inference

A single **model-agnostic** script ([`inference/crystal_inference.py`](inference/crystal_inference.py)) runs any vision-language model over CRYSTAL via an OpenAI-compatible endpoint — the same code for **Ollama** and **vLLM**, only `--base-url` and `--model` change:

```bash
# Ollama
ollama serve & ; ollama pull qwen2.5vl:7b
python inference/crystal_inference.py --model qwen2.5vl:7b --base-url http://localhost:11434/v1

# vLLM
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8000 &
python inference/crystal_inference.py --model Qwen/Qwen2.5-VL-7B-Instruct --base-url http://localhost:8000/v1
```

It writes per-sample `{reasoning_steps, answer}` JSON (resumable) ready for `crystal-metrics evaluate`. See [inference docs](docs/inference.md).

## Examples

CRYSTAL spans diverse reasoning scenarios &mdash; from spatial understanding to scientific reasoning and mathematical problem-solving:

<table>
<tr>
<td width="33%" align="center">
<img src="assets/dataset_example_sample3.jpg" width="280"/><br/>
<b>Spatial Reasoning</b><br/>
<sub><i>"Is it safe to turn right?"</i> &mdash; Requires identifying traffic signals, pedestrian crossings, and vehicle positions</sub>
</td>
<td width="33%" align="center">
<img src="assets/example_scienceqa_3807.jpg" width="280"/><br/>
<b>Scientific Reasoning</b><br/>
<sub><i>"Which organism would have the most energy available?"</i> &mdash; Requires tracing food web dependencies</sub>
</td>
<td width="33%" align="center">
<img src="assets/example_mathvision_767.jpg" width="280"/><br/>
<b>Mathematical Reasoning</b><br/>
<sub><i>"Which kite is different?"</i> &mdash; Requires systematic comparison of geometric properties</sub>
</td>
</tr>
</table>

<details>
<summary><b>See a full reasoning trace example</b></summary>

**Image:** Egg placement puzzle (2x3 grid + chicken)

<div align="center">
<img src="assets/conservatism_example_sample818.jpg" width="300"/>
</div>

**Question:** *"How many eggs can you place so no two eggs are in the same row or column?"*

**Baseline model (Qwen2.5-VL-3B):**
> *"The answer is 3."* &mdash; Correct answer, but states the goal without explaining *how*.

**After CPR-Curriculum training:**
> *"Step 1: The grid is 2 rows x 3 columns. Step 2: Apply checkerboard placement constraint. Step 3: Place eggs at positions (1,1), (2,2) avoiding shared rows/columns. Step 4: Maximum is 2 per constraint... but wait, with 2 rows we can place at most 2. Re-examine: 3 columns allow diagonal placement. Answer: 3."* &mdash; Structured reasoning with explicit strategy.

</details>

## Key Findings

We evaluated **20 MLLMs** (16 open-source from 1B to 38B, plus GPT-5, GPT-5-mini, GPT-5.2 Instant, and Gemini 2.5 Flash):

### Finding 1: Cherry-Picking is Universal

**19 out of 20 models** produce high-precision but low-recall reasoning &mdash; they say a few correct things but skip most of the required steps.

| Model | Params | Accuracy | Match F1 | Precision | Recall |
|:--|:--:|:--:|:--:|:--:|:--:|
| | | ***Commercial*** | | | |
| GPT-5 | n/a | **57.99%** | 0.612 | 0.925 | 0.479 |
| GPT-5-mini | n/a | 55.59% | **0.773** | **0.978** | 0.669 |
| Gemini 2.5 Flash | n/a | 53.95% | 0.673 | 0.701 | **0.765** |
| GPT-5.2 Instant | n/a | 47.35% | 0.564 | 0.974 | 0.416 |
| | | ***Qwen Family*** | | | |
| Qwen3-VL-8B | 8B | **57.66%** | 0.659 | 0.827 | 0.590 |
| Qwen3-VL-32B | 32B | 49.22% | 0.718 | 0.819 | 0.704 |
| Qwen2.5-VL-32B | 32B | 47.63% | 0.653 | 0.943 | 0.524 |
| Qwen2.5-VL-3B | 3B | 39.85% | 0.480 | 0.898 | 0.347 |
| Qwen3-VL-2B | 2B | 34.15% | 0.595 | 0.726 | 0.535 |
| Qwen2.5-VL-7B | 7B | 30.43% | 0.475 | 0.765 | 0.365 |
| | | ***InternVL Family*** | | | |
| InternVL3.5-8B | 8B | 51.98% | 0.530 | 0.882 | 0.416 |
| InternVL3.5-38B | 38B | 51.21% | 0.612 | 0.892 | 0.498 |
| InternVL3.5-4B | 4B | 37.61% | 0.432 | 0.895 | 0.325 |
| InternVL3.5-2B | 2B | 33.02% | 0.469 | 0.725 | 0.371 |
| InternVL3.5-1B | 1B | 30.13% | 0.330 | 0.616 | 0.243 |
| | | ***Other Open-Source*** | | | |
| Gemma3-12B | 12B | 33.83% | 0.605 | 0.838 | 0.499 |
| Gemma3-4B | 4B | 28.65% | 0.618 | 0.878 | 0.506 |
| Llama 3.2-11B | 11B | 24.83% | 0.471 | 0.713 | 0.379 |
| LLaVA-v1.6-7B | 7B | 24.66% | 0.512 | 0.961 | 0.370 |
| MiniCPM-v2.6-8B | 8B | 25.54% | 0.215 | 0.709 | 0.134 |

> Even GPT-5 (best accuracy: 57.99%) recovers only **47.9% of reference steps**. Models say the right things but omit most of the reasoning.

### Finding 2: Accuracy and Reasoning Diverge

GPT-5 leads accuracy but ranks **8th** in Match F1. GPT-5-mini leads F1 at lower accuracy. **Gemma3-4B outperforms InternVL3.5-38B** in reasoning despite 9.5x fewer parameters.

### Finding 3: No Model Reasons in Order

Among competitive models, **no model preserves more than 60% of matched steps in the correct order**. Models retrieve relevant reasoning steps but fail to organize them coherently.

### Finding 4: Scaling is Non-Monotonic

Bigger models don't uniformly improve: Qwen3-VL-32B achieves better F1 (0.718) but *lower accuracy* (49.22%) than Qwen3-VL-8B (0.659 F1, 57.66% accuracy). Scale can improve answer extraction while suppressing reasoning coverage.


## Training with Causal Process Reward

CRYSTAL isn't just a benchmark &mdash; it enables a new training paradigm. We propose **Causal Process Reward (CPR)**, a multiplicative reward that couples answer correctness with step-level alignment:

$$R_{\text{CPR}} = \begin{cases} a_w + s_w \cdot \text{F1}_{\text{step}} & \text{if answer correct} \\ s_w \cdot \text{F1}_{\text{step}} \cdot \lambda & \text{otherwise} \end{cases}$$

Unlike additive rewards (which let models maximize accuracy by guessing while ignoring reasoning), CPR requires both correct answers **and** faithful reasoning.

### CPR-Curriculum: +32% Match F1 where others fail

| Strategy | Accuracy | Match F1 | Precision | Recall | Ordered F1 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Baseline (Qwen2.5-VL-3B) | 39.85% | 0.480 | 0.898 | 0.347 | 0.434 |
| Composite (additive) | 44.92% | 0.426 | 0.983 | 0.284 | 0.392 |
| Answer-Only | 44.30% | 0.429 | 0.803 | 0.308 | 0.380 |
| **CPR** | 41.40% | **0.633** | 0.975 | 0.489 | **0.560** |
| **CPR-Curriculum** | **47.52%** | **0.633** | 0.963 | **0.493** | **0.560** |

> Additive strategies collapse at step 600 and diverge to NaN by step 1,500. CPR trains stably through 2,800 steps.

<div align="center">
<img src="assets/grpo_f1_trajectory.png" width="48%"/> <img src="assets/grpo_accuracy_trajectory.png" width="48%"/>

*Left: Match F1 trajectory. Right: Accuracy trajectory. Composite collapses at step 600; CPR variants train stably through 2,800 steps.*
</div>

### Cross-Model Generalization: CPR-Curriculum on InternVL3.5-4B

Same two-phase protocol, no architecture-specific tuning. Recall nearly **triples** (0.325 &rarr; 0.811):

| Configuration | Accuracy | Match F1 | Precision | Recall | Ordered F1 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Baseline (InternVL3.5-4B) | 37.61% | 0.432 | 0.895 | 0.325 | 0.387 |
| **CPR-Curriculum** | **45.76%** | **0.833** | **0.903** | **0.811** | **0.719** |
| &Delta; | +8.15 | +0.401 | +0.008 | +0.486 | +0.332 |

> CPR-Curriculum generalizes across architectures: **+93% Match F1** on InternVL3.5-4B without any model-specific tuning.

## Ablation Studies

### Encoder and Threshold Selection

We tested 4 sentence encoders across 5 thresholds (100 experiments). DistilRoBERTa-v1 dominates with 4-8pp advantages across all model families. Model rankings remain stable across all configurations.

<div align="center">
<img src="assets/ablation_encoders.png" width="700"/>

*Encoder comparison across thresholds. DistilRoBERTa-v1 consistently outperforms alternatives. Threshold variation yields only 2-3pp swings.*
</div>

### Human Agreement Study

On 100 adversarially sampled step pairs, the encoder achieves **84% agreement** with a human annotator, with **100% agreement below the threshold** (zero false matches on semantically unrelated steps).

| Similarity Band | Pairs | Agreement | Note |
|:--|:--:|:--:|:--|
| < 0.20 | 33 | 100.0% | No false matches |
| [0.20, 0.35) | 14 | 100.0% | No false matches |
| [0.35, 0.50) | 19 | 68.4% | Borderline zone |
| [0.50, 0.70) | 25 | 64.0% | Borderline zone |
| >= 0.70 | 9 | 88.9% | Clear matches |
| **Overall** | **100** | **84.0%** | Cohen's kappa = 0.534 |

## Supplementary Examples

<details>
<summary><b>CRYSTAL Benchmark Example (MathVision)</b></summary>

<div align="center">
<img src="assets/crystal_ex1_sample1265.jpg" width="400"/>
</div>

**Q:** *"Anna starts in the direction of the arrow. At each crossing she turns either right or left. At the first crossing she turns right, at the next left, then left again, then right, then left and left again. What will she find at the next crossing?"*

**Ground Truth:** A

**Reference Reasoning Steps (12 steps):**
1. Anna starts in the direction of an arrow
2. She turns at each crossing either right or left
3. First crossing: turns right
4. Second crossing: left
5. Third crossing: left again
6. Fourth crossing: right
7. Fifth crossing: left
8. Sixth crossing: left again
9. Question asks what she finds at the next crossing
10. Answer format is multiple-choice (A, B, C, D, E)
11. *[2 more steps for spatial tracing]*

</details>

<details>
<summary><b>GRPO Qualitative Improvement</b></summary>

<div align="center">
<img src="assets/grpo_example_sample3.jpg" width="400"/>
</div>

**Q:** *"What color is the traffic light?"* &nbsp; **GT:** Green

**Baseline** (2 steps, 33% coverage):
> 1. Noted the traffic light is green. 2. Identified the traffic light as green. &mdash; *Cherry-picks high-confidence steps only.*

**After GRPO** (4 steps, 67% coverage):
> 1. The traffic lights in the scene are green. 2. The traffic lights are positioned at an intersection. 3. The traffic lights are part of the traffic control system. 4. The traffic lights are visible from a distance. &mdash; *More comprehensive reasoning (+34pp coverage).*

</details>

## Dataset

The CRYSTAL benchmark is available on HuggingFace:

> 🤗 **[huggingface.co/datasets/waybarrios/CRYSTAL](https://huggingface.co/datasets/waybarrios/CRYSTAL)** &mdash; 6,372 instances with reference reasoning steps, released under CC-BY-NC-SA-4.0.
>
> *Evaluation scripts and pre-trained CPR-Curriculum checkpoints coming soon.*

## Why CRYSTAL Matters

| If you are... | CRYSTAL helps you... |
|:--|:--|
| **Building MLLMs** | Diagnose *where* reasoning fails (perception vs. logic) |
| **Evaluating models** | Go beyond accuracy to measure reasoning transparency |
| **Training with RL** | Use CPR rewards to improve reasoning without manual step annotations |
| **Researching scaling** | Understand non-monotonic trade-offs between accuracy and reasoning |
| **Deploying AI systems** | Identify models that guess vs. models that actually reason |

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

This project is released under the MIT License.
