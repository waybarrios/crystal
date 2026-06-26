# VQA Training with GRPO

This guide explains how to train Vision-Language Models on Visual Question Answering (VQA) tasks using GRPO (Group Relative Policy Optimization) with reasoning steps.

## Quick Start

```bash
# 1. Install dependencies (transformers 4.52.4 recommended for Qwen2.5-VL)
pip install transformers==4.52.4 datasets>=3.0.0 trl peft accelerate

# 2. Run training
bash train_vqa_multi.sh  # Uses Qwen2.5-VL-3B-Instruct by default
```

## Overview

The VQA training mode uses a JSON output format with reasoning steps and final answers, evaluated using three reward functions:
- **Format Reward**: Validates JSON structure
- **Accuracy Reward**: Evaluates answer correctness (rule-based or LLM-based with `gpt-oss:20b`)
- **Reasoning Reward**: Measures reasoning quality using Match F1

**Key Features:**
- ✅ HuggingFace dataset support with `load_from_disk()`
- ✅ Multi-GPU training with Accelerate
- ✅ Optional LLM judge for semantic verification (default: `gpt-oss:20b`)
- ✅ Reproducible training with seed control (via `--seed` from TrainingArguments)
- ✅ Dataset shuffling with `--shuffle_train_dataset`
- ✅ Support for Qwen2.5-VL, Aria, and GLM-4V models
- ✅ **Automatic dataset caching** - Processed datasets are cached for instant loading on subsequent runs

## Dataset Requirements

### HuggingFace Dataset Format

Your dataset should be saved using `datasets.save_to_disk()` with the following structure:

```python
Dataset({
    features: ['image', 'question', 'answer', 'reference_steps', 'choices', 'options'],
    num_rows: N
})
```

**Required fields:**
- `image`: PIL Image object
- `question`: str - The question text
- `answer`: str - Ground truth answer
- `reference_steps`: List[str] - Reference reasoning steps for the answer

**Optional fields:**
- `choices`: List[str] - Multiple choice options (if applicable)
- `options`: List[str] - Alternative format for multiple choice options
- `source`: str - Dataset source identifier

### Example Dataset Entry

```python
{
    'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1024x730>,
    'question': 'what is the brand of phone?',
    'answer': 'nokia',
    'reference_steps': [
        'Check for visible text related to brands on the mobile phone in the image.',
        "Identify the brand text 'NOKIA' in close proximity to the mobile phone.",
        'Verify there are no other text labels or logos visible on the mobile phone.',
        'Return the exact answer string.'
    ],
    'choices': [],
    'options': [],
    'source': 'textvqa'
}
```

## Installation

### Prerequisites

```bash
# Install base requirements
# NOTE: For Qwen2.5-VL with Flash Attention support, use transformers 4.52.4
# For all models (Qwen2.5-VL + Aria + GLM-4V), use transformers >= 4.56.0
pip install transformers==4.52.4  # or transformers>=4.56.0 for all models
pip install torch datasets pillow pyyaml

# Install TRL and GRPO dependencies
pip install trl peft accelerate
pip install datasets>=3.0.0  # Required by TRL

# Install mllm_evaluator dependencies
pip install sentence-transformers pandas numpy openai tqdm
```

**Transformers Version Notes:**
- **transformers 4.52.4**: ✅ Best for Qwen2.5-VL with Flash Attention support
- **transformers ≥4.56.0**: ✅ Supports all models (Qwen2.5-VL + Aria + GLM-4V), but Flash Attention monkey patch may be skipped (still works with standard attention)

### Optional: Ollama Setup (for LLM-based accuracy grading)

If you want to use LLM-based semantic verification for free-form text answers:

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve

# Pull the gpt-oss model (in another terminal)
ollama pull gpt-oss:20b
```

**Note**: The accuracy calculator uses rule-based matching by default for speed. Enable LLM-based semantic verification with `--use_llm_judge` if needed. The default LLM judge model is `gpt-oss:20b` (configurable via `--llm_judge_model`).

## Training Commands

### Basic VQA Training (Qwen2.5-VL-3B)

```bash
python src/open-r1-multimodal/src/open_r1/grpo_rec.py \
    --model_name_or_path "Qwen/Qwen2.5-VL-3B-Instruct" \
    --dataset_name "/path/to/your/huggingface/dataset" \
    --use_huggingface_dataset \
    --task_type "vqa" \
    --reward_funcs "format" "accuracy" \
    --output_dir "./output/qwen2.5-vl-3b-vqa-grpo" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --seed 42 \
    --shuffle_train_dataset \
    --logging_steps 10 \
    --save_steps 500 \
    --max_pixels 12845056 \
    --min_pixels 3136
```

### VQA Training with LLM Judge (Enhanced Accuracy)

Use LLM-based semantic verification for free-form text answers:

```bash
python src/open-r1-multimodal/src/open_r1/grpo_rec.py \
    --model_name_or_path "Qwen/Qwen2.5-VL-3B-Instruct" \
    --dataset_name "/path/to/your/huggingface/dataset" \
    --use_huggingface_dataset \
    --task_type "vqa" \
    --reward_funcs "format" "accuracy" \
    --use_llm_judge \
    --llm_judge_model "gpt-oss:20b" \
    --llm_judge_base_url "http://localhost:11434/v1" \
    --output_dir "./output/qwen2.5-vl-3b-vqa-llm-judge" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --logging_steps 10 \
    --save_steps 500
```

### With Reasoning Reward

To also evaluate reasoning quality (requires reference_steps in dataset):

```bash
python src/open-r1-multimodal/src/open_r1/grpo_rec.py \
    --model_name_or_path "Qwen/Qwen2.5-VL-3B-Instruct" \
    --dataset_name "/path/to/your/huggingface/dataset" \
    --use_huggingface_dataset \
    --task_type "vqa" \
    --reward_funcs "format" "accuracy" "reasoning" \
    --output_dir "./output/qwen2.5-vl-3b-vqa-reasoning" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --logging_steps 10 \
    --save_steps 500 \
    --max_pixels 12845056 \
    --min_pixels 3136
```

### Multi-GPU Training

```bash
accelerate launch --num_processes 4 \
    src/open-r1-multimodal/src/open_r1/grpo_rec.py \
    --model_name_or_path "Qwen/Qwen2.5-VL-3B-Instruct" \
    --dataset_name "/path/to/your/huggingface/dataset" \
    --use_huggingface_dataset \
    --task_type "vqa" \
    --reward_funcs "format" "accuracy" "reasoning" \
    --output_dir "./output/qwen2.5-vl-3b-vqa-grpo" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --bf16 \
    --logging_steps 10 \
    --save_steps 500
```

### With DeepSpeed ZeRO-3 (for larger models like 7B)

```bash
deepspeed --num_gpus 4 \
    src/open-r1-multimodal/src/open_r1/grpo_rec.py \
    --model_name_or_path "Qwen/Qwen2.5-VL-7B-Instruct" \
    --dataset_name "/path/to/your/huggingface/dataset" \
    --use_huggingface_dataset \
    --task_type "vqa" \
    --reward_funcs "format" "accuracy" "reasoning" \
    --output_dir "./output/qwen2.5-vl-7b-vqa-grpo" \
    --deepspeed configs/deepspeed_zero3.json \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-5 \
    --bf16 \
    --logging_steps 10 \
    --save_steps 500
```

## Key Parameters

### Task Configuration

- `--task_type "vqa"`: **Required** - Sets the task to VQA mode with JSON output
- `--use_huggingface_dataset`: **Required** - Enables HuggingFace dataset loading
- `--dataset_name`: Path to your HuggingFace dataset directory

### Dataset Shuffling & Reproducibility

Control dataset shuffling and random seed for reproducible training:

- `--shuffle_train_dataset`: Whether to shuffle the training dataset (default: True)
- `--seed`: Random seed for reproducibility (default: 42)

**Reproducibility:**
- Sets seed for Python `random`, NumPy, PyTorch (CPU and CUDA)
- Ensures deterministic shuffling and sampling
- Use the same seed to reproduce exact training order

**Example usage:**
```bash
# Reproducible training with specific seed
--seed 42 --shuffle_train_dataset

# Reproducible training without shuffling (sequential order)
--seed 42 --no-shuffle_train_dataset

# Different shuffle order
--seed 123 --shuffle_train_dataset
```

### Reward Functions

Choose from the following reward functions (can use multiple):

- `"format"`: Validates JSON structure ({"reasoning_steps": [], "answer": ""})
- `"accuracy"`: Evaluates answer correctness using accuracy_calculator
- `"reasoning"`: Evaluates reasoning quality using Match F1 (requires reference_steps)

**Recommended combinations:**
- Fast training: `--reward_funcs "format" "accuracy"`
- High quality: `--reward_funcs "format" "accuracy" "reasoning"`

### LLM Judge Configuration

Configure LLM-based semantic verification for free-form text answers:

- `--use_llm_judge`: Enable LLM judge for accuracy evaluation (default: False)
- `--llm_judge_model`: Ollama model name (default: "gpt-oss:20b")
  - Options: `"gpt-oss:20b"`, `"llama3.2"`, `"llama3.1"`, `"mistral"`, `"phi3"`, `"gemma2"`
- `--llm_judge_base_url`: Ollama API endpoint (default: "http://localhost:11434/v1")

**When to use LLM Judge:**
- Dataset contains free-form text answers (not just multiple choice or numbers)
- Answers require semantic understanding (e.g., "The capital of France is Paris" vs "Paris")
- Higher accuracy evaluation quality is needed
- Ollama is installed and running locally

**Trade-offs:**
- ✅ Better semantic understanding of text answers
- ✅ Handles paraphrasing and variations
- ❌ Slower training (LLM inference per sample)
- ❌ Requires Ollama setup

**Example usage:**
```bash
# Fast training (rule-based only)
--reward_funcs "accuracy"

# Semantic training (LLM-based with gpt-oss:20b)
--reward_funcs "accuracy" --use_llm_judge --llm_judge_model "gpt-oss:20b"

# Using a different model
--reward_funcs "accuracy" --use_llm_judge --llm_judge_model "llama3.2"
```

### GPU Selection

The training script supports selecting specific GPUs via `CUDA_VISIBLE_DEVICES`. Edit `train_vqa_multi.sh`:

```bash
# Use specific GPUs (GPUs 0, 1, 2, 3)
GPU_IDS="0,1,2,3"

# Use only 2 GPUs (GPUs 0 and 2)
GPU_IDS="0,2"

# Use all available GPUs
GPU_IDS=""
```

**Alternative**: Set `CUDA_VISIBLE_DEVICES` before running:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./train_vqa_multi.sh
```

**Important**: Ensure `num_generations` is compatible with your GPU count:
- 2 GPUs × 4 batch = 8 → use `--num_generations 2` or `4` or `8`
- 3 GPUs × 4 batch = 12 → use `--num_generations 2`, `3`, `4`, `6`, or `12`
- 4 GPUs × 4 batch = 16 → use `--num_generations 2`, `4`, `8`, or `16`

### Model Configuration

- `--model_name_or_path`: Model to fine-tune
  - **Qwen2.5-VL** (Recommended):
    - `"Qwen/Qwen2.5-VL-3B-Instruct"` - 3B parameters, efficient
    - `"Qwen/Qwen2.5-VL-7B-Instruct"` - 7B parameters, higher quality
  - **Qwen2-VL**:
    - `"Qwen/Qwen2-VL-7B-Instruct"`
  - **InternVL**: `"OpenGVLab/InternVL2-8B"`
  - **GLM**: `"THUDM/glm-4v-9b"`

**Model Selection Guide:**
- **Qwen2.5-VL-3B**: Best for fast training, limited resources (24GB GPU)
- **Qwen2.5-VL-7B**: Best for high quality, more resources (80GB or multi-GPU)
- **InternVL/GLM**: Alternative architectures

- `--max_pixels`: Maximum pixels for image processing (default: 12845056)
- `--min_pixels`: Minimum pixels for image processing (default: 3136)

### Training Configuration

- `--num_train_epochs`: Number of training epochs
- `--per_device_train_batch_size`: Batch size per GPU
- `--gradient_accumulation_steps`: Gradient accumulation steps
- `--learning_rate`: Learning rate (recommended: 1e-5 to 5e-6)
- `--bf16` or `--fp16`: Use mixed precision training

### LoRA/QLoRA (Optional)

To use parameter-efficient fine-tuning (recommended for limited GPU memory):

```bash
python src/open-r1-multimodal/src/open_r1/grpo_rec.py \
    --model_name_or_path "Qwen/Qwen2.5-VL-3B-Instruct" \
    --dataset_name "/path/to/your/huggingface/dataset" \
    --use_huggingface_dataset \
    --task_type "vqa" \
    --reward_funcs "format" "accuracy" \
    --output_dir "./output/qwen2.5-vl-3b-vqa-lora" \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --target_modules "q_proj" "v_proj" "k_proj" "o_proj" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-4
```

## Expected Model Output

The model will generate predictions in JSON format:

```json
{
  "reasoning_steps": [
    "The image shows a bar chart with two groups of bars, each representing a year.",
    "The first group of bars is labeled '2003', and the second group is labeled '2010'.",
    "The question asks for the label of the second group of bars from the left.",
    "The second group of bars is clearly labeled '2010'."
  ],
  "answer": "2010"
}
```

### Reasoning Steps Guidelines

The prompt instructs the model to generate reasoning steps that:
- Are single-clause sentences (≤14 words)
- Contain directly checkable facts or cue-based inferences
- Are anchored to visible cues in the image
- Build towards the final answer
- Avoid chains like "because/therefore"

### Answer Format

The answer should follow these rules:
- **Multiple choice**: Return only the letter (e.g., "B") if options have letters
- **Numeric**: Include units and follow requested precision
- **Text**: Provide exact answer grounded in visible content
- **Ambiguous**: Return "insufficient information" if unclear

## Debugging and Monitoring

### Enable Debug Mode

To log detailed reward information:

```bash
export DEBUG_MODE=true
export LOG_PATH="./logs/training_debug.txt"

python src/open-r1-multimodal/src/open_r1/grpo_rec.py \
    --model_name_or_path "Qwen/Qwen2-VL-7B-Instruct" \
    --dataset_name "/path/to/your/huggingface/dataset" \
    --use_huggingface_dataset \
    --task_type "vqa" \
    --reward_funcs "format" "accuracy" "reasoning" \
    --output_dir "./output/qwen2vl-vqa-grpo"
```

This will create log files:
- `training_debug_format_vqa.txt`: Format reward logs
- `training_debug_accuracy_vqa.txt`: Accuracy reward logs
- `training_debug_reasoning_vqa.txt`: Reasoning reward logs

### Monitor Training

Watch the logs for:
- Format reward: Should quickly reach ~1.0 (model learns JSON format)
- Accuracy reward: Should gradually improve (0.0 to 1.0)
- Reasoning reward: Should gradually improve (Match F1 score)

## Important Considerations

### 1. Dataset Preparation

**Ensure your dataset has:**
- Valid PIL Images (RGB format)
- Non-empty questions and answers
- High-quality reference_steps (if using reasoning reward)
- Consistent formatting for multiple choice options

**Convert your dataset:**
```python
from datasets import Dataset, load_from_disk

# Example conversion
data = {
    'image': list_of_pil_images,
    'question': list_of_questions,
    'answer': list_of_answers,
    'reference_steps': list_of_reference_steps,
    'choices': list_of_choices,
    'options': list_of_options,
}

dataset = Dataset.from_dict(data)
dataset.save_to_disk("/path/to/save/dataset")
```

### 2. Memory Management

**For Qwen2.5-VL-3B:**
- Single GPU (24GB): batch_size=4, gradient_accumulation=4
- Multi-GPU (4x24GB): batch_size=4 per GPU, gradient_accumulation=2
- With LoRA: batch_size=8, gradient_accumulation=2

**For 7B models:**
- Single GPU (24GB): batch_size=2, gradient_accumulation=8
- Single GPU (80GB): batch_size=8, gradient_accumulation=2
- Multi-GPU (4x24GB): batch_size=2 per GPU, gradient_accumulation=4
- With DeepSpeed ZeRO-3: batch_size=1, gradient_accumulation=16

**For larger models (13B+):**
- Use DeepSpeed ZeRO-3 or QLoRA
- Reduce max_pixels if OOM occurs

**Memory tips:**
- Qwen2.5-VL-3B requires ~12GB VRAM for inference + ~8GB for training
- Qwen2.5-VL-7B requires ~28GB VRAM for inference + ~20GB for training
- Use gradient checkpointing to reduce memory: `--gradient_checkpointing`

### 3. Training Speed

**Factors affecting speed:**
- Reasoning reward is slower (semantic similarity computation)
- Use `--reward_funcs "format" "accuracy"` for faster training
- Batch size and gradient accumulation affect throughput

**Optimization tips:**
- Use bf16 if supported (`--bf16`)
- Reduce max_pixels if not needed
- Use gradient checkpointing for larger models

### 4. Reward Balancing

If using multiple rewards, they are combined. Monitor each reward separately:
- Format should reach ~1.0 quickly (within first epoch)
- Accuracy should steadily improve
- Reasoning reward may be noisy; consider it a secondary signal

### 5. Prompt Engineering

The VQA prompt is fixed in the code. If you need modifications:
- Edit `qwen_module.py:72-107`
- Ensure `{USER_INSTRUCTION}` placeholder remains
- Maintain JSON schema in instructions

### 6. Reproducibility

**For reproducible training:**
- Always set `--seed` to a fixed value (e.g., 42)
- Use `--shuffle_train_dataset` for better generalization
- Same seed + same data order = identical training trajectory

**Important notes:**
- Seed is set for Python random, NumPy, PyTorch (CPU + CUDA)
- Distributed training may have slight variations due to parallelism
- For exact reproducibility, use single GPU with fixed seed

**Example reproducible run:**
```bash
# Run 1
--seed 42 --shuffle_train_dataset

# Run 2 (should produce identical results)
--seed 42 --shuffle_train_dataset

# Run 3 (different shuffle order)
--seed 123 --shuffle_train_dataset
```

### 7. Evaluation

**During training:**
- Monitor reward trends in logs/tensorboard
- Check sample outputs in debug logs

**After training:**
- Use mllm_evaluator to evaluate on test set
- Compare Match F1 and accuracy metrics

## Troubleshooting

### Issue: "cannot import name 'AriaForConditionalGeneration'" or "cannot import name 'Glm4vForConditionalGeneration'"

**Cause**: Using transformers version that doesn't include newer models (Aria, GLM-4V).

**Solution**: Upgrade transformers to latest version:
```bash
pip install --upgrade transformers>=4.56.0
```

**Note**: With transformers ≥4.56.0, Flash Attention monkey patch may be skipped (you'll see a warning), but training will still work with standard attention.

### Issue: "cannot import name 'Qwen2_5_VLVisionFlashAttention2'"

**Cause**: Using transformers version with breaking changes in Flash Attention.

**Solution**: Use transformers 4.52.4 for best Qwen2.5-VL Flash Attention support:
```bash
pip uninstall transformers -y
pip install transformers==4.52.4
```

### Issue: "argument --seed: conflicting option string"

**Cause**: Duplicate seed parameter definition (fixed in latest version).

**Solution**: This was a bug that has been fixed. The `--seed` parameter is now from `TrainingArguments` (standard Hugging Face parameter).

### Issue: "pip's dependency resolver" - datasets version conflict

**Cause**: TRL requires datasets>=3.0.0.

**Solution**:
```bash
pip install --upgrade datasets
```

### Issue: "Unsupported file type"

**Solution**: Ensure you use `--use_huggingface_dataset` flag for HuggingFace datasets.

### Issue: Format reward always 0.0

**Cause**: Model not generating valid JSON.

**Solutions:**
- Check model is generating output (debug logs)
- Ensure sufficient training steps
- Try lower learning rate
- Increase format reward weight

### Issue: OOM (Out of Memory)

**Error Example**: `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 148.00 MiB. GPU 2 has a total capacity of 79.25 GiB...`

**Critical Solutions** (apply in order):

1. **Reduce batch size and increase gradient accumulation**:
   ```bash
   # Original (OOM)
   --per_device_train_batch_size 4 --gradient_accumulation_steps 2

   # Memory optimized (same effective batch size)
   --per_device_train_batch_size 1 --gradient_accumulation_steps 8
   ```

2. **Enable gradient checkpointing** (saves ~40% memory):
   ```bash
   --gradient_checkpointing
   ```

3. **Reduce image resolution** (cut max_pixels in half):
   ```bash
   # Original
   --max_pixels 12845056

   # Reduced (saves ~50% memory per image)
   --max_pixels 6422528
   ```

4. **Set PyTorch memory optimization**:
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   ```

5. **Use fewer GPUs** (if you have limited VRAM):
   ```bash
   # Use only 2 GPUs instead of 4
   GPU_IDS="0,1"
   ```

6. **Advanced: Use DeepSpeed ZeRO-3** (for 7B+ models):
   ```bash
   --deepspeed configs/deepspeed_zero3.json
   ```

**Memory Usage Guide**:
- **Qwen2.5-VL-3B**: ~20-25GB per GPU (with optimizations)
- **Qwen2.5-VL-7B**: ~40-50GB per GPU (requires DeepSpeed ZeRO-3)

**Updated train_vqa_multi.sh** (memory optimized):
- `per_device_train_batch_size: 1` (was 4)
- `gradient_accumulation_steps: 8` (was 2)
- `max_pixels: 6422528` (was 12845056)
- `gradient_checkpointing: enabled`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

### Issue: Reasoning reward is 0.0

**Causes:**
- Dataset missing reference_steps
- Model not generating reasoning_steps
- Steps not semantically similar to references

**Solutions:**
- Verify dataset has reference_steps
- Check debug logs for model outputs
- Consider removing reasoning reward initially

### Issue: Training is very slow

**Solutions:**
- Remove reasoning reward (most expensive)
- Increase batch size if memory allows
- Use multiple GPUs
- Reduce dataset size for testing

### Issue: "The global train batch size must be evenly divisible by the number of generations"

**Error Example**: `ValueError: The global train batch size (3 x 4) must be evenly divisible by the number of generations per prompt (8)`

**Cause**: GRPO requires `(num_gpus × batch_size)` to be evenly divisible by `num_generations` (default: 8)

**Solution**: Add `--num_generations` parameter with a valid value:
```bash
# For 3 GPUs × 4 batch size = 12 total batch size
# Valid options: 2, 3, 4, 6, or 12
--num_generations 4  # Recommended

# Or adjust batch size instead
--per_device_train_batch_size 2  # 3 GPUs × 2 = 6 (divisible by 2, 3, 6)
```

### Issue: "TypeError: VLMGRPOTrainer._get_train_sampler() takes 1 positional argument but 2 were given"

**Cause**: Method signature incompatibility with newer transformers versions.

**Solution**: This has been fixed in the latest code. The `_get_train_sampler()` method now accepts an optional `train_dataset` parameter for compatibility with transformers 4.52.4+.

## Dataset Caching

**Automatic Caching**: The dataset is automatically cached after the first load to speed up subsequent training runs.

**How it works:**
- First run: Dataset is loaded from source, converted, shuffled, and cached (~1-2 minutes for 30K samples)
- Subsequent runs: Dataset loads instantly from cache (<1 second)
- Cache location: `.dataset_cache/` directory next to your dataset
- Cache key: Based on dataset path, seed, and shuffle settings

**Output on first run:**
```
Loading HuggingFace dataset from: /path/to/dataset
This may take a while if dataset is large or on network storage...
✓ Dataset loaded! Total rows: 30312
Converting dataset to internal format...
  Progress: 3031/30312 (10.0%)
  ...
✓ Loaded 30312 samples from HuggingFace dataset
Dataset shuffled with seed 42
Total samples in dataset: 30312
💾 Saving dataset to cache: .dataset_cache/dataset_cache_abc123.pkl
✓ Cache saved! Future runs will load instantly.
```

**Output on subsequent runs:**
```
📦 Loading dataset from cache: .dataset_cache/dataset_cache_abc123.pkl
✓ Loaded 30312 samples from cache!
```

**Clear cache** (if you update your dataset):
```bash
rm -rf /path/to/dataset/.dataset_cache/
```

## Example Training Script

Create `train_vqa.sh`:

```bash
#!/bin/bash

# Configuration
MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
DATASET="/path/to/your/dataset"
OUTPUT="./output/qwen2.5-vl-3b-vqa-$(date +%Y%m%d_%H%M%S)"
NUM_GPUS=4

# Reproducibility
SEED=42
SHUFFLE_DATASET=true

# LLM Judge Configuration (optional)
USE_LLM_JUDGE=false  # Set to true to enable LLM judge
LLM_JUDGE_MODEL="gpt-oss:20b"
LLM_JUDGE_URL="http://localhost:11434/v1"

# Enable debug logging
export DEBUG_MODE=true
export LOG_PATH="$OUTPUT/training_debug.txt"

# Create output directory
mkdir -p $OUTPUT

# Build LLM judge arguments
LLM_JUDGE_ARGS=""
if [ "$USE_LLM_JUDGE" = true ]; then
    LLM_JUDGE_ARGS="--use_llm_judge --llm_judge_model $LLM_JUDGE_MODEL --llm_judge_base_url $LLM_JUDGE_URL"
    echo "LLM Judge enabled with model: $LLM_JUDGE_MODEL"
fi

# Build shuffle argument
SHUFFLE_ARGS=""
if [ "$SHUFFLE_DATASET" = true ]; then
    SHUFFLE_ARGS="--shuffle_train_dataset"
fi

# Train with accelerate
accelerate launch --num_processes $NUM_GPUS \
    src/open-r1-multimodal/src/open_r1/grpo_rec.py \
    --model_name_or_path $MODEL \
    --dataset_name $DATASET \
    --use_huggingface_dataset \
    --task_type "vqa" \
    --reward_funcs "format" "accuracy" "reasoning" \
    --seed $SEED \
    $SHUFFLE_ARGS \
    $LLM_JUDGE_ARGS \
    --output_dir $OUTPUT \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --bf16 \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 3 \
    --warmup_steps 100 \
    --max_grad_norm 1.0 \
    --logging_dir $OUTPUT/logs \
    --report_to "tensorboard" \
    2>&1 | tee $OUTPUT/training.log

echo "Training completed! Output saved to: $OUTPUT"
```

Run with:
```bash
chmod +x train_vqa.sh
./train_vqa.sh
```

### Quick Start Script (Single GPU)

For quick testing on a single GPU:

```bash
#!/bin/bash
# quick_train.sh

python src/open-r1-multimodal/src/open_r1/grpo_rec.py \
    --model_name_or_path "Qwen/Qwen2.5-VL-3B-Instruct" \
    --dataset_name "/path/to/your/dataset" \
    --use_huggingface_dataset \
    --task_type "vqa" \
    --reward_funcs "format" "accuracy" \
    --output_dir "./output/qwen2.5-vl-3b-test" \
    --seed 42 \
    --shuffle_train_dataset \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --bf16 \
    --logging_steps 10 \
    --save_steps 1000
```

## References

- **mllm_evaluator**: See `mllm_evaluator/readme.md` for detailed evaluation metrics
- **Original VLM-R1**: Refer to repository README for base model training
- **TRL Documentation**: https://huggingface.co/docs/trl/

## Support

For issues or questions:
1. Check debug logs (`DEBUG_MODE=true`)
2. Verify dataset format matches requirements
3. Review error messages in training logs
4. Consult mllm_evaluator documentation for reward function details
