# Multi-Model Support Setup Guide

This guide explains how to enable support for **Qwen2.5-VL**, **Aria**, and **GLM-4V** models simultaneously in VLM-R1.

## Quick Setup

Run these commands on your server to upgrade transformers and enable all models:

```bash
cd /gpudata3/Wayner/VLM-R1

# Upgrade transformers to latest version (supports all models)
pip uninstall transformers -y
pip install transformers>=4.56.0

# Verify installation
python -c "from transformers import Qwen2_5_VLForConditionalGeneration, AriaForConditionalGeneration, Glm4vForConditionalGeneration; print('✓ All models available!')"
```

## What Changed?

### 1. **Version-Compatible Flash Attention Monkey Patch**
   - File: `src/open-r1-multimodal/src/open_r1/qwen2_5vl_monkey_patch.py`
   - Now handles both old (transformers ≤4.52.4) and new (transformers ≥4.56.0) versions
   - Automatically detects available Flash Attention classes
   - Gracefully skips monkey patch if Flash Attention isn't available

### 2. **All Models Enabled**
   - GLM-4V support re-enabled (requires transformers ≥4.56.0)
   - Aria support re-enabled (requires transformers ≥4.56.0)
   - Qwen2.5-VL fully supported (requires transformers ≥4.46.0)

### 3. **No Breaking Changes**
   - Existing training scripts work without modification
   - Flash Attention is automatically used when available
   - Backward compatible with older transformers versions (with limited model support)

## Supported Model Matrix

| Model                  | transformers 4.46.0 | transformers 4.52.4 | transformers ≥4.56.0 |
|------------------------|---------------------|---------------------|----------------------|
| Qwen2.5-VL-3B          | ✅                   | ✅                   | ✅                    |
| Qwen2.5-VL-7B          | ✅                   | ✅                   | ✅                    |
| Aria                   | ❌                   | ❌                   | ✅                    |
| GLM-4V                 | ❌                   | ❌                   | ✅                    |
| InternVL               | ✅                   | ✅                   | ✅                    |
| Flash Attention (Qwen) | ✅                   | ✅                   | ⚠️ (auto-detect)      |

**Legend:**
- ✅ Fully supported
- ❌ Not available
- ⚠️ Available but may require additional setup

## Training with Different Models

### Qwen2.5-VL (Current Setup - Working)
```bash
./train_vqa_multi.sh  # Uses Qwen/Qwen2.5-VL-3B-Instruct
```

### Aria
```bash
accelerate launch \
    --num_processes 4 \
    ${SRC_DIR}/open_r1/grpo_rec.py \
    --model_name_or_path "rhymes-ai/Aria" \
    --dataset_name "/path/to/dataset" \
    --use_huggingface_dataset \
    --task_type "vqa" \
    --reward_funcs "format" "accuracy" "reasoning"
```

### GLM-4V
```bash
accelerate launch \
    --num_processes 4 \
    ${SRC_DIR}/open_r1/grpo_rec.py \
    --model_name_or_path "THUDM/glm-4v-9b" \
    --dataset_name "/path/to/dataset" \
    --use_huggingface_dataset \
    --task_type "vqa" \
    --reward_funcs "format" "accuracy" "reasoning"
```

## Flash Attention Notes

### For transformers ≥4.56.0:
- Flash Attention may work differently than older versions
- The monkey patch will auto-detect and skip if not compatible
- You'll see one of these messages:
  - `✓ Qwen2.5-VL Flash Attention monkey patch applied successfully` ← Flash Attention is active
  - `⚠ Skipping Flash Attention monkey patch (not available in this transformers version)` ← Standard attention used

### Performance Impact:
- **With Flash Attention**: ~30-50% faster training
- **Without Flash Attention**: Standard PyTorch attention (still works, just slower)

## Troubleshooting

### Issue: "cannot import name 'Glm4vForConditionalGeneration'"
**Solution:** Upgrade transformers to ≥4.56.0
```bash
pip install --upgrade transformers
```

### Issue: "cannot import name 'AriaForConditionalGeneration'"
**Solution:** Upgrade transformers to ≥4.56.0
```bash
pip install --upgrade transformers
```

### Issue: Flash Attention monkey patch warning
**Solution:** This is normal for newer transformers versions. Training will still work with standard attention.

### Issue: CUDA out of memory with Flash Attention disabled
**Solution:**
1. Reduce `per_device_train_batch_size`
2. Increase `gradient_accumulation_steps`
3. Reduce `max_pixels` parameter

## Recommended Setup

**For maximum compatibility and performance:**

```bash
# Install latest versions
pip install transformers>=4.56.0
pip install flash-attn>=2.5.0  # Optional but recommended for speed
pip install datasets>=3.0.0
pip install trl>=0.18.0

# Verify all imports
python -c "
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AriaForConditionalGeneration,
    Glm4vForConditionalGeneration
)
print('✓ All models available!')
"
```

## Current Training Status

Your current setup with Qwen2.5-VL-3B should continue working after upgrade:

```bash
cd /gpudata3/Wayner/VLM-R1

# Upgrade transformers
pip install --upgrade transformers

# Run training (same command as before)
./train_vqa_multi.sh
```

The training script will automatically:
1. Detect available models (Qwen2.5-VL, Aria, GLM-4V)
2. Apply Flash Attention monkey patch if available
3. Use optimal attention mechanism for your transformers version

## Summary

To support **all three models** (Qwen2.5-VL + Aria + GLM-4V) simultaneously:

1. **Upgrade transformers**: `pip install transformers>=4.56.0`
2. **That's it!** The code now auto-detects and supports all models

The monkey patch has been updated to be version-compatible, so Flash Attention will work when available without breaking on newer transformers versions.
