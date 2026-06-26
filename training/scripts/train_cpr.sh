#!/bin/bash
# CRYSTAL — Causal Process Reward (CPR) GRPO training.
# Default model: Qwen2.5-VL. Model-agnostic: override MODEL for any VLM the
# base trainer supports (see train_cpr_internvl.sh for InternVL specifics).
#
# Requires the open-r1-multimodal GRPO trainer (set OPENR1_SRC) and
# `pip install crystal-metrics`. See ../README.md.
set -e

# --- paths (override via env) ---------------------------------------------
OPENR1_SRC="${OPENR1_SRC:?Set OPENR1_SRC to <open-r1-multimodal>/src}"
TRAINING_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${TRAINING_DIR}/configs/zero3.json}"

# crystal_metrics rewards + the model-agnostic adapter must be importable.
export PYTHONPATH="${OPENR1_SRC}:${TRAINING_DIR}:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- run config (override via env) ----------------------------------------
MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
DATASET="${DATASET:-waybarrios/CRYSTAL}"          # HF name or local load_from_disk path
OUTPUT="${OUTPUT:-./output/cpr-$(date +%Y%m%d_%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_GPUS="${NUM_GPUS:-4}"

PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-5}"
GRADIENT_ACCUM="${GRADIENT_ACCUM:-2}"
NUM_GENERATIONS="${NUM_GENERATIONS:-5}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
SAVE_STEPS="${SAVE_STEPS:-100}"
LOGGING_STEPS="${LOGGING_STEPS:-2}"
SEED="${SEED:-42}"

# Qwen vision token budget (ignored by non-Qwen models).
MAX_PIXELS="${MAX_PIXELS:-602112}"
MIN_PIXELS="${MIN_PIXELS:-3136}"

# CPR weights (paper defaults).
CAUSAL_ANSWER_WEIGHT="${CAUSAL_ANSWER_WEIGHT:-0.6}"
CAUSAL_STEP_WEIGHT="${CAUSAL_STEP_WEIGHT:-0.4}"

mkdir -p "$OUTPUT"
export DEBUG_MODE="${DEBUG_MODE:-false}"
export LOG_PATH="${OUTPUT}/reward.txt"

echo "CRYSTAL CPR training | model=$MODEL | aw=$CAUSAL_ANSWER_WEIGHT sw=$CAUSAL_STEP_WEIGHT | out=$OUTPUT"

accelerate launch \
    --num_processes "$NUM_GPUS" --num_machines 1 --mixed_precision bf16 \
    --use_deepspeed --deepspeed_config_file "$DEEPSPEED_CONFIG" \
    --zero3_init_flag true --zero3_save_16bit_model true \
    --gradient_accumulation_steps "$GRADIENT_ACCUM" \
    "${OPENR1_SRC}/open_r1/grpo_rec.py" \
    --model_name_or_path "$MODEL" \
    --dataset_name "$DATASET" \
    --use_huggingface_dataset \
    --task_type "vqa" \
    --reward_funcs "format" "accuracy" "reasoning" \
    --reward_weights 2.0 2.0 2.0 \
    --use_causal_reasoning_reward true \
    --causal_answer_weight "$CAUSAL_ANSWER_WEIGHT" \
    --causal_step_weight "$CAUSAL_STEP_WEIGHT" \
    --use_pcgrad true \
    --output_dir "$OUTPUT" \
    --seed "$SEED" --shuffle_train_dataset \
    --num_train_epochs "$NUM_EPOCHS" \
    --per_device_train_batch_size "$PER_DEVICE_BATCH" \
    --gradient_accumulation_steps "$GRADIENT_ACCUM" \
    --learning_rate "$LEARNING_RATE" \
    --num_generations "$NUM_GENERATIONS" \
    --gradient_checkpointing \
    --logging_steps "$LOGGING_STEPS" --save_steps "$SAVE_STEPS" \
    --max_pixels "$MAX_PIXELS" --min_pixels "$MIN_PIXELS" \
    --bf16 --deepspeed "$DEEPSPEED_CONFIG" \
    2>&1 | tee "$OUTPUT/training.log"
