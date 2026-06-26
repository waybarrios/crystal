#!/bin/bash
# CRYSTAL — Semantic Process Reward (SPR) GRPO training.  [EXPERIMENTAL]
#
# NOT the method reported in the paper — the CRYSTAL results use CPR /
# CPR-Curriculum (see ../README.md). SPR is an alternative reasoning reward that
# matches steps by SentenceTransformer cosine similarity instead of the causal
# interaction. Kept for reference. Set MODEL for any VLM.
set -e

OPENR1_SRC="${OPENR1_SRC:?Set OPENR1_SRC to <open-r1-multimodal>/src}"
TRAINING_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${TRAINING_DIR}/configs/zero3.json}"

export PYTHONPATH="${OPENR1_SRC}:${TRAINING_DIR}:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
DATASET="${DATASET:-waybarrios/CRYSTAL}"
OUTPUT="${OUTPUT:-./output/spr-$(date +%Y%m%d_%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_GPUS="${NUM_GPUS:-4}"

PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-5}"
GRADIENT_ACCUM="${GRADIENT_ACCUM:-2}"
NUM_GENERATIONS="${NUM_GENERATIONS:-5}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SEED="${SEED:-42}"
MAX_PIXELS="${MAX_PIXELS:-602112}"
MIN_PIXELS="${MIN_PIXELS:-3136}"
SEMANTIC_THRESHOLD="${SEMANTIC_THRESHOLD:-0.70}"

mkdir -p "$OUTPUT"
export DEBUG_MODE="${DEBUG_MODE:-false}"
export LOG_PATH="${OUTPUT}/reward.txt"

echo "CRYSTAL SPR training | model=$MODEL | threshold=$SEMANTIC_THRESHOLD | out=$OUTPUT"

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
    --reward_weights 3.0 1.0 3.0 \
    --use_semantic_reasoning_reward true \
    --semantic_similarity_threshold "$SEMANTIC_THRESHOLD" \
    --output_dir "$OUTPUT" \
    --seed "$SEED" --shuffle_train_dataset \
    --num_train_epochs "$NUM_EPOCHS" \
    --per_device_train_batch_size "$PER_DEVICE_BATCH" \
    --gradient_accumulation_steps "$GRADIENT_ACCUM" \
    --learning_rate "$LEARNING_RATE" \
    --num_generations "$NUM_GENERATIONS" \
    --gradient_checkpointing \
    --save_steps "$SAVE_STEPS" \
    --max_pixels "$MAX_PIXELS" --min_pixels "$MIN_PIXELS" \
    --bf16 --deepspeed "$DEEPSPEED_CONFIG" \
    2>&1 | tee "$OUTPUT/training.log"
