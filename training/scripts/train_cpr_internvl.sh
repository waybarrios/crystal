#!/bin/bash
# CRYSTAL — CPR GRPO training for InternVL.
# InternVL differs from Qwen in two ways: image patches are controlled by
# --max_anyres_num (not max/min pixels), and it uses the InternVL DeepSpeed
# config. Otherwise identical to train_cpr.sh.
set -e

# InternVL usually needs its own env (transformers/flash-attn build).
# eval "$(conda shell.bash hook 2>/dev/null)" && conda activate internvl

OPENR1_SRC="${OPENR1_SRC:?Set OPENR1_SRC to <open-r1-multimodal>/src}"
TRAINING_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${TRAINING_DIR}/configs/zero3_internvl.json}"

export PYTHONPATH="${OPENR1_SRC}:${TRAINING_DIR}:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="${MODEL:-OpenGVLab/InternVL3_5-4B}"
DATASET="${DATASET:-waybarrios/CRYSTAL}"
OUTPUT="${OUTPUT:-./output/internvl-cpr-$(date +%Y%m%d_%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_GPUS="${NUM_GPUS:-4}"

PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-5}"
GRADIENT_ACCUM="${GRADIENT_ACCUM:-2}"
NUM_GENERATIONS="${NUM_GENERATIONS:-5}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
SAVE_STEPS="${SAVE_STEPS:-100}"
LOGGING_STEPS="${LOGGING_STEPS:-2}"
SEED="${SEED:-42}"

# InternVL-specific: number of dynamic high-res image patches.
MAX_ANYRES_NUM="${MAX_ANYRES_NUM:-12}"

# CPR weights (curriculum Phase-2 defaults for InternVL).
CAUSAL_ANSWER_WEIGHT="${CAUSAL_ANSWER_WEIGHT:-0.65}"
CAUSAL_STEP_WEIGHT="${CAUSAL_STEP_WEIGHT:-0.35}"

mkdir -p "$OUTPUT"
export DEBUG_MODE="${DEBUG_MODE:-false}"
export LOG_PATH="${OUTPUT}/reward.txt"

echo "CRYSTAL CPR (InternVL) | model=$MODEL | anyres=$MAX_ANYRES_NUM | out=$OUTPUT"

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
    --max_anyres_num "$MAX_ANYRES_NUM" \
    --bf16 --deepspeed "$DEEPSPEED_CONFIG" \
    2>&1 | tee "$OUTPUT/training.log"
