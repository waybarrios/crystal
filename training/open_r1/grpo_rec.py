# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import pickle
import hashlib

from PIL import Image
from torch.utils.data import Dataset
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from open_r1.trainer import VLMGRPOTrainer, GRPOConfig
from open_r1.vlm_modules import *
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from transformers import TrainingArguments
import yaml
import json
import random
import math
import numpy as np
import torch
from datasets import load_from_disk

# PyTorch 2.6 compatibility fix for DeepSpeed checkpoint loading
# PyTorch 2.6 changed torch.load default to weights_only=True, but DeepSpeed
# checkpoints contain non-tensor objects that require weights_only=False
import functools
_original_torch_load = torch.load
@functools.wraps(_original_torch_load)
def _torch_load_with_compat(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _torch_load_with_compat

from open_r1.qwen2_5vl_monkey_patch import monkey_patch_qwen2_5vl_flash_attn, monkey_patch_qwen2_5vl_forward
from open_r1.internvl_monkey_patch import monkey_patch_internvl_forward, monkey_patch_torch_load
# Only apply Qwen monkey patch at module level if not running InternVL
# The forward patch is applied conditionally in __main__ based on model name
monkey_patch_qwen2_5vl_flash_attn()


# ----------------------- Main Script -----------------------
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image (for QwenVL)"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image (for QwenVL)"},
    )
    max_anyres_num: Optional[int] = field(
        default=12,
        metadata={"help": "Maximum number of anyres blocks for the image (for InternVL)"},
    )
    image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root directory of the image"},
    )
    use_huggingface_dataset: bool = field(
        default=False,
        metadata={"help": "Whether to use HuggingFace dataset format instead of JSON/JSONL"},
    )
    task_type: str = field(
        default="rec",
        metadata={"help": "Task type for the VLM module. Possible values: 'rec', 'vqa', 'ic', 'odLength'"},
    )
    use_llm_judge: bool = field(
        default=False,
        metadata={"help": "Whether to use LLM-based judge for accuracy evaluation (requires Ollama)"},
    )
    llm_judge_model: str = field(
        default="gpt-oss:20b",
        metadata={"help": "Ollama model name for LLM judge (e.g., 'gpt-oss:20b', 'llama3.2', 'mistral', 'phi3')"},
    )
    llm_judge_base_url: str = field(
        default="http://localhost:11434/v1",
        metadata={"help": "Base URL for Ollama API (OpenAI-compatible endpoint)"},
    )
    shuffle_train_dataset: bool = field(
        default=True,
        metadata={"help": "Whether to shuffle the training dataset"},
    )
    # Semantic Process Reward (SPR) settings
    use_semantic_reasoning_reward: bool = field(
        default=False,
        metadata={"help": "Use semantic similarity (SentenceTransformer) instead of word overlap for reasoning reward"},
    )
    semantic_similarity_threshold: float = field(
        default=0.70,
        metadata={"help": "Cosine similarity threshold for semantic matching (default: 0.70)"},
    )
    # Causal Process Reward (CPR) settings
    use_causal_reasoning_reward: bool = field(
        default=False,
        metadata={"help": "Use Causal Process Reward instead of word overlap for reasoning"},
    )
    causal_answer_weight: float = field(
        default=0.6,
        metadata={"help": "Weight for answer correctness in CPR (default: 0.6)"},
    )
    causal_step_weight: float = field(
        default=0.4,
        metadata={"help": "Weight for step alignment in CPR (default: 0.4)"},
    )
    # PCGrad settings for gradient conflict resolution
    use_pcgrad: bool = field(
        default=False,
        metadata={"help": "Use PCGrad to resolve gradient conflicts between accuracy and reasoning"},
    )

@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, script_args: GRPOScriptArguments, question_template: str, seed: Optional[int] = None):
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args
        self.seed = seed
        self.list_data_dict = []
        self.question_template = question_template
        self.use_huggingface = script_args.use_huggingface_dataset

        # Set random seed for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            print(f"Random seed set to: {seed}")

        # Create cache filename based on data_path and seed
        cache_key = f"{data_path}_{seed}_{script_args.shuffle_train_dataset}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        cache_dir = os.path.join(os.path.dirname(data_path), ".dataset_cache")
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_dir, f"dataset_cache_{cache_hash}.pkl")

        # Try to load from cache
        if os.path.exists(self.cache_file):
            print(f"📦 Loading dataset from cache: {self.cache_file}")
            try:
                with open(self.cache_file, 'rb') as f:
                    self.list_data_dict = pickle.load(f)
                print(f"✓ Loaded {len(self.list_data_dict)} samples from cache!")
                return  # Skip loading and processing
            except Exception as e:
                print(f"⚠ Cache loading failed: {e}. Loading from source...")

        if self.use_huggingface:
            # Load HuggingFace dataset from disk
            print(f"Loading HuggingFace dataset from: {data_path}")
            print("This may take a while if dataset is large or on network storage...")
            dataset = load_from_disk(data_path)
            print(f"✓ Dataset loaded! Total rows: {len(dataset)}")

            # Convert HuggingFace dataset to list of dicts
            print("Converting dataset to internal format...")
            dataset_len = len(dataset)
            print_every = max(1, dataset_len // 10)  # Print progress every 10%

            for idx, item in enumerate(dataset):
                self.list_data_dict.append({
                    'image': item['image'],  # PIL Image
                    'question': item['question'],
                    'answer': item['answer'],
                    'reference_steps': item.get('reference_steps', []),
                    'choices': item.get('choices', []),
                    'options': item.get('options', []),
                    'source': item.get('source', ''),
                })
                if (idx + 1) % print_every == 0:
                    progress = (idx + 1) / dataset_len * 100
                    print(f"  Progress: {idx + 1}/{dataset_len} ({progress:.1f}%)")

            print(f"✓ Loaded {len(self.list_data_dict)} samples from HuggingFace dataset at {data_path}")
        elif data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                # file should be in the format of:
                # datasets:
                #   - json_path: xxxx1.json
                #     sampling_strategy: first:1000
                #   - json_path: xxxx2.json
                #     sampling_strategy: end:3000
                #   - json_path: xxxx3.json
                #     sampling_strategy: random:999

                for data in datasets:
                    json_path = data.get("json_path")
                    sampling_strategy = data.get("sampling_strategy", "all")
                    sampling_number = None

                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path, "r") as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        with open(json_path, "r") as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)

                    # Apply the sampling strategy
                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]
                    print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
        else:
            raise ValueError(f"Unsupported file type: {data_path}")

        # Shuffle the entire dataset if requested
        if script_args.shuffle_train_dataset:
            random.shuffle(self.list_data_dict)
            print(f"Dataset shuffled with seed {self.seed}")

        print(f"Total samples in dataset: {len(self.list_data_dict)}")

        # Save to cache for future runs
        try:
            print(f"💾 Saving dataset to cache: {self.cache_file}")
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.list_data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"✓ Cache saved! Future runs will load instantly.")
        except Exception as e:
            print(f"⚠ Failed to save cache (non-critical): {e}")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        example = self.list_data_dict[i]

        if self.use_huggingface:
            # HuggingFace dataset format
            # Format the user question with choices/options
            if example.get("choices"):
                formatted_options = "\n".join([f"{chr(65+i)}) {c}" for i, c in enumerate(example["choices"])])
            elif example.get("options"):
                formatted_options = "\n".join([f"{chr(65+i)}) {c}" for i, c in enumerate(example["options"])])
            else:
                formatted_options = ""

            # Format user question with options/choices if they exist
            user_question = f"{example['question']}\n\n{formatted_options}".strip()

            # Image is already PIL Image from HuggingFace dataset
            image = example['image']

            # System prompt with comprehensive JSON format instructions
            system_prompt = """You are a vision-language model. First, analyze the provided image(s) and any user text silently. Do NOT reveal your internal reasoning.

Return ONLY a single, valid JSON object with this exact schema:
{"reasoning_steps": [], "answer": ""}

Rules for "reasoning_steps":
- Decide the number of steps based on task complexity; include enough to make the answer evident without filler.
- Include some inference from visual information, always anchored to visible cues.
- Write single-clause sentences, each adding a new, directly checkable fact or cue-based inference.
- You may include cautious, visually grounded commonsense using words such as "appears", "suggests", or "likely", but always anchor it to visible cues (lighting/shadows; perspective/vanishing lines/horizon/tilt; scale/relative size; focus/DOF; parallax; occlusion/contact shadows; reflections/transparency; material/texture; symmetry/patterns/alignment; position/orientation/foreground–background; density/motion cues; human pose/gaze/gesture; interactions/affordances; object state; physics plausibility; signage/text/logos/typography; numbers/units; plots/charts: type, axes/ticks/units, scale (lin/log), legend↔series, gridlines/baseline, error bars/CI, trendlines, outliers/binning, colorbar; maps: scale bar, north arrow; math/geometry: labels/givens, unit checks, angle rules, Pythagorean, distance/slope, transformations, area/volume, circle theorems, trig (incl. sine/cosine laws), vectors, systems/quadratics, combinatorics, logs/exponents, probability/statistics, exact forms, conversions, plots, graphs, math equations, diagrams).
- Keep each step ≤14 words. No multi-sentence items. No chains like "because/therefore". No internal monologue.

Rules for "answer":
- Provide the final answer grounded strictly in visible content and given text.
- If information is missing or ambiguous, set "answer" to "insufficient information" and include steps noting what is missing (e.g., "Noted the license plate is unreadable due to blur.").
- Multiple-choice: if options have letters, return only the single best LETTER (e.g., "B"); if unlabeled, return the exact option text verbatim.
- Numeric: include required units; obey requested rounding; otherwise give exact/simplest form.

What to notice in steps (express as sentences, not labels):
- Objects & attributes (classes, colors, materials, states), logos/brands if clearly visible.
- Positions & spatial relations (left/right/above/below/front/behind, proximity, alignment, orientation, foreground/background).
- Depth cues (relative size, position in frame vs. horizon, sharpness/detail, shadow contact, occlusion order).
- Scene & lighting/time cues (indoor/outdoor, daylight vs. night, weather indications, activity/no-activity).
- Occlusion effects and how they affect certainty.
- Text/OCR with exact casing/punctuation ("Read text: 'SPEED LIMIT 25'. ").
- Counts & quantities for distinct instances; approximate only if visually justified.
- Graphics/plots/diagrams: axes, ticks, units, legends; read exact values rather than guessing.

Output formatting:
- Output only the JSON object. No extra keys, comments, code fences, or prose.
- Use double quotes for all strings; no trailing commas; any valid JSON whitespace is acceptable."""

            # Create prompt with system message (instructions) and user message (question only)
            prompt = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": system_prompt},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_question},
                    ],
                },
            ]

            return {
                'image': image,
                'problem': user_question,
                'solution': example['answer'],
                'reference_steps': example.get('reference_steps', []),
                'prompt': prompt,
                'image_file': f"{example.get('source', 'unknown')}_{i}",
                'data_index': i,
            }
        else:
            # Original JSON/JSONL format
            def make_conversation(example):
                return {
                    "prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": example["problem"]},
                    ],
                }
            QUESTION_TEMPLATE = self.question_template
            def make_conversation_image(example):
                return {
                    "prompt": [
                        # {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                            ],
                        },
                    ],
                }

            image_root = self.script_args.image_root
            if 'image' in example:
                image_path = os.path.join(image_root, example['image'])
                # In case the image is not found
                while not os.path.exists(image_path):
                    print(f"Warning: Image {image_path} not found, randomly selecting another image")
                    new_index = random.randint(0, len(self.list_data_dict)-1)
                    example = self.list_data_dict[new_index]
                    image_path = os.path.join(image_root, example['image'])
                image = Image.open(image_path).convert("RGB")
            else:
                image = None


            return {
                'image': image,
                'problem': example['problem'],
                'solution': example['solution'],
                'prompt': make_conversation_image(example)['prompt'] if 'image' in example else make_conversation(example)['prompt'],
                'image_file': example.get('image', 'no_image'),
                'data_index': i,
            }


def get_vlm_module(model_name_or_path):
    name_lower = model_name_or_path.lower()
    if "qwen" in name_lower:
        return Qwen2VLModule
    elif "internvl" in name_lower:
        return InvernVLModule
    elif "glm" in name_lower:
        return GLMVModule
    else:
        # For checkpoint paths, check config.json for model_type
        import json
        config_path = os.path.join(model_name_or_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            model_type = config.get("model_type", "").lower()
            if "qwen" in model_type:
                return Qwen2VLModule
            elif "internvl" in model_type:
                return InvernVLModule
            elif "glm" in model_type:
                return GLMVModule
        raise ValueError(f"Unsupported model: {model_name_or_path}")

def main(script_args, training_args, model_args):
    # Load the VLM module
    vlm_module_cls = get_vlm_module(model_args.model_name_or_path)

    # Configure LLM judge if using VQA task
    if script_args.task_type == "vqa":
        vlm_module_cls.configure_llm_judge(
            use_llm_judge=script_args.use_llm_judge,
            llm_judge_model=script_args.llm_judge_model,
            llm_judge_base_url=script_args.llm_judge_base_url
        )
        print(f"LLM Judge configured: use_llm_judge={script_args.use_llm_judge}, model={script_args.llm_judge_model}")

    # Configure Semantic Process Reward (SPR) if enabled
    if script_args.use_semantic_reasoning_reward:
        vlm_module_cls.configure_semantic_reward(threshold=script_args.semantic_similarity_threshold)
        # Replace 'reasoning' with 'reasoning_semantic' in reward_funcs
        script_args.reward_funcs = [
            "reasoning_semantic" if f == "reasoning" else f
            for f in script_args.reward_funcs
        ]
        print(f"Semantic Process Reward ENABLED (threshold={script_args.semantic_similarity_threshold})")

    # Configure Causal Process Reward (CPR) if enabled
    if script_args.use_causal_reasoning_reward:
        vlm_module_cls.configure_causal_reward(
            answer_weight=script_args.causal_answer_weight,
            step_weight=script_args.causal_step_weight
        )
        # Replace 'reasoning' with 'reasoning_causal' in reward_funcs
        script_args.reward_funcs = [
            "reasoning_causal" if f == "reasoning" else f
            for f in script_args.reward_funcs
        ]
        print(f"Causal Process Reward ENABLED (answer_weight={script_args.causal_answer_weight}, step_weight={script_args.causal_step_weight})")

    # Configure PCGrad for gradient conflict resolution
    if script_args.use_pcgrad:
        print("PCGrad ENABLED for gradient conflict resolution between accuracy and reasoning")

    # Load the reward functions based on task type
    reward_funcs = [vlm_module_cls.select_reward_func(func, script_args.task_type) for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)
    print("task_type:", script_args.task_type)

    # Load the dataset
    dataset = LazySupervisedDataset(
        script_args.dataset_name,
        script_args,
        question_template=vlm_module_cls.get_question_template(task_type=script_args.task_type),
        seed=training_args.seed
    )

    trainer_cls = VLMGRPOTrainer
    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        vlm_module=vlm_module_cls(),
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        max_anyres_num=script_args.max_anyres_num,
        torch_dtype=model_args.torch_dtype,
    )

    # Train and push the model to the Hub
    # Check for checkpoint to resume from
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        print(f"Resuming training from checkpoint: {checkpoint}")

    trainer.train(resume_from_checkpoint=checkpoint)

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    # Only apply Qwen-specific monkey patches for Qwen models
    is_qwen = "qwen" in model_args.model_name_or_path.lower()
    if not is_qwen:
        # Check config.json for checkpoint paths
        import json as _json
        _config_path = os.path.join(model_args.model_name_or_path, "config.json")
        if os.path.exists(_config_path):
            with open(_config_path) as _f:
                is_qwen = "qwen" in _json.load(_f).get("model_type", "").lower()
    if is_qwen and training_args.deepspeed and "zero3" in training_args.deepspeed:
        print("zero3 is used, qwen2_5vl forward monkey patch is applied")
        monkey_patch_qwen2_5vl_forward()

    # Apply InternVL-specific monkey patches for ZeRO-3
    is_internvl = "internvl" in model_args.model_name_or_path.lower()
    if not is_internvl:
        _config_path = os.path.join(model_args.model_name_or_path, "config.json")
        if os.path.exists(_config_path):
            import json as _json2
            with open(_config_path) as _f2:
                _cfg = _json2.load(_f2)
                is_internvl = (
                    "internvl" in _cfg.get("architectures", [""])[0].lower()
                    or "internvl" in _cfg.get("model_type", "").lower()
                )
    if is_internvl and training_args.deepspeed and "zero3" in training_args.deepspeed:
        print("zero3 is used, InternVL forward monkey patch is applied")
        monkey_patch_internvl_forward(model_args.model_name_or_path)
        monkey_patch_torch_load()

    main(script_args, training_args, model_args)
