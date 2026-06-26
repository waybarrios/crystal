from open_r1.vlm_modules.vlm_module import VLMBaseModule
from typing import Dict, Any, Union
from transformers import AutoModel, AutoProcessor, AutoConfig
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers.feature_extraction_sequence_utils import BatchFeature

# Import from mllm_evaluator (added to PYTHONPATH by training script)
from accuracy_calculator import AccuracyCalculator

IMG_START_TOKEN='<img>'
IMG_END_TOKEN='</img>'
IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class InvernVLModule(VLMBaseModule):
    # Class variables for LLM judge configuration
    _use_llm_judge = False
    _llm_judge_model = "gpt-oss:20b"
    _llm_judge_base_url = "http://localhost:11434/v1"

    # Causal Process Reward (CPR) configuration
    _causal_answer_weight = 0.6
    _causal_step_weight = 0.4

    # ZeRO-3 monkey patch flag (set by internvl_monkey_patch.monkey_patch_internvl_forward)
    _zero3_patched = False

    def __init__(self):
        super().__init__()
        self.conv_template = None
        self.num_image_token = None

    @classmethod
    def configure_llm_judge(cls, use_llm_judge=False, llm_judge_model="gpt-oss:20b", llm_judge_base_url="http://localhost:11434/v1"):
        """Configure LLM judge settings for accuracy evaluation."""
        cls._use_llm_judge = use_llm_judge
        cls._llm_judge_model = llm_judge_model
        cls._llm_judge_base_url = llm_judge_base_url

    @classmethod
    def configure_causal_reward(cls, answer_weight: float = 0.6, step_weight: float = 0.4):
        """Configure Causal Process Reward weights."""
        cls._causal_answer_weight = answer_weight
        cls._causal_step_weight = step_weight
        print(f"Causal Process Reward configured: answer_weight={answer_weight}, step_weight={step_weight}")

    def get_vlm_key(self):
        return "internvl"

    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        self.model_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        model_cls = AutoModel
        model_init_kwargs["trust_remote_code"] = True
        model_init_kwargs.pop("use_cache", None)
        if "flash_attention_2" in (model_init_kwargs.get("attn_implementation") or ""):
            model_init_kwargs["use_flash_attn"] = True
            model_init_kwargs.pop("attn_implementation")
        return model_cls

    def post_model_init(self, model, processing_class):
        self.conv_template = model.conv_template if self.conv_template is None else self.conv_template
        self.num_image_token = model.num_image_token if self.num_image_token is None else self.num_image_token
        img_context_token_id = processing_class.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        model.img_context_token_id = img_context_token_id

    def is_embeds_input(self):
        # When ZeRO-3 monkey patch is active, we use standard HF generate()
        # which returns full prompt+completion ids (not just completion).
        return not self._zero3_patched

    def get_processing_class(self):
        return AutoProcessor

    def get_eos_token_id(self, processing_class):
        eos_token_id = processing_class.convert_tokens_to_ids(self.conv_template.sep.strip())
        return eos_token_id

    def get_vision_modules_keywords(self):
        return ['vision_model']

    def get_custom_multimodal_keywords(self):
        return ['pixel_values', 'image_flags']

    def get_non_generate_params(self):
        return ['image_flags']

    def get_custom_processing_keywords(self):
        return [('None', 'max_anyres_num')]

    def prepare_prompt(self, processing_class, inputs: dict[str, Union[torch.Tensor, Any]]):
        prompts_text = []
        for example in inputs:
            template = self.conv_template.copy()
            conversation_list = example["prompt"]
            system_message = extract_system_message(conversation_list)
            if system_message is not None:
                template.system_message = system_message

            processed_list = process_conversation_list(conversation_list, system_message)
            for i, processed_item in enumerate(processed_list):
                if i % 2 == 0:
                    template.append_message(template.roles[0], processed_item)
                else:
                    template.append_message(template.roles[1], processed_item)
            if len(processed_list) % 2 == 1:
                template.append_message(template.roles[1], None)
            query = template.get_prompt()
            prompts_text.append(query)
        return prompts_text

    def prepare_model_inputs(self, processing_class, prompts_text, images, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False):
        full_pixel_values = []
        num_patches_list = []
        for img in images:
            pixel_values = self._load_image(img, input_size=self.model_config.vision_config.image_size, max_num=processing_class.max_anyres_num)
            full_pixel_values.append(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
        full_pixel_values = torch.cat(full_pixel_values, dim=0)

        queries = []
        image_idx = 0
        for query in prompts_text:
            while "<image>" in query:
                num_patches = num_patches_list[image_idx]
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
                query = query.replace("<image>", image_tokens, 1)
                image_idx += 1
            queries.append(query)
        assert image_idx == len(num_patches_list)

        model_inputs = processing_class(
            queries,
            return_tensors=return_tensors,
            padding=padding,
            padding_side=padding_side,
            add_special_tokens=add_special_tokens,
        )
        model_inputs["pixel_values"] = full_pixel_values
        model_inputs['image_flags'] = torch.ones(full_pixel_values.shape[0], dtype=torch.long)

        model_inputs = BatchFeature(data=model_inputs)

        return model_inputs, None

    def _load_image(self, image: Image.Image, input_size: int=448, max_num:int=12):
        transform = build_transform(input_size=input_size)
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    @staticmethod
    def get_question_template(task_type: str):
        match task_type:
            case "vqa":
                return "{USER_INSTRUCTION}"
            case _:
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."

    # ── VQA Reward Functions ──

    @staticmethod
    def _parse_and_validate_json(content: str):
        """Parse and validate JSON from model output.

        Returns:
            tuple: (parsed_dict, is_valid, error_message)
        """
        import json
        import re

        content_cleaned = re.sub(r'```json\s*|\s*```', '', content).strip()

        json_str = None
        parsed = None

        try:
            parsed = json.loads(content_cleaned)
            json_str = content_cleaned
        except (json.JSONDecodeError, Exception):
            pass

        if parsed is None:
            first_brace = content_cleaned.find('{')
            if first_brace != -1:
                brace_count = 0
                for idx in range(first_brace, len(content_cleaned)):
                    if content_cleaned[idx] == '{':
                        brace_count += 1
                    elif content_cleaned[idx] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_str = content_cleaned[first_brace:idx+1]
                            break

        if json_str:
            json_str = json_str.replace('\u201c', '"').replace('\u201d', '"').replace('\u2018', "'").replace('\u2019', "'")
            json_str = re.sub(r'''(?<=[:,\[])\s*'([^']*)'(?=\s*[,\]\}])''', r' "\1"', json_str)
            json_str = re.sub(r'"\s*\n\s*"', '",\n    "', json_str)
            json_str = re.sub(r'"\s+(?=")', '", ', json_str)
            json_str = re.sub(r',(\s*[\]}])', r'\1', json_str)

            if parsed is None:
                try:
                    parsed = json.loads(json_str)
                except json.JSONDecodeError as e:
                    return None, False, f"JSON parse error: {str(e)}"

        if parsed is None:
            return None, False, "No valid JSON found"

        if not isinstance(parsed, dict):
            return None, False, "JSON is not an object"

        if "reasoning_steps" not in parsed:
            return None, False, "Missing 'reasoning_steps' key"

        if "answer" not in parsed:
            return None, False, "Missing 'answer' key"

        if not isinstance(parsed["reasoning_steps"], list):
            return None, False, "'reasoning_steps' must be a list"

        if not isinstance(parsed["answer"], str):
            return None, False, "'answer' must be a string"

        for idx, step in enumerate(parsed["reasoning_steps"]):
            if not isinstance(step, str):
                return None, False, f"'reasoning_steps[{idx}]' must be a string, not {type(step).__name__}"

        reasoning_steps = [s.strip() for s in parsed["reasoning_steps"] if s and s.strip()]

        if not reasoning_steps:
            return None, False, "'reasoning_steps' is empty or contains no valid strings"

        return {
            "reasoning_steps": reasoning_steps,
            "answer": parsed["answer"].strip()
        }, True, None

    @staticmethod
    def format_reward_vqa(completions, **kwargs):
        """Check if the model output is valid JSON with reasoning_steps and answer fields."""
        import json
        import re
        import os
        from datetime import datetime

        completion_contents = [completion[0]["content"] for completion in completions]
        problems = kwargs.get("problem", [])
        solutions = kwargs.get("solution", [])
        image_files = kwargs.get("image_file", [])
        data_indices = kwargs.get("data_index", [])
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

        for i, content in enumerate(completion_contents):
            reward = 0.0
            parsed = None

            try:
                content_cleaned = re.sub(r'```json\s*|\s*```', '', content).strip()

                try:
                    parsed = json.loads(content_cleaned)
                except json.JSONDecodeError:
                    first_brace = content_cleaned.find('{')
                    if first_brace != -1:
                        brace_count = 0
                        for idx in range(first_brace, len(content_cleaned)):
                            if content_cleaned[idx] == '{':
                                brace_count += 1
                            elif content_cleaned[idx] == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    json_str = content_cleaned[first_brace:idx+1]
                                    json_str = re.sub(r',(\s*[\]}])', r'\1', json_str)
                                    parsed = json.loads(json_str)
                                    break

                if parsed is not None:
                    if "reasoning_steps" in parsed and "answer" in parsed:
                        if isinstance(parsed["reasoning_steps"], list) and isinstance(parsed["answer"], str):
                            if len(parsed["reasoning_steps"]) > 0:
                                all_strings = all(isinstance(step, str) for step in parsed["reasoning_steps"])
                                if all_strings:
                                    reward = 1.0

            except (json.JSONDecodeError, Exception):
                pass

            rewards.append(reward)

            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                with open(log_path.replace(".txt", "_format_vqa.txt"), "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Format VQA reward: {int(reward)} -------------\n")
                    f.write(f"Data Index: {data_indices[i] if i < len(data_indices) else 'N/A'}\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Has valid format: {reward == 1.0}\n\n")

        return rewards

    @staticmethod
    def vqa_accuracy_reward(completions, solution, **kwargs):
        """Calculate accuracy reward using accuracy_calculator for VQA task."""
        import json
        import re
        import os
        from datetime import datetime

        use_llm_grader = kwargs.get("use_llm_judge", InvernVLModule._use_llm_judge)
        llm_model = kwargs.get("llm_judge_model", InvernVLModule._llm_judge_model)
        base_url = kwargs.get("llm_judge_base_url", InvernVLModule._llm_judge_base_url)

        accuracy_calc = AccuracyCalculator(
            use_llm_grader=use_llm_grader,
            llm_model=llm_model,
            base_url=base_url
        )

        completion_contents = [completion[0]["content"] for completion in completions]
        problems = kwargs.get("problem", [])
        image_files = kwargs.get("image_file", [])
        data_indices = kwargs.get("data_index", [])
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

        for i, (content, sol, problem) in enumerate(zip(completion_contents, solution, problems)):
            reward = 0.0
            predicted_answer = ""
            try:
                content_cleaned = re.sub(r'```json\s*|\s*```', '', content).strip()

                parsed = None
                try:
                    parsed = json.loads(content_cleaned)
                except (json.JSONDecodeError, Exception):
                    pass

                if parsed is None:
                    first_brace = content_cleaned.find('{')
                    if first_brace != -1:
                        brace_count = 0
                        for idx in range(first_brace, len(content_cleaned)):
                            if content_cleaned[idx] == '{':
                                brace_count += 1
                            elif content_cleaned[idx] == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    json_str = content_cleaned[first_brace:idx+1]
                                    json_str = json_str.replace('\u201c', '"').replace('\u201d', '"')
                                    json_str = re.sub(r',(\s*[\]}])', r'\1', json_str)
                                    try:
                                        parsed = json.loads(json_str)
                                    except json.JSONDecodeError:
                                        json_str_fixed = json_str.replace("'", '"')
                                        json_str_fixed = re.sub(r',(\s*[\]}])', r'\1', json_str_fixed)
                                        parsed = json.loads(json_str_fixed)
                                    break

                predicted_answer = parsed.get("answer", "") if parsed else ""

                if predicted_answer:
                    result = accuracy_calc.evaluate_single(problem, predicted_answer, sol)
                    if result.is_correct:
                        reward = result.confidence

            except (json.JSONDecodeError, Exception) as e:
                if os.getenv("DEBUG_MODE") == "true":
                    log_path = os.getenv("LOG_PATH")
                    with open(log_path.replace(".txt", "_accuracy_errors.txt"), "a", encoding='utf-8') as f:
                        f.write(f"------------- {current_time} JSON Parse Error -------------\n")
                        f.write(f"Data Index: {data_indices[i] if i < len(data_indices) else 'N/A'}\n")
                        f.write(f"Error: {str(e)}\nContent: {content}\n\n")

            rewards.append(reward)

            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                with open(log_path.replace(".txt", "_accuracy_vqa.txt"), "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Accuracy VQA reward: {reward} -------------\n")
                    f.write(f"Data Index: {data_indices[i] if i < len(data_indices) else 'N/A'}\n")
                    f.write(f"Problem: {problem}\nPredicted: {predicted_answer}\nGround Truth: {sol}\n\n")

        return rewards

    @staticmethod
    def vqa_reasoning_reward(completions, **kwargs):
        """Calculate reasoning quality reward using word overlap F1 (DeepSpeed-safe)."""
        import os
        from datetime import datetime

        completion_contents = [completion[0]["content"] for completion in completions]
        reference_steps_list = kwargs.get("reference_steps", [])
        data_indices = kwargs.get("data_index", [])
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

        for i, (content, ref_steps) in enumerate(zip(completion_contents, reference_steps_list)):
            reward = 0.0
            predicted_steps = []
            metrics_info = ""

            parsed_data, is_valid, error_msg = InvernVLModule._parse_and_validate_json(content)

            if not is_valid:
                metrics_info = f"Validation failed: {error_msg}"
                rewards.append(reward)
                continue

            predicted_steps = parsed_data["reasoning_steps"]

            if ref_steps:
                ref_steps_cleaned = [str(s).strip() for s in ref_steps if s and str(s).strip()]

                if not ref_steps_cleaned:
                    metrics_info = "No valid reference steps"
                    rewards.append(reward)
                    continue

                from simple_similarity import best_match_f1
                try:
                    f1, matched_pred, matched_ref = best_match_f1(predicted_steps, ref_steps_cleaned, threshold=0.45)
                    reward = f1
                    metrics_info = f"F1: {f1:.3f}"
                except Exception as eval_error:
                    reward = 0.0
                    metrics_info = f"Eval error: {str(eval_error)[:50]}"
            else:
                metrics_info = "No reference steps provided"

            rewards.append(reward)

            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                with open(log_path.replace(".txt", "_reasoning_vqa.txt"), "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Reasoning VQA reward: {reward} -------------\n")
                    f.write(f"Data Index: {data_indices[i] if i < len(data_indices) else 'N/A'}\n")
                    f.write(f"Predicted steps ({len(predicted_steps)}): {predicted_steps}\n")
                    f.write(f"Reference steps ({len(ref_steps) if ref_steps else 0}): {ref_steps}\n")
                    f.write(f"Metrics: {metrics_info}\n\n")

        return rewards

    @staticmethod
    def vqa_causal_reasoning_reward(completions, **kwargs):
        """
        Causal Process Reward (CPR) for reasoning quality.

        Rewards reasoning steps based on their causal necessity for the correct answer.
        """
        import os
        from datetime import datetime

        try:
            from causal_reward import causal_intervention_reward
        except ImportError:
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', 'mllm_evaluator'))
            from causal_reward import causal_intervention_reward

        try:
            completion_contents = [completion[0]["content"] for completion in completions]
        except (KeyError, IndexError, TypeError) as e:
            print(f"Warning: Error extracting completion contents: {e}")
            return [0.0] * len(completions)

        reference_steps_list = kwargs.get("reference_steps", [])
        ground_truths = kwargs.get("ground_truth", kwargs.get("answer", kwargs.get("solution", [])))
        data_indices = kwargs.get("data_index", [])

        if not isinstance(ground_truths, list):
            ground_truths = [ground_truths] * len(completions)
        ground_truths = [str(gt) if gt is not None else "" for gt in ground_truths]

        if not isinstance(reference_steps_list, list):
            reference_steps_list = [[]] * len(completions)

        formatted_completions = [[{"content": c}] for c in completion_contents]

        answer_weight = InvernVLModule._causal_answer_weight
        step_weight = InvernVLModule._causal_step_weight

        try:
            rewards = causal_intervention_reward(
                completions=formatted_completions,
                ground_truths=ground_truths,
                reference_steps=reference_steps_list,
                data_indices=data_indices,
                answer_weight=answer_weight,
                step_weight=step_weight,
                debug_mode=os.getenv("DEBUG_MODE") == "true",
                log_path=os.getenv("LOG_PATH")
            )
        except Exception as e:
            print(f"Warning: CPR computation failed: {e}")
            rewards = [0.0] * len(completions)

        return rewards

    # ── Bounding box reward functions (original) ──

    @staticmethod
    def format_reward_rec(completions, **kwargs):
        """Check if the InternVL model output matches a specific format."""
        import re
        import os
        from datetime import datetime
        pattern = r"<think>.*?</think>\s*<answer>.*?\[\d+,\s*\d+,\s*\d+,\s*\d+\].*?</answer>"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path.replace(".txt", "_format.txt"), "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Format reward -------------\n")
                for content, match in zip(completion_contents, matches):
                    f.write(f"Content: {content}\n")
                    f.write(f"Has format: {bool(match)}\n")
        return [1.0 if match else 0.0 for match in matches]

    @staticmethod
    def iou_reward(completions, solution, **kwargs):
        """Calculate IoU reward between predicted bounding box and ground truth."""
        import re
        import os
        import json
        from datetime import datetime
        def iou(box1, box2):
            inter_x1 = max(box1[0], box2[0])
            inter_y1 = max(box1[1], box2[1])
            inter_x2 = min(box1[2]-1, box2[2]-1)
            inter_y2 = min(box1[3]-1, box2[3]-1)
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
            else:
                inter = 0
            union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
            return float(inter)/union
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'
        for i, (content, sol) in enumerate(zip(contents, solution)):
            sol = re.findall(answer_tag_pattern, sol, re.DOTALL)[-1]
            sol = json.loads(sol.strip())
            reward = 0.0
            try:
                content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
                if content_answer_match:
                    content_answer = content_answer_match.group(1).strip()
                    bbox_match = re.search(bbox_pattern, content_answer)
                    if bbox_match:
                        bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]
                        reward = iou(bbox, sol)
            except Exception:
                pass

            rewards.append(reward)
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                image_path = kwargs.get("image_path")[i] if "image_path" in kwargs else None
                problem = kwargs.get("problem")[i]
                if reward <= 1.0:
                    with open(log_path, "a", encoding='utf-8') as f:
                        f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                        f.write(f"image_path: {image_path}\n")
                        f.write(f"problem: {problem}\n")
                        f.write(f"Content: {content}\n")
                        f.write(f"Solution: {sol}\n")
        return rewards

    @staticmethod
    def select_reward_func(func: str, task_type: str):
        if func == "accuracy":
            match task_type:
                case "vqa":
                    return InvernVLModule.vqa_accuracy_reward
                case "rec":
                    return InvernVLModule.iou_reward
                case _:
                    raise ValueError(f"Unsupported reward function: {func} for task type: {task_type}")
        elif func == "format":
            match task_type:
                case "vqa":
                    return InvernVLModule.format_reward_vqa
                case "rec":
                    return InvernVLModule.format_reward_rec
                case _:
                    raise ValueError(f"Unsupported reward function: {func} for task type: {task_type}")
        elif func == "reasoning":
            match task_type:
                case "vqa":
                    return InvernVLModule.vqa_reasoning_reward
                case _:
                    raise ValueError(f"Unsupported reward function: {func} for task type: {task_type}")
        elif func == "reasoning_causal":
            match task_type:
                case "vqa":
                    return InvernVLModule.vqa_causal_reasoning_reward
                case _:
                    raise ValueError(f"Unsupported reward function: {func} for task type: {task_type}")
        else:
            raise ValueError(f"Unsupported reward function: {func}")


def process_conversation_list(conversation_list, system_message=None, image_newline=True):
    if system_message is not None:
        conversation_list = conversation_list[1:]
    processed_list = []

    for item in conversation_list:
        role = item["role"]
        content = item["content"]

        if isinstance(content, list):
            overall_str = ""
            for content_item in content:
                if content_item.get("type") == "image":
                    overall_str += "<image>" if not image_newline else "<image>\n"
                elif content_item.get("type") == "text":
                    overall_str += content_item.get("text")
                else:
                    raise ValueError(f"Unsupported content type: {type(content_item)}")
            processed_list.append(overall_str)
        elif isinstance(content, str):
            processed_list.append(content)
        else:
            raise ValueError(f"Unsupported content type: {type(content)}")

    return processed_list

def extract_system_message(conversation_list):
    if conversation_list[0]["role"] == "system":
        if isinstance(conversation_list[0]["content"], list):
            return conversation_list[0]["content"][0]["text"]
        else:
            return conversation_list[0]["content"]
    return None


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images
