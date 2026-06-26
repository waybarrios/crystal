from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
from typing import Dict, Any, Union
from trl.data_utils import maybe_apply_chat_template
import torch
from copy import deepcopy
from open_r1.vlm_modules.vlm_module import VLMBaseModule
from PIL import Image
import sys
import os

# Import from mllm_evaluator (added to PYTHONPATH by training script)
from accuracy_calculator import AccuracyCalculator
from mllm_evaluator import MLLMReasoningEvaluator

class Qwen2VLModule(VLMBaseModule):
    # Class variables for LLM judge configuration
    _use_llm_judge = False
    _llm_judge_model = "gpt-oss:20b"
    _llm_judge_base_url = "http://localhost:11434/v1"

    # Class variable for reasoning evaluator (process-local to avoid threading issues)
    _reasoning_evaluator = None
    _evaluator_process_id = None
    _sentence_transformer_model = None
    _sentence_transformer_pid = None

    # Semantic Process Reward (SPR) configuration
    _semantic_similarity_threshold = 0.70

    def __init__(self):
        super().__init__()

    @classmethod
    def get_sentence_transformer(cls):
        """Get or initialize SentenceTransformer model (process-local, CPU-only)."""
        import os
        current_pid = os.getpid()

        # Reinitialize if we're in a different process (DeepSpeed fork)
        if cls._sentence_transformer_model is None or cls._sentence_transformer_pid != current_pid:
            try:
                from sentence_transformers import SentenceTransformer
                import torch

                # Use CPU only to avoid CUDA tensor shape issues in multi-process
                device = "cpu"

                # Disable tokenizer parallelism warnings
                os.environ["TOKENIZERS_PARALLELISM"] = "false"

                # Load model on CPU
                cls._sentence_transformer_model = SentenceTransformer("all-distilroberta-v1", device=device)
                cls._sentence_transformer_model.eval()
                cls._sentence_transformer_pid = current_pid

                print(f"✓ SentenceTransformer initialized for PID {current_pid} (device=cpu)")
            except Exception as e:
                print(f"⚠ SentenceTransformer initialization failed for PID {current_pid}: {e}")
                cls._sentence_transformer_model = None
                cls._sentence_transformer_pid = current_pid

        return cls._sentence_transformer_model

    @classmethod
    def configure_semantic_reward(cls, threshold: float = 0.70):
        """Configure Semantic Process Reward (SPR) threshold.

        SPR uses SentenceTransformer embeddings + cosine similarity instead of word overlap.
        This captures semantic equivalence: "green light" ≈ "traffic signal shows green"
        """
        cls._semantic_similarity_threshold = threshold
        print(f"✓ Semantic Process Reward configured: threshold={threshold}")

    @classmethod
    def get_reasoning_evaluator(cls):
        """Get or initialize the reasoning evaluator (process-local singleton)."""
        import os
        import torch
        current_pid = os.getpid()

        # Reinitialize if we're in a different process (DeepSpeed fork)
        if cls._reasoning_evaluator is None or cls._evaluator_process_id != current_pid:
            # Clean up old evaluator if exists
            if cls._reasoning_evaluator is not None:
                try:
                    del cls._reasoning_evaluator
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except:
                    pass

            try:
                # Determine device: use local GPU for this process
                # DeepSpeed assigns each process a local rank, use that GPU
                local_rank = int(os.environ.get("LOCAL_RANK", "0"))
                device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

                # Force fresh model load by setting environment variable
                # This prevents issues with model state being inherited from parent process
                os.environ["TOKENIZERS_PARALLELISM"] = "false"

                # Initialize with GPU to avoid CPU tensor shape issues in multi-process context
                # Use threshold 0.35 (validated by ablation study across 5 VLMs)
                cls._reasoning_evaluator = MLLMReasoningEvaluator(
                    model_name="all-distilroberta-v1",
                    similarity_threshold=0.35,  # Ablation-validated threshold
                    device=device,  # Use local GPU to avoid CPU multi-process tensor issues
                    debug_mode=False
                )
                cls._evaluator_process_id = current_pid
                print(f"✓ MLLMReasoningEvaluator initialized for PID {current_pid} (device={device}, threshold={cls._reasoning_evaluator.similarity_threshold:.3f})")
            except Exception as init_error:
                print(f"⚠ MLLMReasoningEvaluator initialization failed for PID {current_pid}: {init_error}")
                print(f"  Will use simple_similarity fallback for this process")
                cls._reasoning_evaluator = None
                cls._evaluator_process_id = current_pid

        return cls._reasoning_evaluator

    @classmethod
    def configure_llm_judge(cls, use_llm_judge=False, llm_judge_model="gpt-oss:20b", llm_judge_base_url="http://localhost:11434/v1"):
        """Configure LLM judge settings for accuracy evaluation."""
        cls._use_llm_judge = use_llm_judge
        cls._llm_judge_model = llm_judge_model
        cls._llm_judge_base_url = llm_judge_base_url

    def get_vlm_key(self):
        return "qwen"

    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        if "Qwen2-VL" in model_id:
            model_cls = Qwen2VLForConditionalGeneration
        elif "Qwen2.5-VL" in model_id or "qwen2.5-vl" in model_id.lower():
            model_cls = Qwen2_5_VLForConditionalGeneration
        else:
            # For checkpoint paths, check config.json for model_type
            import json, os
            config_path = os.path.join(model_id, "config.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    config = json.load(f)
                model_type = config.get("model_type", "")
                if "qwen2_5_vl" in model_type:
                    return Qwen2_5_VLForConditionalGeneration
                elif "qwen2_vl" in model_type:
                    return Qwen2VLForConditionalGeneration
            raise ValueError(f"Unsupported model: {model_id}")
        return model_cls
    
    def post_model_init(self, model, processing_class):
        pass
    
    def get_processing_class(self):
        return AutoProcessor
    
    def get_vision_modules_keywords(self):  
        return ['visual']
    
    def get_custom_multimodal_keywords(self):
        return ['pixel_values', 'image_grid_thw']

    def get_non_generate_params(self):
        return []
    
    def get_custom_processing_keywords(self):
        return [('image_processor', 'max_pixels'), ('image_processor', 'min_pixels')]
    
    def prepare_prompt(self, processing_class, inputs: dict[str, Union[torch.Tensor, Any]]):
        prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"] for example in inputs]
        return prompts_text
    
    def prepare_model_inputs(self, processing_class, prompts_text, images, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False):
        # FIXME
        # This could only process pure-multimodal or pure-text inputs
        additional_output = None
        if len(images) > 0:
            prompt_inputs = processing_class(
                text=prompts_text,
                images=images,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
            additional_output = [{'image_grid_thw': image_grid_thw} for image_grid_thw in prompt_inputs['image_grid_thw']]
        else:
            prompt_inputs = processing_class(
                text=prompts_text,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        return prompt_inputs, additional_output
    
    @staticmethod
    def get_question_template(task_type: str):
        match task_type:
            case "vqa":
                return "{USER_INSTRUCTION}"
            case "rec":
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
            case "ic":
                return "{Question} First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> json format answer here </answer>"
            case "odLength":
                SYSTEM_PROMPT = (
                    #"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
                    "First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
                    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
                    "<think> reasoning process here </think><answer> answer here </answer>"
                )
                return SYSTEM_PROMPT + '\n' + "{Question}"
            case _:
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."
            
    @staticmethod
    def format_reward_rec(completions, **kwargs):
        """Check if the Qwen model output matches a specific format."""
        import re
        import os
        from datetime import datetime
        pattern = r"<think>.*?</think>\s*<answer>.*?\{.*\[\d+,\s*\d+,\s*\d+,\s*\d+\].*\}.*?</answer>"
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
        """Calculate IoU reward between predicted bounding box from Qwen model and ground truth bounding box."""
        import re
        import os
        from datetime import datetime
        import json
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
        def resize_bbox(bbox, input_height, input_width, image_height, image_width):
            bbox[0] = bbox[0] / input_width * image_width
            bbox[1] = bbox[1] / input_height * image_height
            bbox[2] = bbox[2] / input_width * image_width
            bbox[3] = bbox[3] / input_height * image_height
            return bbox
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'

        for i, (content, sol) in enumerate(zip(contents, solution)):
            image_grid_thw = kwargs.get("image_grid_thw")[i]
            image_path = kwargs.get("image_path")[i][0]
            image = Image.open(image_path)
            image_width, image_height = image.size
            input_height = int(image_grid_thw[1]*14)
            input_width = int(image_grid_thw[2]*14)
            
            sol = re.findall(answer_tag_pattern, sol, re.DOTALL)[-1]
            sol = json.loads(sol.strip())
            reward = 0.0
            # Try symbolic verification first
            try:
                content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
                if content_answer_match:
                    content_answer = content_answer_match.group(1).strip()
                    bbox_match = re.search(bbox_pattern, content_answer)
                    if bbox_match:
                        bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]
                        bbox = resize_bbox(bbox, input_height, input_width, image_height, image_width)
                        # if iou(bbox, sol) > 0.5:
                        #     reward = 1.0
                        reward = iou(bbox, sol)
            except Exception:
                pass  # Continue to next verification method if this fails
                    
            rewards.append(reward)
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                image_path = kwargs.get("image_path")[i] if "image_path" in kwargs else None
                problem = kwargs.get("problem")[i]
                if reward <= 1.0:  # this condition can be changed for debug
                    with open(log_path, "a", encoding='utf-8') as f:
                        f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                        f.write(f"image_path: {image_path}\n")
                        f.write(f"problem: {problem}\n")
                        f.write(f"Content: {content}\n")
                        f.write(f"Solution: {sol}\n")
        return rewards

    @staticmethod
    def format_reward_vqa(completions, **kwargs):
        """Check if the Qwen model output is valid JSON with reasoning_steps and answer fields."""
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
                # Remove markdown code fences if present
                content_cleaned = re.sub(r'```json\s*|\s*```', '', content).strip()

                # Method 1: Try to parse the entire content as JSON
                try:
                    parsed = json.loads(content_cleaned)
                except json.JSONDecodeError:
                    # Method 2: Find outermost { } pair and extract that
                    first_brace = content_cleaned.find('{')
                    if first_brace != -1:
                        # Find matching closing brace using brace counting
                        brace_count = 0
                        for idx in range(first_brace, len(content_cleaned)):
                            if content_cleaned[idx] == '{':
                                brace_count += 1
                            elif content_cleaned[idx] == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    json_str = content_cleaned[first_brace:idx+1]

                                    # Fix trailing commas before parsing
                                    json_str = re.sub(r',(\s*[\]}])', r'\1', json_str)

                                    parsed = json.loads(json_str)
                                    break

                # Validate structure if we successfully parsed JSON
                if parsed is not None:
                    # Check if it has the required keys with correct types
                    if "reasoning_steps" in parsed and "answer" in parsed:
                        # Check that reasoning_steps is a list AND all items are strings (not objects)
                        if isinstance(parsed["reasoning_steps"], list) and isinstance(parsed["answer"], str):
                            # Additional validation: must have at least 1 string in reasoning_steps
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
                    f.write(f"Image File: {image_files[i] if i < len(image_files) else 'N/A'}\n")
                    f.write(f"Problem: {problems[i] if i < len(problems) else 'N/A'}\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {solutions[i] if i < len(solutions) else 'N/A'}\n")
                    f.write(f"Has valid format: {reward == 1.0}\n")
                    f.write(f"\n")

        return rewards

    @staticmethod
    def vqa_accuracy_reward(completions, solution, **kwargs):
        """Calculate accuracy reward using accuracy_calculator for VQA task."""
        import json
        import re
        import os
        from datetime import datetime

        # Get LLM judge configuration from kwargs or class variables
        use_llm_grader = kwargs.get("use_llm_judge", Qwen2VLModule._use_llm_judge)
        llm_model = kwargs.get("llm_judge_model", Qwen2VLModule._llm_judge_model)
        base_url = kwargs.get("llm_judge_base_url", Qwen2VLModule._llm_judge_base_url)

        # Initialize accuracy calculator with optional LLM grader
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
                # Extract JSON from content
                content_cleaned = re.sub(r'```json\s*|\s*```', '', content).strip()

                json_str = None
                parsed = None

                # Method 1: Try to parse the entire content as JSON
                try:
                    parsed = json.loads(content_cleaned)
                    if "reasoning_steps" in parsed and "answer" in parsed:
                        json_str = content_cleaned  # For logging purposes
                except (json.JSONDecodeError, Exception):
                    pass

                # Method 2: Find outermost { } pair and extract that
                if parsed is None:
                    first_brace = content_cleaned.find('{')
                    if first_brace != -1:
                        # Find matching closing brace
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
                    # Fix common JSON issues
                    # 1. Replace smart quotes with regular quotes
                    json_str = json_str.replace('"', '"').replace('"', '"').replace("'", "'").replace("'", "'")

                    # 2. Replace single quotes with double quotes for string values
                    json_str = re.sub(r'''(?<=[:,\[])\s*'([^']*)'(?=\s*[,\]\}])''', r' "\1"', json_str)

                    # 3. Add missing commas between array items
                    json_str = re.sub(r'"\s*\n\s*"', '",\n    "', json_str)
                    json_str = re.sub(r'"\s+(?=")', '", ', json_str)

                    # 4. Remove trailing commas before closing brackets/braces
                    json_str = re.sub(r',(\s*[\]}])', r'\1', json_str)

                    # Try parsing (if not already parsed in Method 1)
                    if parsed is None:
                        try:
                            parsed = json.loads(json_str)
                        except json.JSONDecodeError:
                            # If still fails, try more aggressive fixing
                            json_str_fixed = json_str.replace("'", '"')
                            # Try one more time with comma fixes
                            json_str_fixed = re.sub(r'"\s*\n\s*"', '",\n    "', json_str_fixed)
                            json_str_fixed = re.sub(r'"\s+(?=")', '", ', json_str_fixed)
                            # Remove trailing commas
                            json_str_fixed = re.sub(r',(\s*[\]}])', r'\1', json_str_fixed)
                            parsed = json.loads(json_str_fixed)

                    predicted_answer = parsed.get("answer", "") if parsed else ""

                    # Evaluate accuracy
                    if predicted_answer:
                        result = accuracy_calc.evaluate_single(problem, predicted_answer, sol)
                        if result.is_correct:
                            reward = result.confidence

            except (json.JSONDecodeError, Exception) as e:
                # Log JSON errors in debug mode
                if os.getenv("DEBUG_MODE") == "true":
                    import traceback
                    log_path = os.getenv("LOG_PATH")
                    with open(log_path.replace(".txt", "_accuracy_errors.txt"), "a", encoding='utf-8') as f:
                        f.write(f"------------- {current_time} JSON Parse Error -------------\n")
                        f.write(f"Data Index: {data_indices[i] if i < len(data_indices) else 'N/A'}\n")
                        f.write(f"Error: {str(e)}\n")
                        f.write(f"Content: {content}\n")
                        f.write(f"\n")

            rewards.append(reward)

            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                with open(log_path.replace(".txt", "_accuracy_vqa.txt"), "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Accuracy VQA reward: {reward} -------------\n")
                    f.write(f"Data Index: {data_indices[i] if i < len(data_indices) else 'N/A'}\n")
                    f.write(f"Image File: {image_files[i] if i < len(image_files) else 'N/A'}\n")
                    f.write(f"Problem: {problem}\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Predicted Answer: {predicted_answer}\n")
                    f.write(f"Ground Truth: {sol}\n")
                    if reward == 0.0 and predicted_answer:
                        f.write(f"⚠️ Note: Answer extracted but marked incorrect (check normalization/matching)\n")
                    elif reward == 0.0:
                        f.write(f"⚠️ Note: Failed to extract answer from JSON\n")
                    f.write(f"\n")

        return rewards

    @staticmethod
    def _parse_and_validate_json(content: str):
        """Parse and validate JSON from model output.

        Returns:
            tuple: (parsed_dict, is_valid, error_message)
        """
        import json
        import re

        # Remove markdown code blocks
        content_cleaned = re.sub(r'```json\s*|\s*```', '', content).strip()

        json_str = None
        parsed = None

        # Method 1: Try to parse entire content as JSON
        try:
            parsed = json.loads(content_cleaned)
            json_str = content_cleaned
        except (json.JSONDecodeError, Exception):
            pass

        # Method 2: Extract outermost { } pair
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
            # Fix common JSON issues
            json_str = json_str.replace('"', '"').replace('"', '"').replace("'", "'").replace("'", "'")
            json_str = re.sub(r'''(?<=[:,\[])\s*'([^']*)'(?=\s*[,\]\}])''', r' "\1"', json_str)
            json_str = re.sub(r'"\s*\n\s*"', '",\n    "', json_str)
            json_str = re.sub(r'"\s+(?=")', '", ', json_str)
            json_str = re.sub(r',(\s*[\]}])', r'\1', json_str)

            # Try parsing
            if parsed is None:
                try:
                    parsed = json.loads(json_str)
                except json.JSONDecodeError as e:
                    return None, False, f"JSON parse error: {str(e)}"

        # Validate structure
        if parsed is None:
            return None, False, "No valid JSON found"

        if not isinstance(parsed, dict):
            return None, False, "JSON is not an object"

        # Check required keys
        if "reasoning_steps" not in parsed:
            return None, False, "Missing 'reasoning_steps' key"

        if "answer" not in parsed:
            return None, False, "Missing 'answer' key"

        # Validate types
        if not isinstance(parsed["reasoning_steps"], list):
            return None, False, "'reasoning_steps' must be a list"

        if not isinstance(parsed["answer"], str):
            return None, False, "'answer' must be a string"

        # Validate that all items in reasoning_steps are strings (not objects/dicts)
        for idx, step in enumerate(parsed["reasoning_steps"]):
            if not isinstance(step, str):
                return None, False, f"'reasoning_steps[{idx}]' must be a string, not {type(step).__name__}"

        # Clean and validate reasoning steps
        reasoning_steps = [s.strip() for s in parsed["reasoning_steps"] if s and s.strip()]

        if not reasoning_steps:
            return None, False, "'reasoning_steps' is empty or contains no valid strings"

        # Return validated data
        return {
            "reasoning_steps": reasoning_steps,
            "answer": parsed["answer"].strip()
        }, True, None

    @staticmethod
    def vqa_reasoning_reward(completions, **kwargs):
        """Calculate reasoning quality reward using mllm_evaluator (neural semantic similarity)."""
        import os
        from datetime import datetime

        completion_contents = [completion[0]["content"] for completion in completions]
        reference_steps_list = kwargs.get("reference_steps", [])
        problems = kwargs.get("problem", [])
        solutions = kwargs.get("solution", [])
        image_files = kwargs.get("image_file", [])
        data_indices = kwargs.get("data_index", [])
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

        for i, (content, ref_steps) in enumerate(zip(completion_contents, reference_steps_list)):
            reward = 0.0
            predicted_steps = []
            metrics_info = ""

            # Step 1: Parse and validate JSON
            parsed_data, is_valid, error_msg = Qwen2VLModule._parse_and_validate_json(content)

            if not is_valid:
                # JSON validation failed
                metrics_info = f"Validation failed: {error_msg}"
                rewards.append(reward)

                if os.getenv("DEBUG_MODE") == "true":
                    log_path = os.getenv("LOG_PATH")
                    with open(log_path.replace(".txt", "_reasoning_vqa.txt"), "a", encoding='utf-8') as f:
                        f.write(f"------------- {current_time} Reasoning VQA reward: {reward} -------------\n")
                        f.write(f"Data Index: {data_indices[i] if i < len(data_indices) else 'N/A'}\n")
                        f.write(f"Image File: {image_files[i] if i < len(image_files) else 'N/A'}\n")
                        f.write(f"Problem: {problems[i] if i < len(problems) else 'N/A'}\n")
                        f.write(f"Content: {content}\n")
                        f.write(f"Solution: {solutions[i] if i < len(solutions) else 'N/A'}\n")
                        f.write(f"Predicted steps (0): []\n")
                        f.write(f"Reference steps ({len(ref_steps) if ref_steps else 0}): {ref_steps}\n")
                        f.write(f"Metrics: {metrics_info}\n\n")
                continue

            # Step 2: Extract validated data
            predicted_steps = parsed_data["reasoning_steps"]

            # Step 3: Evaluate reasoning if reference steps exist
            if ref_steps:
                # Clean reference steps
                ref_steps_cleaned = [str(s).strip() for s in ref_steps if s and str(s).strip()]

                if not ref_steps_cleaned:
                    metrics_info = "No valid reference steps"
                    rewards.append(reward)
                    continue

                # Use simple_similarity for training (word overlap with threshold 0.45)
                # Neural semantic similarity (mllm_evaluator) is not compatible with
                # DeepSpeed's multi-process forking, causing 'weight' must be 2-D errors
                # For inference evaluation, use evaluate_predictions.py which uses mllm_evaluator
                from simple_similarity import best_match_f1
                try:
                    f1, matched_pred, matched_ref = best_match_f1(predicted_steps, ref_steps_cleaned, threshold=0.45)
                    reward = f1
                    precision = matched_pred / len(predicted_steps) if predicted_steps else 0.0
                    recall = matched_ref / len(ref_steps_cleaned) if ref_steps_cleaned else 0.0
                    metrics_info = (f"F1: {f1:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, "
                                  f"Matches: {matched_pred}/{len(predicted_steps)}")
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
                    f.write(f"Image File: {image_files[i] if i < len(image_files) else 'N/A'}\n")
                    f.write(f"Problem: {problems[i] if i < len(problems) else 'N/A'}\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {solutions[i] if i < len(solutions) else 'N/A'}\n")
                    f.write(f"Predicted steps ({len(predicted_steps)}): {predicted_steps}\n")
                    f.write(f"Reference steps ({len(ref_steps) if ref_steps else 0}): {ref_steps}\n")
                    f.write(f"Metrics: {metrics_info}\n")
                    f.write(f"\n")

        return rewards

    @staticmethod
    def vqa_reasoning_reward_semantic(completions, **kwargs):
        """Calculate reasoning reward using SEMANTIC similarity (Semantic Process Reward).

        Uses SentenceTransformer (CPU-only) + cosine similarity instead of word overlap.
        Algorithm is the same as MLLMReasoningEvaluator but with pre-initialized model
        for DeepSpeed compatibility.

        Benefits over word overlap:
        - "The light is green" ≈ "Traffic signal shows green" (captures equivalence)
        - Lower training variance, more stable gradients
        - Aligned with CRYSTAL thesis: reasoning quality matters
        """
        import os
        from datetime import datetime

        completion_contents = [completion[0]["content"] for completion in completions]
        reference_steps_list = kwargs.get("reference_steps", [])
        data_indices = kwargs.get("data_index", [])
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

        # Get threshold (model is managed internally by semantic_match_f1)
        threshold = Qwen2VLModule._semantic_similarity_threshold

        for i, (content, ref_steps) in enumerate(zip(completion_contents, reference_steps_list)):
            reward = 0.0
            predicted_steps = []
            metrics_info = ""

            # Parse and validate JSON
            parsed_data, is_valid, error_msg = Qwen2VLModule._parse_and_validate_json(content)

            if not is_valid:
                metrics_info = f"Validation failed: {error_msg}"
                rewards.append(reward)
                continue

            predicted_steps = parsed_data["reasoning_steps"]

            # Evaluate with SEMANTIC similarity (uses process-local model internally)
            if ref_steps:
                ref_steps_cleaned = [str(s).strip() for s in ref_steps if s and str(s).strip()]

                if ref_steps_cleaned:
                    try:
                        from simple_similarity import semantic_match_f1
                        # semantic_match_f1 handles model internally with process-local caching
                        f1, matched_pred, matched_ref = semantic_match_f1(
                            predicted_steps, ref_steps_cleaned,
                            model=None,  # Use process-local model
                            threshold=threshold
                        )
                        reward = f1
                        precision = matched_pred / len(predicted_steps) if predicted_steps else 0.0
                        recall = matched_ref / len(ref_steps_cleaned) if ref_steps_cleaned else 0.0
                        metrics_info = f"Semantic F1={f1:.3f}, P={precision:.3f}, R={recall:.3f}"
                    except Exception as e:
                        # Fallback to word overlap if semantic fails
                        from simple_similarity import best_match_f1
                        f1, _, _ = best_match_f1(predicted_steps, ref_steps_cleaned, threshold=0.45)
                        reward = f1
                        metrics_info = f"Fallback: {str(e)[:30]}"

            rewards.append(reward)

            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                with open(log_path.replace(".txt", "_reasoning_semantic.txt"), "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Semantic reward: {reward:.3f} -------------\n")
                    f.write(f"Index: {data_indices[i] if i < len(data_indices) else 'N/A'}\n")
                    f.write(f"Metrics: {metrics_info}\n\n")

        return rewards

    # Causal reward configuration
    _causal_answer_weight = 0.6
    _causal_step_weight = 0.4

    @classmethod
    def configure_causal_reward(cls, answer_weight: float = 0.6, step_weight: float = 0.4):
        """Configure Causal Process Reward weights."""
        cls._causal_answer_weight = answer_weight
        cls._causal_step_weight = step_weight
        print(f"Causal Process Reward configured: answer_weight={answer_weight}, step_weight={step_weight}")

    @staticmethod
    def vqa_causal_reasoning_reward(completions, **kwargs):
        """
        Causal Process Reward (CPR) for reasoning quality.

        Rewards reasoning steps based on their causal necessity for the correct answer.
        Uses a multiplicative interaction between answer correctness and step alignment
        to ensure both are required for high rewards.

        This addresses the "correct answer, wrong reasoning" problem by penalizing
        cases where the answer is correct but reasoning doesn't align with reference.
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
        # Try multiple keys for ground truth (dataset uses 'solution')
        ground_truths = kwargs.get("ground_truth", kwargs.get("answer", kwargs.get("solution", [])))
        data_indices = kwargs.get("data_index", [])

        # Ensure ground_truths is a list of strings
        if not isinstance(ground_truths, list):
            ground_truths = [ground_truths] * len(completions)
        ground_truths = [str(gt) if gt is not None else "" for gt in ground_truths]

        # Ensure reference_steps_list is properly formatted
        if not isinstance(reference_steps_list, list):
            reference_steps_list = [[]] * len(completions)

        # Prepare completions in expected format
        formatted_completions = [[{"content": c}] for c in completion_contents]

        # Get configured weights
        answer_weight = Qwen2VLModule._causal_answer_weight
        step_weight = Qwen2VLModule._causal_step_weight

        try:
            # Compute rewards
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

    @staticmethod
    def select_reward_func(func: str, task_type: str):
        if func == "accuracy":
            match task_type:
                case "vqa":
                    return Qwen2VLModule.vqa_accuracy_reward
                case "rec":
                    return Qwen2VLModule.iou_reward
                case _:
                    raise ValueError(f"Unsupported reward function: {func} for task type: {task_type}")
        elif func == "format":
            match task_type:
                case "vqa":
                    return Qwen2VLModule.format_reward_vqa
                case "rec":
                    return Qwen2VLModule.format_reward_rec
                case _:
                    raise ValueError(f"Unsupported reward function: {func} for task type: {task_type}")
        elif func == "reasoning":
            match task_type:
                case "vqa":
                    return Qwen2VLModule.vqa_reasoning_reward
                case _:
                    raise ValueError(f"Unsupported reward function: {func} for task type: {task_type}")
        elif func == "reasoning_semantic":
            # Semantic Process Reward - uses SentenceTransformer instead of word overlap
            match task_type:
                case "vqa":
                    return Qwen2VLModule.vqa_reasoning_reward_semantic
                case _:
                    raise ValueError(f"Unsupported reward function: {func} for task type: {task_type}")
        elif func == "reasoning_causal":
            # Causal Process Reward - rewards causally necessary reasoning steps
            match task_type:
                case "vqa":
                    return Qwen2VLModule.vqa_causal_reasoning_reward
                case _:
                    raise ValueError(f"Unsupported reward function: {func} for task type: {task_type}")
        else:
            raise ValueError(f"Unsupported reward function: {func}")
