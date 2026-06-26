
# ----------------------- Fix ZeRO-3 compatibility for InternVL models in GRPO training -----------------------
# Similar to qwen2_5vl_monkey_patch.py but for InternVL architecture
#
# Problem: InternVL's original forward() requires pixel_values (not optional) and its custom
# generate() bypasses forward(), causing DeepSpeed ZeRO-3's module trace to desync
# (IndexError: pop from an empty deque).
#
# Solution: Monkey-patch forward() to make pixel_values optional + add ZeRO-3 dummy vision
# forward for cross-GPU sync. Remove custom generate() so standard HF generate pipeline
# is used (which calls forward() consistently).

import torch
from typing import List, Optional, Tuple, Union
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast


def internvl_forward_zero3(
    self,
    pixel_values: Optional[torch.FloatTensor] = None,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    image_flags: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    """
    ZeRO-3 compatible forward for InternVL.

    Key changes from original:
    1. pixel_values is Optional (was required) - needed for autoregressive generation steps
    2. Uses all_reduce to sync image presence across GPUs for ZeRO-3
    3. Runs dummy vision forward on GPUs without images (ZeRO-3 parameter sync)
    4. Handles inputs_embeds for flexibility
    """
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # Build text embeddings
    if inputs_embeds is None:
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()
    else:
        input_embeds = inputs_embeds

    # Determine if this rank has images to process
    has_pixel_values = pixel_values is not None

    # ZeRO-3: all ranks must execute the same modules. Use all_reduce to check globally.
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        has_images_local = torch.tensor(1 if has_pixel_values else 0, device=input_embeds.device)
        torch.distributed.all_reduce(has_images_local, op=torch.distributed.ReduceOp.MAX)
        has_images_global = has_images_local.item() > 0
    else:
        has_images_global = has_pixel_values

    if has_images_global:
        if has_pixel_values:
            # Normal image processing
            vit_embeds = self.extract_feature(pixel_values)

            # Filter by image_flags if provided (flags=1 means real image patch)
            # If image_flags is None, treat all patches as real (same as original generate())
            if image_flags is not None:
                image_flags = image_flags.squeeze(-1)
                vit_embeds = vit_embeds[image_flags == 1]

            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)
            flat_input_ids = input_ids.reshape(B * N)
            selected = (flat_input_ids == self.img_context_token_id)
            try:
                input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
            except Exception as e:
                vit_embeds = vit_embeds.reshape(-1, C)
                n_token = min(selected.sum(), vit_embeds.size(0))
                input_embeds[selected][:n_token] = input_embeds[selected][:n_token] * 0.0 + vit_embeds[:n_token]
            input_embeds = input_embeds.reshape(B, N, C)
        else:
            # Dummy vision forward for ZeRO-3 parameter synchronization.
            # All GPUs must call vision_model + mlp1 even if this rank has no images.
            with torch.no_grad():
                img_size = self.config.vision_config.image_size
                dummy_pixel = torch.zeros(
                    (1, 3, img_size, img_size),
                    device=input_embeds.device,
                    dtype=next(self.vision_model.parameters()).dtype,
                )
                _ = self.extract_feature(dummy_pixel)

    outputs = self.language_model(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    logits = outputs.logits

    loss = None
    if labels is not None:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def internvl_prepare_inputs_for_generation(
    self, input_ids, past_key_values=None, attention_mask=None,
    pixel_values=None, image_flags=None, inputs_embeds=None, **kwargs
):
    """
    Prepare inputs for each autoregressive generation step.

    First step (no KV cache): include pixel_values so vision encoder runs.
    Subsequent steps (has KV cache): only pass the new token, no images
    (image context is already in the KV cache).
    """
    if past_key_values is not None:
        # After first step: only the last generated token, no images
        input_ids = input_ids[:, -1:]
        pixel_values = None
        image_flags = None

    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "pixel_values": pixel_values,
        "image_flags": image_flags,
        "use_cache": True,
    }
    return model_inputs


def monkey_patch_internvl_forward(model_id="OpenGVLab/InternVL3_5-4B"):
    """
    Apply ZeRO-3 compatible monkey patches to InternVLChatModel.

    1. Replaces forward() with ZeRO-3 compatible version (optional pixel_values + dummy vision sync)
    2. Adds prepare_inputs_for_generation() for proper KV cache handling
    3. Removes custom generate() so standard HF generate pipeline is used
    4. Sets main_input_name to 'input_ids' (was 'pixel_values')
    5. Updates InvernVLModule.is_embeds_input() since standard generate returns full sequence
    """
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    try:
        InternVLChatModel = get_class_from_dynamic_module(
            "modeling_internvl_chat.InternVLChatModel", model_id
        )
    except (OSError, ValueError, ImportError):
        # Fallback for local checkpoints that don't have modeling files.
        # The class is already cached from the original model download.
        InternVLChatModel = get_class_from_dynamic_module(
            "modeling_internvl_chat.InternVLChatModel", "OpenGVLab/InternVL3_5-4B"
        )

    # 1. Patch forward for ZeRO-3 compatibility
    InternVLChatModel.forward = internvl_forward_zero3

    # 2. Add prepare_inputs_for_generation for proper autoregressive generation
    InternVLChatModel.prepare_inputs_for_generation = internvl_prepare_inputs_for_generation

    # 3. Remove custom generate() so standard HF GenerationMixin.generate() is used.
    #    The custom generate() bypasses forward() which breaks ZeRO-3 module tracking.
    if 'generate' in InternVLChatModel.__dict__:
        delattr(InternVLChatModel, 'generate')

    # 4. Standard generate expects main_input_name='input_ids' to find the input tensor.
    #    Original InternVL sets it to 'pixel_values' which confuses the standard generate.
    InternVLChatModel.main_input_name = 'input_ids'

    # 5. Update VLM module: with standard generate, generate() returns full prompt+completion
    #    (not just completion), so is_embeds_input should be False.
    from open_r1.vlm_modules.internvl_module import InvernVLModule
    InvernVLModule._zero3_patched = True

    print("InternVL ZeRO-3 monkey patch applied (forward + generate + prepare_inputs_for_generation)")


# ----------------------- Fix torch.load weights_only for PyTorch 2.6+ -----------------------
# Same fix as in qwen2_5vl_monkey_patch.py - needed for DeepSpeed checkpoint loading

def monkey_patch_torch_load():
    from deepspeed.runtime.checkpoint_engine.torch_checkpoint_engine import TorchCheckpointEngine
    from deepspeed.utils import logger

    def weights_only_load(self, path: str, map_location=None):
        logger.info(f"[Torch] Loading checkpoint from {path}...")
        partition = torch.load(path, map_location=map_location, weights_only=False)
        logger.info(f"[Torch] Loaded checkpoint from {path}.")
        return partition

    TorchCheckpointEngine.load = weights_only_load
    print("Torch checkpoint weights_only=False patch applied")
