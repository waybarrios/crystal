from .vlm_module import VLMBaseModule
from .qwen_module import Qwen2VLModule
from .internvl_module import InvernVLModule

try:
    from .glm_module import GLMVModule
    __all__ = ["VLMBaseModule", "Qwen2VLModule", "InvernVLModule", "GLMVModule"]
except ImportError:
    # GLMVModule requires newer transformers (Glm4vForConditionalGeneration)
    # Skip if not available - only needed for GLM models
    __all__ = ["VLMBaseModule", "Qwen2VLModule", "InvernVLModule"]