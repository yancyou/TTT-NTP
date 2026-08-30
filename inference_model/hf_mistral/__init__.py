from .configuration_mistral import MistralConfig
from .modeling_mistral import MistralModel, MistralForCausalLM
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

AutoConfig.register("mistral", MistralConfig, exist_ok=True)
AutoModel.register(MistralConfig, MistralModel, exist_ok=True)
AutoModelForCausalLM.register(MistralConfig, MistralForCausalLM, exist_ok=True)

__all__ = ["MistralConfig"]
