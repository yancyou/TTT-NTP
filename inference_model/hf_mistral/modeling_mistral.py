# Mistral inference modeling: reuses the Llama modeling; only config_class differs.
from torch import nn

from ..hf_llama3.modeling_llama import LlamaModel, LlamaForCausalLM, LlamaPreTrainedModel
from .configuration_mistral import MistralConfig


class MistralModel(LlamaModel):
    config_class = MistralConfig


class MistralForCausalLM(LlamaForCausalLM):
    config_class = MistralConfig

    def __init__(self, config):
        LlamaPreTrainedModel.__init__(self, config)
        self.model = MistralModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()


__all__ = ["MistralModel", "MistralForCausalLM"]
