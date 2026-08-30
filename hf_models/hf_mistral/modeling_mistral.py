# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""TTT-enabled Mistral modeling.

Mistral has the same state_dict structure as Llama (gate/up/down MLP,
q/k/v/o attention, RMSNorm, RoPE). We subclass our TTT-enabled
LlamaModel / LlamaForCausalLM.

Mistral-7B-v0.3: rope_theta=1e6, no rope_scaling, sliding_window=None -> full attention.
Mistral-7B-v0.1: sliding_window=4096 (SWA), rope_theta=1e4. For v0.1 we MUST honor the
sliding window: its RoPE was only trained for within-window (<=4096) relative positions,
so forcing full attention makes RoPE out-of-distribution beyond the window and collapses
long context. We therefore build a sliding-window causal mask when config.sliding_window
is set; otherwise we fall back to the plain causal mask (v0.3 behaviour, unchanged).
"""

import torch
from torch import nn

from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import auto_docstring
from transformers.utils.generic import check_model_inputs

from ..hf_llama.modeling_llama import LlamaModel, LlamaForCausalLM, LlamaPreTrainedModel
from .configuration_mistral import MistralConfig


class MistralModel(LlamaModel):
    config_class = MistralConfig

    @check_model_inputs()
    @auto_docstring
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        cache_position=None,
        use_cache=None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # SWA (Mistral-v0.1): honor sliding_window; else plain causal (v0.3 unchanged).
        mask_fn = create_sliding_window_causal_mask if getattr(self.config, "sliding_window", None) else create_causal_mask
        causal_mask = mask_fn(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                target_states=self._resolve_ttt_target_states(decoder_layer, inputs_embeds),
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class MistralForCausalLM(LlamaForCausalLM):
    config_class = MistralConfig

    def __init__(self, config):
        # Mirror LlamaForCausalLM.__init__ but build the SWA-aware MistralModel
        # (avoids constructing a throwaway LlamaModel).
        LlamaPreTrainedModel.__init__(self, config)
        self.model = MistralModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()


__all__ = ["MistralModel", "MistralForCausalLM"]
