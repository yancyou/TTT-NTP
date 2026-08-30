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

"""Mistral model configuration with TTT support.

Mistral-{7B,Nemo,Small} has the same MLP/Attn structure as Llama at the
state_dict level. We subclass LlamaConfig and switch model_type to "mistral"
so HuggingFace's AutoConfig routes Mistral checkpoints to our TTT-enabled
LlamaForCausalLM. The only structural quirk is `sliding_window`, which we
accept and ignore (full attention is used during CPT/TTT).
"""

from ..hf_llama3.configuration_llama import LlamaConfig


class MistralConfig(LlamaConfig):
    model_type = "mistral"

    def __init__(
        self,
        # Mistral defaults
        vocab_size=32768,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=32768,
        rms_norm_eps=1e-5,
        rope_theta=1000000.0,
        tie_word_embeddings=False,
        # Mistral-specific (accepted but unused by our modeling)
        sliding_window=None,
        ttt_layers=[0, 6, 12, 18, 24, 30],
        ttt_mode=True,
        ttt_proj=True,
        ttt_proj_init="diag_shared",
        ttt_lr=0.15,
        ttt_chunk=1024,
        ttt_inner_opt="sgd",
        ttt_norm_preserve=True,
        ttt_target="hidden_states",
        ttt_conv=False,
        ttt_predict_mode="next",
        **kwargs,
    ):
        # Persist Mistral-specific fields so save_pretrained round-trips cleanly
        self.sliding_window = sliding_window
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            tie_word_embeddings=tie_word_embeddings,
            ttt_layers=ttt_layers,
            ttt_mode=ttt_mode,
            ttt_proj=ttt_proj,
            ttt_proj_init=ttt_proj_init,
            ttt_lr=ttt_lr,
            ttt_chunk=ttt_chunk,
            ttt_inner_opt=ttt_inner_opt,
            ttt_norm_preserve=ttt_norm_preserve,
            ttt_target=ttt_target,
            ttt_conv=ttt_conv,
            ttt_predict_mode=ttt_predict_mode,
            **kwargs,
        )


__all__ = ["MistralConfig"]
