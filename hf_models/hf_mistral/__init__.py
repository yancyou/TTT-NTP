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

"""TTT-enabled Mistral support.

Mistral has the same state_dict structure as Llama (gate/up/down MLP,
q/k/v/o attention, RMSNorm, RoPE). We register MistralConfig → our
TTT-enabled Llama modeling so AutoModel routes Mistral checkpoints
through the same code path.
"""

from .configuration_mistral import MistralConfig
from .modeling_mistral import MistralModel, MistralForCausalLM
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

AutoConfig.register("mistral", MistralConfig, exist_ok=True)
AutoModel.register(MistralConfig, MistralModel, exist_ok=True)
AutoModelForCausalLM.register(MistralConfig, MistralForCausalLM, exist_ok=True)
