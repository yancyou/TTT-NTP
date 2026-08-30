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

from .configuration_llama import LlamaConfig
from .modeling_llama import LlamaModel, LlamaForCausalLM
from liger_kernel.transformers.model.llama import lce_forward as llama_lce_forward
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

AutoConfig.register("llama", LlamaConfig, exist_ok=True)
AutoModel.register(LlamaConfig, LlamaModel, exist_ok=True)
AutoModelForCausalLM.register(LlamaConfig, LlamaForCausalLM, exist_ok=True)

LlamaForCausalLM.forward = llama_lce_forward
