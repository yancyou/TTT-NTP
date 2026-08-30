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

from mmengine.config import read_base

from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

with read_base():
    from opencompass.configs.datasets.ruler.ruler_cwe_gen import cwe_datasets  # CWE
    from opencompass.configs.datasets.ruler.ruler_fwe_gen import fwe_datasets  # FWE
    from opencompass.configs.datasets.ruler.ruler_niah_gen import niah_datasets  # Niah
    from opencompass.configs.datasets.ruler.ruler_qa_gen import qa_datasets  # QA
    from opencompass.configs.datasets.ruler.ruler_vt_gen import vt_datasets  # VT
    from .models import models as qwen3_models
    from opencompass.configs.summarizers.groups.ruler import ruler_summary_groups

import_datasets = sum(
    [qa_datasets, niah_datasets, vt_datasets, fwe_datasets, cwe_datasets], []
)

# Evaluation config
NUM_SAMPLES = 100
# Change the context lengths to be tested
max_seq_lens = [32 * 1024]
abbr_suffixs = ["32k"]
work_dir = "./results/ruler/32k"

# Model Settings
model_settings = [
    [qwen3_models[i], qwen3_models[i]["path"]] for i in range(len(qwen3_models))
]

# Dataset Model Combination
datasets = []
models = []
model_dataset_combinations = []

# Different seq length
for max_seq_len, abbr_suffix in zip(max_seq_lens, abbr_suffixs):
    for model, model_path in model_settings:
        _tmp_datasets = []
        for dataset in import_datasets:
            tmp_dataset = dataset.deepcopy()
            tmp_dataset["tokenizer_model"] = model_path
            tmp_dataset["abbr"] = tmp_dataset["abbr"] + "_" + abbr_suffix
            tmp_dataset["num_samples"] = NUM_SAMPLES
            tmp_dataset["max_seq_length"] = max_seq_len
            _tmp_datasets.append(tmp_dataset)
        model_dataset_combinations.append(dict(models=[model], datasets=_tmp_datasets))
        models.append(model)
        datasets.extend(_tmp_datasets)

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner),
    runner=dict(
        type=LocalRunner, max_num_workers=16, task=dict(type=OpenICLInferTask), retry=5
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(type=LocalRunner, max_num_workers=32, task=dict(type=OpenICLEvalTask)),
)

summarizer = dict(
    dataset_abbrs=abbr_suffixs,
    summary_groups=sum([ruler_summary_groups], []),
)
