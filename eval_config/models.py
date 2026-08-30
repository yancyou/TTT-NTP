# Model entry consumed by eval_config/ruler_*.py.
# eval_cf_ruler.sh regenerates this file per run with the
# checkpoint under evaluation; edit `_model_configs` only for manual OpenCompass runs.
from opencompass.models import HuggingFaceBaseModel
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

_model_configs = [
    ("ttt-ntp-model", "/path/to/hf_ckpt"),
]

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr=name,
        path=path,
        model_kwargs=dict(trust_remote_code=True),
        tokenizer_kwargs=dict(trust_remote_code=True),
        gen_config=dict(do_sample=False, temperature=0, top_p=0.95, max_new_tokens=64),
        max_seq_len=262144,
        max_out_len=64,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
    for name, path in _model_configs
]
