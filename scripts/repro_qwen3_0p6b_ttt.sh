#!/usr/bin/env bash
# TTT-NTP on Qwen3-0.6B-Base: train (configs/pretrain/qwen3_0p6b.yaml, 100 steps), export the
# checkpoint to HuggingFace layout, and run the closed-form RULER Full-13 evaluation.
#
# Usage: BASE=/path/to/Qwen3-0.6B-Base DATA=/path/to/long-data-collections OUT=/path/out \
#        bash scripts/repro_qwen3_0p6b_ttt.sh
set -euo pipefail

BASE=${BASE:?path to Qwen3-0.6B-Base checkpoint}
DATA=${DATA:?path to hf-long-data-collections-sharded (arrow shards)}
OUT=${OUT:?output root}
STEPS=${STEPS:-100}
SEED=${SEED:-42}
PROJ=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$PROJ"
cd "$PROJ"

F='{"ttt_layers":[0,4,8,12,16,20,24],"ttt_mode":true,"ttt_proj":true,"ttt_proj_init":"diag_shared","ttt_conv":false,"ttt_predict_mode":"next","ttt_target":"hidden_states","ttt_chunk":1024,"ttt_norm_preserve":true,"ttt_inner_opt":"sgd","ttt_lr":2.6}'

# ---- 1. train ----
mkdir -p "$OUT"
bash train.sh tasks/train_torch.py configs/pretrain/qwen3_0p6b.yaml \
  --model.model_path "$BASE" --model.foundation "$F" \
  --data.train_path "$DATA" --data.text_keys text --data.max_seq_len 32768 \
  --train.output_dir "$OUT" --train.save_checkpoint_path "$OUT" \
  --train.max_steps "$STEPS" --train.save_steps "$STEPS" --train.save_epochs 0 \
  --train.global_batch_size 64 --train.micro_batch_size 8 \
  --train.lr 5.0e-6 --train.lr_warmup_ratio 0.05 --train.seed "$SEED" \
  --train.save_hf_weights true --train.save_async false --train.use_wandb false

# ---- 2. export to HF layout ----
D="$OUT/checkpoints/global_step_$STEPS"
python scripts/merge_dcp_to_hf.py --load-dir "$D" --save-dir "$D/hf_ckpt" \
  --model-assets-dir "$BASE"
cp "$BASE/generation_config.json" "$D/hf_ckpt/" 2>/dev/null || true
python scripts/patch_ttt_config.py "$D/hf_ckpt" "$F"

# CF_NORM_PRESERVE=1 applies the row-norm-preserving closed-form write (see eval_cf_ruler.sh).
CF_NORM_PRESERVE=1 CKPT="$D/hf_ckpt" MODEL_ABBR=qwen3-0p6b-ttt OUT_DIR="$OUT/results" \
  LENGTHS="4k 8k 16k 32k" NUM_SAMPLES=100 \
  BATCH_4k=32 BATCH_8k=16 BATCH_16k=8 BATCH_32k=4 \
  bash eval_cf_ruler.sh

echo "done"
