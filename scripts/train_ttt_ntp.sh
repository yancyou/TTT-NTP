#!/usr/bin/env bash
# General TTT-NTP training launcher (one node, all visible GPUs).
#
#   bash scripts/train_ttt_ntp.sh <backbone> BASE=/path/to/base DATA=/path/to/ldc OUT=/path/out [extra --key value ...]
#
# <backbone> selects the release config in configs/pretrain/:
#   qwen3_4b | qwen3_0p6b | llama3_8b | mistral_7b
# Every value in the yaml can be overridden on the command line, e.g.
#   --train.max_steps 10 --train.micro_batch_size 1 --train.seed 1
set -euo pipefail
BACKBONE=${1:?usage: train_ttt_ntp.sh <qwen3_4b|qwen3_0p6b|llama3_8b|mistral_7b> BASE=.. DATA=.. OUT=.. [overrides]}
shift
for kv in "$@"; do case "$kv" in BASE=*|DATA=*|OUT=*) export "$kv"; shift || true;; *) break;; esac; done
: "${BASE:?BASE=/path/to/base model}"; : "${DATA:?DATA=/path/to/hf-long-data-collections-sharded}"; : "${OUT:?OUT=/path/to/output}"
PROJ=$(cd "$(dirname "$0")/.." && pwd)
CFG="$PROJ/configs/pretrain/${BACKBONE}.yaml"
[ -f "$CFG" ] || { echo "unknown backbone '$BACKBONE' (no $CFG)" >&2; exit 1; }
export PYTHONPATH="$PROJ"
cd "$PROJ"
mkdir -p "$OUT"
bash train.sh tasks/train_torch.py "$CFG" \
  --model.model_path "$BASE" \
  --data.train_path "$DATA" \
  --train.output_dir "$OUT" --train.save_checkpoint_path "$OUT" \
  "$@"
