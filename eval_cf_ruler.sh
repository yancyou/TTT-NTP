#!/usr/bin/env bash
# Closed-form (CF) RULER Full-13 evaluation for TTT-NTP.
#
# Per context length: (1) OpenCompass materializes the RULER prompts and the
# model's own generations, sharded over $GPUS; (2) one CF process per GPU fits
# the per-layer ridge write  Y_t = W_proj^T h_{t+1}  on the cached gated-MLP
# activations, applies it to each adapted down-projection, decodes with the
# prompt KV cache reused, restores the weight, and the shards are merged.
#
# Usage:
#   CKPT=/path/to/hf_ckpt bash eval_cf_ruler.sh
#
# Env vars [defaults]: LENGTHS [4k 8k 16k 32k] | GPUS [0,1,2,3,4,5,6,7] |
#   NUM_SAMPLES [100] | MAX_NEW_TOKENS [64] | MODEL_ABBR | OUT_DIR | PRED_ROOT |
#   SKIP_GEN [0] | BATCH_<len> | CF_ETA [0.1] | CF_ETA_TARGET [0.1] |
#   CF_LAM [1.0] | CF_SHIFT [1] | CF_LAST_K [8192] | CF_NORM_PRESERVE [1] |
#   CF_VERBOSE [1]
# $CKPT must keep the ttt_* fields in config.json and the trained ttt_proj
# weights (scripts/merge_dcp_to_hf.py + scripts/patch_ttt_config.py produce this).
set -euo pipefail

: "${CKPT:?CKPT is required (absolute path to HF checkpoint dir)}"
: "${LENGTHS:=4k 8k 16k 32k}"
: "${GPUS:=0,1,2,3,4,5,6,7}"
: "${NUM_SAMPLES:=100}"
: "${MAX_SAMPLES:=$NUM_SAMPLES}"
: "${MAX_NEW_TOKENS:=64}"
: "${SKIP_GEN:=0}"

: "${CF_ETA:=0.1}"
: "${CF_ETA_TARGET:=0.1}"
: "${CF_LAM:=1.0}"
: "${CF_SHIFT:=1}"
: "${CF_LAST_K:=8192}"
: "${CF_NORM_PRESERVE:=1}"   # 1 -> row-renormalise down_proj to ||W_0||_row after the write (release default)
: "${CF_VERBOSE:=1}"         # 1 -> per-layer write stats (raw_ratio, ||Y||/||WX||) in the shard logs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${PROJECT_DIR:=$SCRIPT_DIR}"

IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
NUM_GPUS=${#GPU_ARR[@]}
if [ "$NUM_GPUS" -lt 1 ]; then
  echo "ERROR: GPUS must list at least one GPU id" >&2
  exit 1
fi

if [ ! -f "${CKPT}/config.json" ]; then
  echo "ERROR: no config.json at ${CKPT}" >&2
  exit 1
fi

# Default model tag: prefer the run-name two levels up (…/<run>/checkpoints/global_step_N/hf_ckpt).
if [ -z "${MODEL_ABBR:-}" ]; then
  MODEL_ABBR="$(basename "$(cd "${CKPT}/../../.." 2>/dev/null && pwd || echo cf-model)")"
  [ -z "$MODEL_ABBR" ] && MODEL_ABBR="cf-model"
fi
: "${OUT_DIR:=${PROJECT_DIR}/results/cf_ruler/${MODEL_ABBR}}"

# Validate that ttt_layers survived in config.json (required to locate adapted layers).
HAS_TTT=$(python3 -c "import json; c=json.load(open('${CKPT}/config.json')); print(1 if c.get('ttt_layers') else 0)")
if [ "$HAS_TTT" != "1" ]; then
  echo "ERROR: config.json has no 'ttt_layers' field — the closed-form patch cannot" >&2
  echo "       locate the adapted MLP down-projections. Re-export the checkpoint" >&2
  echo "       WITHOUT stripping ttt_* fields, or add ttt_layers back to config.json." >&2
  exit 1
fi

# Warn about lengths exceeding the model's position window.
MAX_POS=$(python3 -c "import json;print(json.load(open('${CKPT}/config.json')).get('max_position_embeddings',0))")
for len in $LENGTHS; do
  needed=$(( ${len%k} * 1024 ))
  if [ "$needed" -gt "$MAX_POS" ]; then
    echo "[WARN] length=${len} needs ${needed} positions but model max_position_embeddings=${MAX_POS}; scores may be near-random without rope scaling." >&2
  fi
done

TS=$(date +%s)
: "${PRED_ROOT:=/tmp/cf_ruler_${MODEL_ABBR}_${TS}}"
CFG_DIR="${PRED_ROOT}/_cfg"
mkdir -p "$CFG_DIR" "$OUT_DIR"
touch "$CFG_DIR/__init__.py"

echo "=============================================================="
echo "CF-RULER  model=${MODEL_ABBR}"
echo "  ckpt     = ${CKPT}"
echo "  lengths  = ${LENGTHS}   (serial)"
echo "  gpus     = ${GPUS}   (data parallel, ${NUM_GPUS} cards per length)"
echo "  samples  = ${NUM_SAMPLES} (score <= ${MAX_SAMPLES})"
echo "  out_dir  = ${OUT_DIR}"
echo "  CF: eta=${CF_ETA} eta_target=${CF_ETA_TARGET} lam=${CF_LAM} shift=${CF_SHIFT} last_k=${CF_LAST_K} norm_preserve=${CF_NORM_PRESERVE}"
echo "=============================================================="

export PYTHONPATH="${PROJECT_DIR}:${CFG_DIR}:${PYTHONPATH:-}"
export COMPASS_DATA_CACHE="${COMPASS_DATA_CACHE:-/tmp/opencompass-data}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$COMPASS_DATA_CACHE"

# Per-length batch sizes for Phase 1 (configurable via env). Defaults sized for 0.6B/bf16 on 80GB A100.
: "${BATCH_4k:=32}"
: "${BATCH_8k:=16}"
: "${BATCH_16k:=8}"
: "${BATCH_32k:=4}"

# models.py is regenerated PER LENGTH below with the appropriate batch_size.
# (the shared file written here is only a placeholder; eval_config/ruler_${len}.py
#  imports `models` from this file, and we sed-rewrite batch_size per length.)
true

# ----------------------------------------------------------------------------
# Per length (serial): Phase 1 generation (8-GPU) + Phase 2 closed-form (8-GPU).
# ----------------------------------------------------------------------------
for len in $LENGTHS; do
  echo ""
  echo "############## length=${len} ##############"

  # ---- Phase 1: materialize RULER prompts + gold via OpenCompass (all GPUs) ----
  if [ "$SKIP_GEN" != "1" ]; then
    src="${PROJECT_DIR}/eval_config/ruler_${len}.py"
    if [ ! -f "$src" ]; then
      echo "ERROR: missing eval_config/ruler_${len}.py" >&2
      exit 1
    fi
    dst="$CFG_DIR/ruler_${len}.py"
    cp "$src" "$dst"
    # Pick per-length batch size and write fresh models.py
    case "$len" in
      4k)  BS="${BATCH_4k}" ;;
      8k)  BS="${BATCH_8k}" ;;
      16k) BS="${BATCH_16k}" ;;
      32k) BS="${BATCH_32k}" ;;
      *)   BS=1 ;;
    esac
    echo "[phase1] using batch_size=${BS} for ${len}"
    cat > "$CFG_DIR/models.py" <<PY
from opencompass.models import HuggingFaceBaseModel
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

_model_configs = [
    ("${MODEL_ABBR}", "${CKPT}"),
]

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr=name,
        path=path,
        model_kwargs=dict(trust_remote_code=True),
        tokenizer_kwargs=dict(trust_remote_code=True),
        gen_config=dict(
            do_sample=False,
            temperature=0,
            top_p=0.95,
            max_new_tokens=${MAX_NEW_TOKENS},
        ),
        max_seq_len=262144,
        max_out_len=${MAX_NEW_TOKENS},
        batch_size=${BS},
        run_cfg=dict(num_gpus=1),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
    for name, path in _model_configs
]
PY
    # Redirect work_dir, set sample count, and fan the 13 subtasks out across
    # all GPUs by giving the partitioner num_worker = NUM_GPUS.
    sed -i \
      -e "s|work_dir = f\"\./results/ruler/${len}\"|work_dir = \"${PRED_ROOT}/${len}\"|" \
      -e "s|work_dir = \"\./results/ruler/${len}\"|work_dir = \"${PRED_ROOT}/${len}\"|" \
      -e "s|^NUM_SAMPLES = [0-9]\+|NUM_SAMPLES = ${NUM_SAMPLES}|" \
      -e "s|partitioner=dict(type=NumWorkerPartitioner)|partitioner=dict(type=NumWorkerPartitioner, num_worker=${NUM_GPUS})|" \
      "$dst"

    echo "[phase1] generating RULER prompts for ${len} on GPUs ${GPUS}"
    CUDA_VISIBLE_DEVICES=${GPUS} python -c "
import inference_model  # noqa: F401  (registers the Qwen3 TTT modeling)
from opencompass.cli.main import main
import sys
sys.argv = ['opencompass', '${dst}']
main()
"
  fi

  # Locate the predictions dir: <PRED_ROOT>/<len>/<timestamp>/predictions/<MODEL_ABBR>/
  pred_dir="$(ls -dt "${PRED_ROOT}/${len}"/*/predictions/"${MODEL_ABBR}" 2>/dev/null | head -1 || true)"
  if [ -z "$pred_dir" ] || [ ! -d "$pred_dir" ]; then
    echo "[WARN] no predictions dir for ${len} under ${PRED_ROOT}/${len}; skipping." >&2
    continue
  fi

  # ---- Phase 2: closed-form patch, data-parallel over shards ----
  # SHARDS_PER_GPU lets us pack multiple cf-patch processes on one GPU when
  # the model is small (0.6B fits ~4x per 80GB card).
  : "${SHARDS_PER_GPU:=1}"
  TOTAL_SHARDS=$(( NUM_GPUS * SHARDS_PER_GPU ))
  shard_root="${OUT_DIR}/${len}/shards"
  mkdir -p "$shard_root"
  echo "[phase2] closed-form patch on ${len}: ${pred_dir} (${TOTAL_SHARDS} shards = ${NUM_GPUS} GPUs × ${SHARDS_PER_GPU} per GPU)"
  declare -a PIDS=()
  for sid in $(seq 0 $((TOTAL_SHARDS - 1))); do
    gpu="${GPU_ARR[$(( sid % NUM_GPUS ))]}"
    shard_out="${shard_root}/shard_${sid}"
    logf="${shard_root}/shard_${sid}.log"
    extra_flags=""
    [ "$CF_NORM_PRESERVE" = "1" ] && extra_flags="$extra_flags --norm_preserve"
    [ "$CF_VERBOSE" = "1" ] && extra_flags="$extra_flags --verbose"
    CUDA_VISIBLE_DEVICES=${gpu} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      python "${PROJECT_DIR}/inference_closed_form_patch.py" \
      --ckpt "${CKPT}" \
      --pred_dir "${pred_dir}" \
      --out_dir "${shard_out}" \
      --max_samples "${MAX_SAMPLES}" \
      --max_new_tokens "${MAX_NEW_TOKENS}" \
      --eta "${CF_ETA}" \
      --per_layer_eta --eta_target "${CF_ETA_TARGET}" \
      --lam "${CF_LAM}" \
      --shift "${CF_SHIFT}" \
      --last_k "${CF_LAST_K}" \
      --kv_reuse \
      --device cuda:0 \
      $extra_flags \
      --num_shards "${TOTAL_SHARDS}" --shard_id "${sid}" > "${logf}" 2>&1 &
    PIDS+=($!)
  done
  fail=0
  for i in "${!PIDS[@]}"; do
    sid=$i
    gpu="${GPU_ARR[$(( sid % NUM_GPUS ))]}"
    if wait "${PIDS[$i]}"; then
      echo "  [shard ${sid}] done (GPU ${gpu})"
    else
      echo "  [shard ${sid}] FAILED (GPU ${gpu}) — see ${shard_root}/shard_${sid}.log" >&2
      fail=1
    fi
  done
  unset PIDS
  if [ "$fail" != "0" ]; then
    echo "[ERROR] one or more shards failed for ${len}; not merging." >&2
    continue
  fi

  # ---- Merge shards: concat per-task samples, recompute averages ----
  python3 - "${OUT_DIR}/${len}" "${shard_root}" <<'PY'
import glob, json, os, sys
len_dir, shard_root = sys.argv[1], sys.argv[2]

# Gather per-task sample lists across all shard subdirs.
per_task = {}          # task -> list of sample dicts
for tjson in glob.glob(os.path.join(shard_root, "shard_*", "*.json")):
    base = os.path.basename(tjson)
    if base == "summary.json":
        continue
    d = json.load(open(tjson))
    task = d.get("task") or base[:-5]
    per_task.setdefault(task, []).extend(d.get("samples", []))

summary = {}
for task, samples in per_task.items():
    if not samples:
        continue
    n = len(samples)
    p = sum(s["patch_score"] for s in samples) / n * 100
    summary[task] = {"n": n, "patch": p}
    with open(os.path.join(len_dir, f"{task}.json"), "w") as f:
        json.dump({"task": task, "n": n, "patch_avg": p, "samples": samples}, f, indent=2)

with open(os.path.join(len_dir, "summary.json"), "w") as f:
    json.dump({"summary": summary, "n_tasks": len(summary),
               "merged_from_shards": True}, f, indent=2)
print(f"  merged {len(summary)} tasks -> {os.path.join(len_dir, 'summary.json')}")
PY
done

# Final: aggregate the closed-form RULER Full-13 scores across lengths.
python3 - "$OUT_DIR" $LENGTHS <<'PY'
import json, os, sys

out_dir = sys.argv[1]
lengths = sys.argv[2:]

# RULER Full-13 task set (task keys as emitted by the CF patch).
FULL13 = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multiquery", "niah_multivalue",
    "vt", "fwe", "cwe",
    "qa_squad", "qa_hotpotqa",
]

agg = {"out_dir": out_dir, "lengths": {}, "full13_avg": {}}
print("\n========= CLOSED-FORM RULER Full-13 =========")
print(f"{'length':>8s} {'n_tasks':>8s} {'patch':>9s}")
print("-" * 30)

patch_means = []
for length in lengths:
    sp = os.path.join(out_dir, length, "summary.json")
    if not os.path.isfile(sp):
        print(f"{length:>8s}   (no summary.json — skipped)")
        continue
    summ = json.load(open(sp)).get("summary", {})
    patch, present = [], []
    for task in FULL13:
        if task in summ:
            patch.append(summ[task]["patch"])
            present.append(task)
    if not present:
        print(f"{length:>8s}   (no Full-13 tasks found — skipped)")
        continue
    p = sum(patch) / len(patch)
    agg["lengths"][length] = {
        "n_tasks": len(present), "patch": p,
        "per_task": {t: summ[t] for t in present},
        "missing": [t for t in FULL13 if t not in summ],
    }
    patch_means.append(p)
    print(f"{length:>8s} {len(present):>8d} {p:>9.2f}")

if patch_means:
    full13_avg = sum(patch_means) / len(patch_means)
    agg["full13_avg"]["patch"] = full13_avg
    print("-" * 30)
    print(f"{'AVG':>8s} {'':>8s} {full13_avg:>9.2f}")

with open(os.path.join(out_dir, "ruler_cf_summary.json"), "w") as f:
    json.dump(agg, f, indent=2)
print(f"\nwrote {os.path.join(out_dir, 'ruler_cf_summary.json')}")
PY

echo "[done] closed-form RULER results under ${OUT_DIR}"
