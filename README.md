# [EMNLP 2026 Findings] Test-Time Training with Next-Token Prediction

[Xuan Ouyang](https://yancyou.github.io/), [Zefan Cai](https://zefan-cai.github.io/), [Junjie Hu](https://junjiehu.github.io/)

University of Wisconsin–Madison

<p align="left">
  <a href="https://arxiv.org/abs/2606.21803"><img src="https://img.shields.io/badge/arXiv-2606.21803-b31b1b.svg" alt="arXiv"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python"></a>
</p>

## 🎉 Update

- **2026-08** — TTT-NTP is accepted to **EMNLP 2026 (Findings)**. Code for all four backbones is public (see [Code](#-code)).
- **2026-06** — Paper on [arXiv](https://arxiv.org/abs/2606.21803).

## What should a test-time write store?

Next-token prediction is the self-supervised signal that trains language models, and every observed prompt token provides the same signal at test time. In-place test-time training (TTT) makes fast-weight adaptation possible for *pretrained* LLMs without redesigning the backbone — but existing recipes train the fast weight to match a learned local value proxy that is not tied to the next-token signal.

**TTT-NTP** is a drop-in fast-weight adaptation method for pretrained LLMs that supervises each test-time write with **the model's own next contextual hidden state**. Each local write then follows the same causal computation that supports next-token prediction: the value target is a pointwise linear projection of a single next-position contextual state. The released backbone, context window and architecture stay fixed.

## 🔍 Overview

- **Slot** — a fast weight `W` living in the down-projection of selected MLP blocks (every 6th layer for 7–8B models, every 4th for 0.6B).
- **Key** — the block's gated activation `z_t`.
- **Value target** — the same block's next-position contextual hidden state `h_{t+1}`, through a learned linear projection `W_proj`.
- **Write** — a rank-one outer product accumulated as a causal chunk prefix-sum during continual pretraining (chunk 1024), and a single closed-form ridge write applied before decoding at inference (§3.4 of the paper).

## Installation

```bash
git clone https://github.com/yancyou/TTT-NTP.git
cd TTT-NTP
conda create -n tttntp python=3.11 -y && conda activate tttntp

# PyTorch + FlashAttention
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl

# VeOmni (validated commit) + the rest
pip install "veomni @ git+https://github.com/ByteDance-Seed/VeOmni.git@9b91e164bea9e17f17ed490aab5e076c2335ca25"
pip install liger-kernel torchdata blobfile datasets tiktoken timm transformers==4.57.3 opt_einsum einops
pip install opencompass          # RULER harness (evaluation only)
export PYTHONPATH=$(pwd)
```

## 💻 Code

### Quickstart (one node, 8 GPUs)

```bash
# Qwen3-4B (1000 steps)
BASE=/path/to/Qwen3-4B-Base  DATA=/path/to/long-data-collections  OUT=/path/out/qwen3-4b \
  bash scripts/repro_qwen3_4b_ttt.sh

# Qwen3-0.6B (100 steps)
BASE=/path/to/Qwen3-0.6B-Base DATA=... OUT=... bash scripts/repro_qwen3_0p6b_ttt.sh

# Llama-3.1-8B (200 steps)
BASE=/path/to/Llama-3.1-8B DATA=... OUT=... bash scripts/repro_llama_ttt.sh

# Mistral-7B-v0.3 (48 steps)
BASE=/path/to/Mistral-7B-v0.3 DATA=... OUT=... bash scripts/repro_mistral_ttt.sh
```

Each script trains, exports the checkpoint to HuggingFace layout (keeping the trained `ttt_proj` and the `ttt_*` config fields), and runs RULER Full-13 at 4k/8k/16k/32k.

### Training

One config per backbone lives in `configs/pretrain/` (`qwen3_4b`, `qwen3_0p6b`, `llama3_8b`, `mistral_7b`) with the complete release recipe — TTT settings, data and optimizer — so training is a single command:

```bash
bash scripts/train_ttt_ntp.sh qwen3_4b BASE=/path/to/Qwen3-4B-Base DATA=/path/to/long-data-collections OUT=/path/out
# any yaml value can be overridden on the command line, e.g.
bash scripts/train_ttt_ntp.sh qwen3_4b BASE=... DATA=... OUT=... --train.seed 1 --train.micro_batch_size 1
```

The launcher is a thin wrapper around `train.sh tasks/train_torch.py configs/pretrain/<backbone>.yaml`, so the underlying VeOmni command works too. The TTT recipe is the `model.foundation` block of the yaml:

```yaml
foundation:
  ttt_mode: true
  ttt_layers: [0, 6, 12, 18, 24, 30]
  ttt_proj: true
  ttt_proj_init: diag_shared
  ttt_lr: 3.0
  ttt_chunk: 1024
  ttt_inner_opt: sgd
  ttt_norm_preserve: true
  ttt_target: hidden_states
  ttt_conv: false
  ttt_predict_mode: next
```

### Checkpoint conversion

```bash
python scripts/merge_dcp_to_hf.py --load-dir OUT/checkpoints/global_step_1000 \
  --save-dir OUT/checkpoints/global_step_1000/hf_ckpt --model-assets-dir /path/to/base
python scripts/patch_ttt_config.py OUT/checkpoints/global_step_1000/hf_ckpt '<FOUNDATION_JSON>'
```

### Evaluation

RULER evaluation uses the **closed-form inference write** of §3.4 (one ridge solve per adapted layer, applied before decoding, prompt KV-cache reused):

```bash
CF_NORM_PRESERVE=1 CKPT=OUT/checkpoints/global_step_1000/hf_ckpt \
  LENGTHS="4k 8k 16k 32k" GPUS="0,1,2,3,4,5,6,7" NUM_SAMPLES=100 \
  bash eval_cf_ruler.sh
```

Defaults are the paper's Table-7 settings (`eta=0.1`, per-layer cap `0.1`, `lam=1.0`, fit on the last 8192 prefill tokens, row-norm preserved). The summary reports the closed-form (`patch`) scores.

## 📚 Citation

```bibtex
@misc{ouyang2026testtimetrainingnexttokenprediction,
      title={Test-Time Training with Next-Token Prediction}, 
      author={Xuan Ouyang and Zefan Cai and Junjie Hu},
      year={2026},
      eprint={2606.21803},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2606.21803}, 
}
```

## 📝 License

Apache License 2.0 — see [LICENSE](./LICENSE).
