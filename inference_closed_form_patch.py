"""Closed-form (ridge) fast-weight write for TTT-NTP inference. See eval_cf_ruler.sh."""
from __future__ import annotations
import argparse, glob, json, os, re, sys, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("TTT_INFERENCE_REPO", _THIS_DIR))
import inference_model  # noqa: F401

from transformers import AutoModelForCausalLM, AutoTokenizer


def model_param_count(model):
    n = 0
    for name, p in model.named_parameters():
        if "embed" in name.lower():
            continue
        n += p.numel()
    return n


def flops_per_token(num_params):
    return 2 * num_params


def estimate_flops(num_params, seq_len, max_new_tokens, n_ttt_layers, d_hidden, d_inter,
                    last_k=None, kv_reuse=False, measure_drift=False):
    """Closed-form patch FLOPS breakdown — supports last_k, kv_reuse, measure_drift.

    Default CF-patch (no kv_reuse): 1 prompt-fwd (capture) + per-layer solve + 1 generate(prompt+decode)
    With kv_reuse:                  1 prompt-fwd (capture+KV) + per-layer solve + decode-only
    With measure_drift:             + 1 extra prompt-fwd (to compute post-patch KV for drift)
    """
    fpt = flops_per_token(num_params)
    vanilla_total = fpt * (seq_len + max_new_tokens)
    prompt_fwd = fpt * seq_len
    decode_only = fpt * max_new_tokens

    T_eff = min(last_k, seq_len) if last_k is not None else seq_len
    if T_eff < d_inter:
        per_layer = 2 * T_eff * T_eff * d_inter + T_eff**3 + 2 * d_hidden * T_eff * d_inter
    else:
        per_layer = 2 * d_inter * d_inter * T_eff + d_inter**3 + 2 * d_inter * T_eff * d_hidden
    solve_flops = per_layer * n_ttt_layers

    if kv_reuse:
        generate_flops = decode_only
    else:
        generate_flops = vanilla_total  # = prompt+decode

    extra_drift = prompt_fwd if measure_drift else 0

    patch_total = prompt_fwd + solve_flops + generate_flops + extra_drift

    return {
        "num_params_nonembed": num_params,
        "seq_len": seq_len,
        "T_eff_for_solve": T_eff,
        "max_new_tokens": max_new_tokens,
        "n_ttt_layers": n_ttt_layers,
        "d_hidden": d_hidden,
        "d_inter": d_inter,
        "kv_reuse": kv_reuse,
        "measure_drift_extra": measure_drift,
        "vanilla_total_flops": vanilla_total,
        "prompt_fwd_flops": prompt_fwd,
        "solve_flops": solve_flops,
        "generate_flops": generate_flops,
        "drift_extra_flops": extra_drift,
        "patch_total_flops": patch_total,
        "flops_ratio_patch_vs_vanilla": patch_total / max(1, vanilla_total),
    }


def _is_niah(task):
    return task.startswith("niah_")


def score_sample(task, pred, refs):
    pred_lower = pred.lower()
    if _is_niah(task) or task == "vt" or task == "cwe" or task == "fwe":
        hits = sum(1 for r in refs if r.lower() in pred_lower)
        return hits / max(1, len(refs))
    elif task.startswith("qa_"):
        return 1.0 if any(r.lower() in pred_lower for r in refs) else 0.0
    return 0.0


def find_ttt_down_proj_modules(model):
    """Return [(layer_idx, down_proj_module)] for the TTT layers in config.ttt_layers."""
    ttt_layers = getattr(model.config, "ttt_layers", [])
    if isinstance(ttt_layers, str):
        ttt_layers = json.loads(ttt_layers)
    n_layers = len(model.model.layers)
    out = []
    for idx in ttt_layers:
        if idx < 0 or idx >= n_layers:
            print(f"[cf-patch] skipping out-of-range ttt_layer idx={idx} (model has {n_layers} layers)")
            continue
        layer = model.model.layers[idx]
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
            out.append((idx, layer.mlp.down_proj))
    return out


def _attach_ttt_proj_weights(model, ckpt_path, ttt_modules):
    """Load ttt_proj.weight for every TTT layer from the checkpoint onto the MLPs."""
    import glob, os
    from safetensors.torch import load_file
    state = {}
    for path in sorted(glob.glob(os.path.join(ckpt_path, "*.safetensors"))):
        for k, v in load_file(path).items():
            if k.endswith(".mlp.ttt_proj.weight"):
                state[k] = v
    for path in sorted(glob.glob(os.path.join(ckpt_path, "*.bin"))):
        for k, v in torch.load(path, map_location="cpu", weights_only=True).items():
            if k.endswith(".mlp.ttt_proj.weight"):
                state[k] = v
    n_attached = 0
    for idx, mod in ttt_modules:
        key = f"model.layers.{idx}.mlp.ttt_proj.weight"
        if key not in state:
            continue
        mlp = model.model.layers[idx].mlp
        d_hidden = mod.weight.shape[0]
        proj = torch.nn.Linear(d_hidden, d_hidden, bias=False).to(device=mod.weight.device, dtype=torch.bfloat16)
        proj.weight.data.copy_(state[key].to(device=mod.weight.device, dtype=torch.bfloat16))
        mlp.ttt_proj = proj
        n_attached += 1
    return n_attached


def _compute_V_target(mlp, t_state_f32):
    """Return W_proj^T t per position; identity fallback if the layer has no projection."""
    proj_mod = getattr(mlp, "ttt_proj", None)
    if proj_mod is not None:
        W_proj = proj_mod.weight.data.to(torch.float32)   # (d_hidden, d_hidden)
        return W_proj.T @ t_state_f32
    return t_state_f32


def compute_and_apply_patch(model, prompt_ids, eta, lam, shift, verbose=False,
                            last_k=None, per_layer_eta=False, eta_target=0.1,
                            kv_reuse=False, norm_preserve=False):
    """Fit and apply the closed-form ridge write per TTT layer; returns (backups, past_kv)."""
    ttt_modules = find_ttt_down_proj_modules(model)
    captures = {}   # layer_idx -> x  (input to MLP, d_hidden)
    handles = []

    def make_hook(idx):
        def hook(module, inputs, output):
            x = inputs[0].detach()        # (B, T, d_hidden)
            captures[idx] = x
        return hook

    for idx, _ in ttt_modules:
        mlp = model.model.layers[idx].mlp
        handles.append(mlp.register_forward_hook(make_hook(idx)))

    base = model.model if hasattr(model, "model") else model
    with torch.no_grad():
        out = base(input_ids=prompt_ids, use_cache=kv_reuse, return_dict=True)
    past_kv = getattr(out, "past_key_values", None) if kv_reuse else None

    for h in handles:
        h.remove()

    backups = {}
    layer_stats = []
    for idx, mod in ttt_modules:
        if idx not in captures:
            continue
        x = captures[idx]                 # (B, T, d_hidden)
        mlp = model.model.layers[idx].mlp
        b, T, _ = x.shape
        d_hidden, d_inter = mod.weight.shape
        if T <= shift + 1:
            continue
        with torch.no_grad():
            h_full = mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x)   # (B, T, d_inter)
        inp_all = h_full[0].T.to(torch.float32)                        # (d_inter, T)
        W = mod.weight.data.to(torch.float32)                          # (d_hidden, d_inter)

        if shift == 0:
            X_full = inp_all
        else:
            X_full = inp_all[:, :-shift]                                # (d_inter, T-shift)

        t_state_f32 = x[0].T.to(torch.float32)                      # (d_hidden, T)
        V_all = _compute_V_target(mlp, t_state_f32)                 # (d_hidden, T)
        Y_full = V_all if shift == 0 else V_all[:, shift:]          # (d_hidden, T-shift)
        del t_state_f32, V_all


        if last_k is not None and X_full.shape[1] > last_k:
            X = X_full[:, -last_k:]
            Y = Y_full[:, -last_k:]
        else:
            X = X_full
            Y = Y_full

        residual = Y - W @ X
        Tt = X.shape[1]

        if Tt < d_inter:
            G = X.T @ X
            G.diagonal().add_(lam)
            alpha = torch.linalg.solve(G, X.T)                          # (T, d_inter)
            delta_W = residual @ alpha                                  # (d_hidden, d_inter)
            del G, alpha
        else:
            G = X @ X.T
            G.diagonal().add_(lam)
            rhs = (residual @ X.T).T                                    # (d_inter, d_hidden)
            sol = torch.linalg.solve(G, rhs)                            # (d_inter, d_hidden)
            delta_W = sol.T                                             # (d_hidden, d_inter)
            del G, rhs, sol

        # Per-layer eta: bound ||eta_l * ΔW_l|| / ||W_l|| ≤ eta_target
        raw_ratio = delta_W.norm().item() / max(1e-9, W.norm().item())
        if per_layer_eta and raw_ratio > 0:
            eta_l = min(eta, eta_target / raw_ratio)
        else:
            eta_l = eta

        if verbose:
            applied_ratio = eta_l * raw_ratio
            res_norm = residual.norm().item()
            y_over_wx = Y.norm().item() / max(1e-9, (W @ X).norm().item())
            layer_stats.append((idx, Tt, eta_l, raw_ratio, applied_ratio, res_norm, y_over_wx))

        backups[idx] = mod.weight.data.clone()
        mod.weight.data.add_((eta_l * delta_W).to(mod.weight.data.dtype))
        if norm_preserve:
            w = mod.weight.data
            init_norm = backups[idx].float().norm(dim=-1, keepdim=True)
            cur_norm = w.float().norm(dim=-1, keepdim=True).clamp_min(1e-5)
            w.copy_((w.float() / cur_norm * init_norm).to(w.dtype))

        del x, h_full, inp_all, W, X_full, Y_full, X, Y, residual, delta_W
        del captures[idx]

    if verbose and layer_stats:
        print(f"    layer-stats (idx, T, eta_used, raw_ratio, applied_ratio, ||resid||, ||Y||/||WX||):")
        for s in layer_stats:
            print(f"      L{s[0]:2d}  T={s[1]:5d}  eta={s[2]:.4f}  raw={s[3]:.3e}  applied={s[4]:.3e}  resid={s[5]:.3e}  Y/WX={s[6]:.3e}")

    return backups, past_kv


def restore_W_down(model, backups):
    for idx, backup in backups.items():
        model.model.layers[idx].mlp.down_proj.weight.data.copy_(backup)
        del backup
    backups.clear()


def patch_and_generate(model, tok, prompt_ids, eta, lam, shift, max_new_tokens,
                       ctx_max=None, verbose=False,
                       last_k=None, per_layer_eta=False, eta_target=0.1,
                       kv_reuse=False, norm_preserve=False):
    if ctx_max is not None and prompt_ids.shape[1] > ctx_max:
        prompt_ids = prompt_ids[:, -ctx_max:]
    backups, past_kv = compute_and_apply_patch(
        model, prompt_ids, eta, lam, shift, verbose,
        last_k=last_k, per_layer_eta=per_layer_eta, eta_target=eta_target,
        kv_reuse=kv_reuse, norm_preserve=norm_preserve,
    )

    with torch.no_grad():
        if kv_reuse and past_kv is not None:
            base_inner = model.model if hasattr(model, "model") else model
            # Use the cached past_kv but advance by feeding the LAST prompt token (this re-uses cache for [:-1])
            # Actually past_kv covers [0..T-1]; to decode next token we need logits at position T-1.
            # HF caches store K,V for ALL T positions, so we need to feed nothing more — use the lm_head on a
            # captured hidden state. But our hook only captured MLP inputs, not lm_head input.
            # Pragmatic fallback: feed prompt_ids[:, -1:] alongside cache truncated to [:-1].
            #
            # Simpler approach: just call model.generate with cache TRUNCATED to T-1 and full prompt_ids.
            # That makes generate do 1 prefill step (for the last token) + N decode steps.
            try:
                if hasattr(past_kv, "crop"):
                    past_kv.crop(prompt_ids.shape[1] - 1)
                elif hasattr(past_kv, "key_cache"):
                    for i in range(len(past_kv.key_cache)):
                        past_kv.key_cache[i] = past_kv.key_cache[i][..., :-1, :]
                        past_kv.value_cache[i] = past_kv.value_cache[i][..., :-1, :]
            except Exception as e:
                print(f"[cf-patch] cache truncate failed: {e}; falling back to no-reuse")
                past_kv = None

            if past_kv is not None:
                gen = model.generate(
                    input_ids=prompt_ids,
                    past_key_values=past_kv,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                    use_cache=True,
                )
                pred = tok.decode(gen[0, prompt_ids.shape[1]:], skip_special_tokens=True)
            else:
                gen = model.generate(
                    input_ids=prompt_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
                pred = tok.decode(gen[0, prompt_ids.shape[1]:], skip_special_tokens=True)
        else:
            gen = model.generate(
                input_ids=prompt_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
            pred = tok.decode(gen[0, prompt_ids.shape[1]:], skip_special_tokens=True)
    restore_W_down(model, backups)
    return pred


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--max_samples", type=int, default=25)
    ap.add_argument("--eta", type=float, default=0.5, help="Step size on closed-form ΔW")
    ap.add_argument("--lam", type=float, default=1.0, help="Ridge regularization")
    ap.add_argument("--shift", type=int, default=1, help="Target offset Y[t]=out[t+shift]")
    ap.add_argument("--ctx_max", type=int, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--device", default="cuda:0",
                    help="Device or 'auto' for multi-GPU model-parallel")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--last_k", type=int, default=None,
                    help="If set, only use last K positions for ridge fit (avoids 32k overshoot)")
    ap.add_argument("--per_layer_eta", action="store_true",
                    help="Auto-scale per-layer eta so ||eta*ΔW||/||W|| ≤ eta_target")
    ap.add_argument("--norm_preserve", action="store_true",
                    help="After the write, renormalise each down_proj row to its original norm "
                         "(same row-norm-preserve the training-time fast weight uses)")
    ap.add_argument("--eta_target", type=float, default=0.1,
                    help="Max per-layer perturbation ratio if --per_layer_eta")
    ap.add_argument("--kv_reuse", action="store_true",
                    help="Reuse KV cache from patch-fwd in generate (skips redundant prompt fwd; saves ~45%% FLOPS)")
    ap.add_argument("--num_shards", type=int, default=1,
                    help="Data-parallel: total number of shards the per-task samples are split into")
    ap.add_argument("--shard_id", type=int, default=0,
                    help="Data-parallel: which shard (0-indexed) this process handles")
    args = ap.parse_args()
    if args.num_shards < 1 or not (0 <= args.shard_id < args.num_shards):
        ap.error(f"invalid sharding: shard_id={args.shard_id} num_shards={args.num_shards}")

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[cf-patch] loading model from {args.ckpt}")
    tok = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)
    if args.device == "auto":
        n_gpu = torch.cuda.device_count()
        max_memory = {i: "6GiB" for i in range(n_gpu)}
        max_memory["cpu"] = "50GiB"
        model = AutoModelForCausalLM.from_pretrained(
            args.ckpt, trust_remote_code=True, dtype=torch.bfloat16,
            device_map="auto", max_memory=max_memory,
        ).eval()
        input_device = next(model.parameters()).device
        print(f"[cf-patch] forced split across {n_gpu} GPUs; input goes to {input_device}")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.ckpt, trust_remote_code=True, dtype=torch.bfloat16,
        ).to(args.device).eval()
        input_device = args.device

    print(f"[cf-patch] model class: {type(model).__name__}")
    ttt_modules = find_ttt_down_proj_modules(model)
    n_trainable = len(ttt_modules)
    print(f"[cf-patch] num TTT down_proj layers: {n_trainable}")
    if n_trainable == 0:
        print("[cf-patch] ERROR: no TTT layers found"); sys.exit(1)

    n_attached = _attach_ttt_proj_weights(model, args.ckpt, ttt_modules)
    if n_attached == 0:
        print("[cf-patch] WARNING: no ttt_proj weights found in ckpt -- target falls back to identity")
    else:
        print(f"[cf-patch] attached W_proj weights on {n_attached}/{n_trainable} TTT layers")

    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "_cf_patch_vanilla_mode"):
        inner._cf_patch_vanilla_mode = True

    n_params = model_param_count(model)
    w0 = ttt_modules[0][1].weight
    d_hidden, d_inter = w0.shape
    print(f"[cf-patch] d_hidden={d_hidden} d_inter={d_inter}")
    print(f"[cf-patch] eta={args.eta}  lam={args.lam}  shift={args.shift}  norm_preserve={args.norm_preserve}")

    print(f"[cf-patch] scanning {args.pred_dir}")
    files_by_task = {}
    for f in glob.glob(os.path.join(args.pred_dir, "ruler_*.json")):
        # Accept both OpenCompass conventions: sharded ("ruler_<task>_<len>_<shard>.json")
        # and non-sharded ("ruler_<task>_<len>.json"). Missing shard defaults to 0.
        m = re.match(r"ruler_(.+?)_(\d+k)(?:_(\d+))?\.json", os.path.basename(f))
        if not m:
            continue
        task, length = m.group(1), m.group(2)
        shard = int(m.group(3)) if m.group(3) else 0
        if args.tasks and task not in args.tasks:
            continue
        files_by_task.setdefault(task, []).append((shard, f))
    for task in files_by_task:
        files_by_task[task].sort()
    print(f"[cf-patch] {len(files_by_task)} task groups: {sorted(files_by_task.keys())}")

    summary = {}
    for task, files in sorted(files_by_task.items()):
        samples = []
        for _, f in files:
            d = json.load(open(f))
            for k, v in d.items():
                if isinstance(v, dict) and "origin_prompt" in v:
                    samples.append((k, v))
        samples = samples[:args.max_samples]
        # Data-parallel sharding: stride-slice so the union over shards is exactly
        # the first max_samples samples, with each shard a disjoint subset.
        if args.num_shards > 1:
            samples = samples[args.shard_id::args.num_shards]

        print(f"\n[cf-patch] === {task}: {len(samples)} samples "
              f"(shard {args.shard_id}/{args.num_shards}) ===")
        results = []
        t0 = time.time()
        for i, (key, sd) in enumerate(samples):
            prompt = sd["origin_prompt"]
            gold = sd.get("gold", [])

            prompt_ids = tok(prompt, return_tensors="pt").input_ids.to(input_device)

            patch_pred = patch_and_generate(
                model, tok, prompt_ids, args.eta, args.lam, args.shift,
                args.max_new_tokens, ctx_max=args.ctx_max,
                verbose=(args.verbose and i == 0),
                last_k=args.last_k, per_layer_eta=args.per_layer_eta,
                eta_target=args.eta_target, kv_reuse=args.kv_reuse,
                norm_preserve=args.norm_preserve,
            )

            p_score = score_sample(task, patch_pred, gold)
            entry = {
                "key": key,
                "gold": gold,
                "patch_pred": patch_pred[:200],
                "patch_score": p_score,
            }
            results.append(entry)
            if (i + 1) % 5 == 0:
                elapsed = time.time() - t0
                pa = sum(r["patch_score"] for r in results) / len(results)
                print(f"  [{i+1}/{len(samples)}] elapsed={elapsed:.0f}s patch_avg={pa:.3f}")

        patch_avg = sum(r["patch_score"] for r in results) / max(1, len(results))
        summary[task] = {
            "n": len(results),
            "patch": patch_avg * 100,
        }
        with open(os.path.join(args.out_dir, f"{task}.json"), "w") as f:
            json.dump({"task": task, "config": vars(args),
                       "patch_avg": patch_avg * 100,
                       "samples": results}, f, indent=2)
        print(f"  ===> {task}: patch={patch_avg*100:.2f}")

    representative_seq_len = None
    for t, files in files_by_task.items():
        if files:
            d = json.load(open(files[0][1]))
            for k, v in d.items():
                if isinstance(v, dict) and "origin_prompt" in v:
                    representative_seq_len = tok(v["origin_prompt"], return_tensors="pt").input_ids.shape[1]
                    break
        if representative_seq_len:
            break
    if representative_seq_len:
        flops_info = estimate_flops(
            n_params, representative_seq_len, args.max_new_tokens,
            n_trainable, d_hidden, d_inter,
            last_k=args.last_k, kv_reuse=args.kv_reuse,
        )
    else:
        flops_info = {"note": "no samples"}

    out = {"config": vars(args), "summary": summary, "flops": flops_info,
           "n_ttt_layers": n_trainable, "n_total_params_nonembed": n_params}
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n========= CLOSED-FORM PATCH SUMMARY =========")
    print(f"{'task':<25s} {'n':>3s} {'patch':>8s}")
    print("-" * 40)
    for t, s in summary.items():
        print(f"{t:<25s} {s['n']:>3d} {s['patch']:>8.2f}")
    print(f"\n========= FLOPS (per prompt @ seq_len={representative_seq_len}) =========")
    if "note" not in flops_info:
        v = flops_info["vanilla_total_flops"]
        p = flops_info["patch_total_flops"]
        ratio = flops_info["flops_ratio_patch_vs_vanilla"]
        print(f"  Vanilla:    {v/1e12:>10.2f} TFLOPS")
        print(f"  CF patch:   {p/1e12:>10.2f} TFLOPS  (prompt_fwd + solves + generate)")
        print(f"    prompt:   {flops_info['prompt_fwd_flops']/1e12:>10.2f} TFLOPS")
        print(f"    solves:   {flops_info['solve_flops']/1e12:>10.2f} TFLOPS")
        print(f"    generate: {flops_info['vanilla_total_flops']/1e12:>10.2f} TFLOPS")
        print(f"  Cost ratio: {ratio:>10.2f}×")


if __name__ == "__main__":
    main()
