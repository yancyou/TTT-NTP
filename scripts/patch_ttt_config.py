#!/usr/bin/env python3
"""Inject ttt_* fields from a training FOUNDATION json into a merged hf_ckpt
config.json, so the closed-form CF eval path (which requires ttt_layers etc.)
works. merge_dcp_to_hf.py copies the BASE model config which has no ttt_* fields.

Usage: patch_ttt_config.py <hf_ckpt_dir> '<foundation_json>'
"""
import json, sys, pathlib

hf_dir = pathlib.Path(sys.argv[1])
found = json.loads(sys.argv[2])
cfg_path = hf_dir / "config.json"
cfg = json.loads(cfg_path.read_text())

ttt = {k: v for k, v in found.items() if k.startswith("ttt")}
cfg.update(ttt)
cfg_path.write_text(json.dumps(cfg, indent=2))
print("patched ttt keys into %s: %s" % (cfg_path, sorted(ttt)))
