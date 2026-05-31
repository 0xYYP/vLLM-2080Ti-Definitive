#!/usr/bin/env python3
import json
import sys
from pathlib import Path

for item in sys.argv[1:]:
    root = Path(item)
    cfg_path = root / "config.json"
    print(f"== {root} ==")
    if not cfg_path.exists():
        print("missing config.json")
        continue
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    for key in [
        "architectures",
        "model_type",
        "torch_dtype",
        "quantization_config",
        "max_position_embeddings",
        "sliding_window",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "num_nextn_predict_layers",
        "mtp_depth",
    ]:
        if key in cfg:
            print(f"{key}: {cfg[key]}")
    mtp_keys = [k for k in cfg if "mtp" in k.lower() or "next" in k.lower()]
    print(f"mtp_like_keys: {mtp_keys[:30]}")
    files = sorted(p.name for p in root.iterdir() if p.is_file())
    print(f"safetensors: {sum(1 for f in files if f.endswith('.safetensors'))}")
    print(f"has_mtp_files: {[f for f in files if 'mtp' in f.lower() or 'next' in f.lower()][:20]}")
