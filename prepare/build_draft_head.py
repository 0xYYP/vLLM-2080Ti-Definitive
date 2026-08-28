#!/usr/bin/env python3
"""Build the vocab-truncated MTP draft head assets for Qwen3.5/3.8.

Slices the (dense) lm_head rows listed in draft_vocab_ids.json into
``mtp.draft_lm_head.weight`` inside a new ``model_extra_tensors.safetensors``
shard, plus the id map as ``mtp_draft_vocab_ids.pt``. The engine patch in
vllm/model_executor/models/qwen3_5_mtp.py picks both up at model load when
``MTP_DRAFT_VOCAB != 0``. Nothing in the original checkpoint is modified;
delete the two outputs to revert.

Usage:
    venv/bin/python prepare/build_draft_head.py --model DIR [--ids JSON]

Prereqs: ``dir`` contains config.json and model.safetensors(.index.json);
``ids`` defaults to ``dir/mtp_draft_vocab_ids.json`` (the 40k list shipped
by syv-ai/qwen38-27b-rtx3090, Apache-2.0).
"""
import argparse
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model directory")
    ap.add_argument("--ids", default=None, help="draft_vocab_ids.json path")
    ap.add_argument("--max-rows", type=int, default=40960)
    args = ap.parse_args()

    d = os.path.abspath(args.model)
    ids_json = args.ids or os.path.join(d, "mtp_draft_vocab_ids.json")
    cfg = json.load(open(os.path.join(d, "config.json"), encoding="utf-8"))
    vocab_size = cfg.get("vocab_size")
    if vocab_size is None:
        vocab_size = cfg.get("text_config", {}).get("vocab_size")
    if vocab_size is None:
        raise SystemExit("config.json has no vocab_size at top level or text_config")

    # locate lm_head shard
    index_path = os.path.join(d, "model.safetensors.index.json")
    if os.path.exists(index_path):
        wm = json.load(open(index_path, encoding="utf-8"))["weight_map"]
        head_file = wm["lm_head.weight"]
    else:
        head_file = "model.safetensors"
    if not os.path.isabs(head_file):
        head_file = os.path.join(d, head_file)

    ids = torch.tensor(
        json.load(open(ids_json, encoding="utf-8")), dtype=torch.long
    )[: args.max_rows]
    if int(ids.max()) >= int(vocab_size):
        raise SystemExit(f"id {int(ids.max())} out of range for vocab_size {vocab_size}")
    print(f"vocab_size={vocab_size} ids={ids.numel()} max_id={int(ids.max())}", flush=True)

    with safe_open(head_file, framework="pt") as f:
        lm = f.get_tensor("lm_head.weight")
    print(f"lm_head {tuple(lm.shape)} {lm.dtype}", flush=True)
    sub = lm.index_select(0, ids).contiguous()
    del lm

    extra = os.path.join(d, "model_extra_tensors.safetensors")
    save_file({"mtp.draft_lm_head.weight": sub}, extra)
    torch.save(ids, os.path.join(d, "mtp_draft_vocab_ids.pt"))
    print(f"wrote {extra} [{tuple(sub.shape)}] and mtp_draft_vocab_ids.pt", flush=True)
    # round-trip check
    with safe_open(extra, framework="pt") as f:
        back = f.get_tensor("mtp.draft_lm_head.weight")
    assert back.shape == sub.shape and torch.equal(back, sub)
    print("round-trip check OK", flush=True)


if __name__ == "__main__":
    main()