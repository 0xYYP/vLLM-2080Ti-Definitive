#!/usr/bin/env python3
"""Build the draft-vocab small head assets for Qwen3.8 MTP (v2, 2026-09-16).

Slices the (bf16) lm_head rows listed in draft_vocab_ids.json into
``draft_lm_head.weight`` inside ``draft_vocab_head.safetensors`` (the engine
loads it as the Qwen3_5MTP root module ``draft_lm_head`` via
AutoWeightsLoader; the ``draft_id_to_target_id`` offset is rebuilt by the
engine at init from draft_vocab_ids.json, not from the checkpoint). The
engine picks both up when ``MTP_DRAFT_VOCAB=1`` is exported in the launch
script; the proposer must run with ``use_local_argmax_reduction: true``.

The OLD 2026-08-30 design (write-back into model_extra_tensors.safetensors as
``mtp.draft_lm_head`` + mtp_draft_vocab_ids.pt) is superseded: the 0.1-stack
engine patch is gone in the 0.2 stack, so the stored 16k table there is
stale. Nothing in the original checkpoints is modified; delete
draft_vocab_head.safetensors (and keep the ids json) to revert.

Usage:
    venv/bin/python prepare/build_draft_head.py --model DIR [--ids JSON]

Prereqs: ``dir`` contains config.json and a shard physically holding
``lm_head.weight`` (bf16).
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
    ids_json = args.ids or os.path.join(d, "draft_vocab_ids.json")
    cfg = json.load(open(os.path.join(d, "config.json"), encoding="utf-8"))
    vocab_size = cfg.get("vocab_size")
    if vocab_size is None:
        vocab_size = cfg.get("text_config", {}).get("vocab_size")
    if vocab_size is None:
        raise SystemExit("config.json has no vocab_size")

    index_path = os.path.join(d, "model.safetensors.index.json")
    if os.path.exists(index_path):
        wm = json.load(open(index_path, encoding="utf-8"))["weight_map"]
        head_file = wm.get("lm_head.weight", "model.safetensors")
    else:
        head_file = "model.safetensors"
    if not os.path.isabs(head_file):
        head_file = os.path.join(d, head_file)

    ids = json.load(open(ids_json, encoding="utf-8"))
    assert ids == sorted(ids), "draft_vocab_ids.json must be sorted ascending"
    ids = torch.tensor(ids, dtype=torch.long)[: args.max_rows]
    if int(ids.max()) >= int(vocab_size):
        raise SystemExit(f"id {int(ids.max())} out of range for vocab_size {vocab_size}")
    print(f"vocab_size={vocab_size} ids={ids.numel()}", flush=True)

    with safe_open(head_file, framework="pt") as f:
        lm = f.get_tensor("lm_head.weight")
    print(f"lm_head {tuple(lm.shape)} {lm.dtype}", flush=True)
    sub = lm.index_select(0, ids).contiguous()
    del lm

    extra = os.path.join(d, "draft_vocab_head.safetensors")
    save_file({"draft_lm_head.weight": sub}, extra)
    print(f"wrote {extra} [{tuple(sub.shape)}]", flush=True)
    with safe_open(extra, framework="pt") as f:
        back = f.get_tensor("draft_lm_head.weight")
    assert back.shape == sub.shape and torch.equal(back, sub)
    print("round-trip check OK", flush=True)


if __name__ == "__main__":
    main()
