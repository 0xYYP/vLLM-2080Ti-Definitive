#!/usr/bin/env python3
"""Build a draft-vocab id list from a text corpus (model's own outputs etc).

Counts token frequencies over corpus files (.jsonl/.txt), picks the top N ids
plus special tokens, writes the ascending id list to
``<model_dir>/draft_vocab_ids.json``. Every 10th sample is held out for a
coverage estimate (>= 90% is the target). Ported from
syv-ai/qwen38-27b-rtx3090 (Apache-2.0), adapted to read the JSONL emitted by
prepare/sample_model_outputs.py.

Usage:
    venv/bin/python prepare/build_draft_vocab.py --model DIR --n 40960 \
        --corpus corpus.jsonl corpus2.jsonl
    # or reuse a shipped id list (skips counting):
    venv/bin/python prepare/build_draft_vocab.py --model DIR --ids ids.json
"""
import argparse
import collections
import json
import os

from transformers import AutoTokenizer

SPECIAL_NAMES = (
    "<|im_start|>", "<|im_end|>", "<|endoftext|>",
    " thinking", " response", "<tool_call>", "</tool_call>",
    "<tool_response>", "</tool_response>",
)


def texts_from(path, limit_bytes=200_000_000):
    n = 0
    if path.endswith(".jsonl"):
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            for k in ("prompt", "response", "output", "text"):
                if isinstance(r.get(k), str):
                    yield r[k]
                    n += len(r[k])
                    if n > limit_bytes:
                        return
            if isinstance(r.get("messages"), list):
                t = "\n".join(m.get("content", "") for m in r["messages"]
                              if isinstance(m.get("content"), str))
                yield t
                n += len(t)
                if n > limit_bytes:
                    return
    else:
        yield open(path, errors="ignore").read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model directory (tokenizer)")
    ap.add_argument("--n", type=int, default=40960)
    ap.add_argument("--corpus", nargs="*", default=[])
    ap.add_argument("--ids", default=None, help="reuse a shipped id list")
    ap.add_argument("--out", default=None, help="output json path (default: model dir)")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    counts = collections.Counter()
    held = collections.Counter()
    total = 0

    if args.ids:
        ids = sorted(set(json.load(open(args.ids))))
        print(f"using {len(ids)} ids from {args.ids}")
    else:
        for i, path in enumerate(args.corpus):
            for j, t in enumerate(texts_from(path)):
                toks = tok(t, add_special_tokens=False).input_ids
                (held if j % 10 == 0 else counts).update(toks)
                total += len(toks)
        print(f"corpus tokens: {total}")

    special = set(tok.all_special_ids)
    for name in SPECIAL_NAMES:
        tid = tok.convert_tokens_to_ids(name)
        if isinstance(tid, int) and tid >= 0:
            special.add(tid)

    if not args.ids:
        top = [t for t, _ in counts.most_common() if t not in special][: args.n - len(special)]
        ids = sorted(set(top) | special)
        cover = sum(c for t, c in held.items() if t in set(ids)) / max(1, sum(held.values()))
        print(f"draft vocab: {len(ids)} ids, held-out token coverage {cover*100:.2f}%")
        for n_try in (8192, 16384, 32768, 49152, 65536):
            s = set(t for t, _ in counts.most_common(n_try)) | special
            c = sum(c for t, c in held.items() if t in s) / max(1, sum(held.values()))
            print(f"  coverage at N={n_try}: {c*100:.2f}%")
    else:
        cover = (sum(c for t, c in held.items() if t in set(ids)) / max(1, sum(held.values()))
                 if held else None)
        if cover is not None:
            print(f"held-out coverage of shipped list: {cover*100:.2f}%")

    out = args.out or os.path.join(args.model, "draft_vocab_ids.json")
    json.dump(ids, open(out, "w"))
    print(f"id list ({len(ids)}) written to {out}")


if __name__ == "__main__":
    main()