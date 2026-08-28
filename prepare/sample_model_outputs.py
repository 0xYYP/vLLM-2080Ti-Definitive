#!/usr/bin/env python3
"""Sample model outputs for draft-vocab frequency counting.

Batch non-streaming chat requests against a running vLLM server, appending
output text to a corpus JSONL. Diversity comes from the prompt set; greedy
decoding keeps the sampling fast and deterministic.

Usage:
    venv/bin/python prepare/sample_model_outputs.py \
        --base http://127.0.0.1:8000 --model <served-model> \
        --prompts prompts.jsonl --out corpus.jsonl \
        --max-tokens 4096 --temperature 0 --limit 200

prompts.jsonl: one JSON per line with {"prompt": "...", "tag": "zh_news"}.
"""
import argparse
import json
import time
import urllib.request


def call(base, model, prompt, max_tokens, temperature, timeout=1200):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        base + "/v1/chat/completions", body,
        {"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    out = (data["choices"][0]["message"]["content"] or "")
    return out, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    prompts = [json.loads(l) for l in open(args.prompts, encoding="utf-8") if l.strip()]
    if args.limit:
        prompts = prompts[: args.limit]
    total_out = 0
    with open(args.out, "a", encoding="utf-8") as f:
        for i, item in enumerate(prompts):
            out, dt = call(args.base, args.model, item["prompt"],
                           args.max_tokens, args.temperature)
            total_out += len(out)
            rec = {"tag": item.get("tag", "gen"), "prompt": item["prompt"], "output": out}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i+1}/{len(prompts)}] tag={rec['tag']} out_chars={len(out)} "
                  f"wall={dt:.1f}s total_chars={total_out}", flush=True)


if __name__ == "__main__":
    main()