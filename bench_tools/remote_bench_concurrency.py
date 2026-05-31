#!/usr/bin/env python3
import argparse
import concurrent.futures as cf
import json
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer


def make_prompt(tokenizer, target_tokens: int, salt: str) -> str:
    prefix = f"unique-concurrency-{salt}\n"
    ids = tokenizer.encode(prefix + (" the" * (target_tokens + 64)), add_special_tokens=False)
    ids = ids[:target_tokens]
    if len(ids) < target_tokens:
        raise RuntimeError(f"prompt too short: {len(ids)}")
    return tokenizer.decode(ids, skip_special_tokens=False)


def stream_once(base_url, model, prompt, max_tokens, ignore_eos):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": ignore_eos,
        "return_token_ids": True,
    }
    start = time.perf_counter()
    first = None
    chunks = 0
    text_parts = []
    completion_token_ids = []
    with requests.post(f"{base_url}/completions", json=payload, stream=True, timeout=3600) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data: "):
                continue
            data = raw[6:]
            if data == "[DONE]":
                break
            now = time.perf_counter()
            obj = json.loads(data)
            if "choices" not in obj:
                raise RuntimeError(json.dumps(obj, ensure_ascii=False))
            choice = obj["choices"][0]
            token_ids = choice.get("token_ids")
            if isinstance(token_ids, list):
                completion_token_ids.extend(token_ids)
            text = choice.get("text") or ""
            if first is None:
                first = now
            if text:
                text_parts.append(text)
            chunks += 1
    end = time.perf_counter()
    return {
        "ttft_s": first - start if first else None,
        "e2e_s": end - start,
        "chunks": chunks,
        "text": "".join(text_parts),
        "completion_tokens_from_ids": len(completion_token_ids),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--served-name", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:19206/v1")
    ap.add_argument("--prompt-tokens", type=int, required=True)
    ap.add_argument("--gen-tokens", type=int, required=True)
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ignore-eos", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    prompts = [make_prompt(tok, args.prompt_tokens, f"{args.label}-{i}-{time.time_ns()}") for i in range(args.concurrency)]
    prompt_counts = [len(tok.encode(p, add_special_tokens=False)) for p in prompts]

    wall_start = time.perf_counter()
    rows = []
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(stream_once, args.base_url, args.served_name, prompts[i], args.gen_tokens, args.ignore_eos) for i in range(args.concurrency)]
        for i, fut in enumerate(futs):
            res = fut.result()
            completion_tokens_from_ids = int(res.get("completion_tokens_from_ids") or 0)
            text = res.pop("text")
            if completion_tokens_from_ids > 0:
                completion_tokens = completion_tokens_from_ids
                completion_token_source = "token_ids"
            else:
                completion_tokens = len(tok.encode(text, add_special_tokens=False))
                completion_token_source = "text_fallback"
            ttft = res["ttft_s"]
            decode_s = max(res["e2e_s"] - (ttft or 0), 1e-9)
            rows.append({
                "request_index": i,
                "prompt_tokens": prompt_counts[i],
                "completion_tokens": completion_tokens,
                "completion_token_source": completion_token_source,
                **res,
                "prefill_tok_s_proxy": prompt_counts[i] / ttft if ttft else None,
                "decode_tok_s_per_request": completion_tokens / decode_s,
            })
    wall_end = time.perf_counter()
    total_prompt = sum(r["prompt_tokens"] for r in rows)
    total_completion = sum(r["completion_tokens"] for r in rows)
    batch_wall = wall_end - wall_start
    rec = {
        "label": args.label,
        "model": args.served_name,
        "concurrency": args.concurrency,
        "requested_prompt_tokens": args.prompt_tokens,
        "requested_completion_tokens": args.gen_tokens,
        "ignore_eos": args.ignore_eos,
        "batch_wall_s": batch_wall,
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "aggregate_total_tok_s": (total_prompt + total_completion) / batch_wall,
        "aggregate_completion_tok_s": total_completion / batch_wall,
        "mean_ttft_s": sum(r["ttft_s"] for r in rows) / len(rows),
        "max_ttft_s": max(r["ttft_s"] for r in rows),
        "mean_e2e_s": sum(r["e2e_s"] for r in rows) / len(rows),
        "max_e2e_s": max(r["e2e_s"] for r in rows),
        "requests": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps(rec, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
