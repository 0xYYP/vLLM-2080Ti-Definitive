#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer


def build_prompt(tokenizer, target_tokens: int) -> str:
    if target_tokens <= 0:
        return "OK"
    candidates = [" the", " x", "你", "#\n", "x "]
    for item in candidates:
        prompt = item * target_tokens
        n = len(tokenizer.encode(prompt, add_special_tokens=False))
        if n == target_tokens:
            return prompt
    prompt = " the" * target_tokens
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) >= target_tokens:
        return tokenizer.decode(ids[:target_tokens], skip_special_tokens=False)
    raise RuntimeError(
        f"could not construct {target_tokens} prompt tokens; got {len(ids)}"
    )


def stream_once(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    ignore_eos: bool,
) -> dict:
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
    text_parts: list[str] = []
    completion_token_ids: list[int] = []
    raw_events: list[str] = []
    http_status = None
    error = None
    try:
        with requests.post(
            f"{url}/completions", json=payload, stream=True, timeout=1800
        ) as resp:
            http_status = resp.status_code
            resp.raise_for_status()
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data: "):
                    continue
                data = raw[6:]
                raw_events.append(data[:400])
                if data == "[DONE]":
                    break
                now = time.perf_counter()
                if first is None:
                    first = now
                obj = json.loads(data)
                choice = (obj.get("choices") or [{}])[0]
                token_ids = choice.get("token_ids")
                if isinstance(token_ids, list):
                    completion_token_ids.extend(token_ids)
                text = choice.get("text") or ""
                if text:
                    text_parts.append(text)
                chunks += 1
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    end = time.perf_counter()
    return {
        "http_status": http_status,
        "error": error,
        "ttft_s": (first - start) if first is not None else None,
        "e2e_s": end - start,
        "chunks": chunks,
        "text": "".join(text_parts),
        "completion_tokens_from_ids": len(completion_token_ids),
        "raw_events_preview": raw_events[:5],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--served-name", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:19206/v1")
    ap.add_argument("--prompt-tokens", type=int, required=True)
    ap.add_argument("--gen-tokens", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--ignore-eos", action="store_true")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    prompt = build_prompt(tokenizer, args.prompt_tokens)
    prompt_token_count = len(tokenizer.encode(prompt, add_special_tokens=False))
    result = stream_once(
        args.base_url,
        args.served_name,
        prompt,
        args.gen_tokens,
        args.ignore_eos,
    )
    completion_tokens_from_ids = int(result.get("completion_tokens_from_ids") or 0)
    text = result.pop("text")
    if completion_tokens_from_ids > 0:
        completion_tokens = completion_tokens_from_ids
        completion_token_source = "token_ids"
    else:
        completion_tokens = len(
            tokenizer.encode(text, add_special_tokens=False)
        )
        completion_token_source = "text_fallback"
    ttft = result["ttft_s"]
    decode_s = None
    decode_tok_s = None
    if ttft is not None:
        decode_s = max(result["e2e_s"] - ttft, 1e-9)
        decode_tok_s = completion_tokens / decode_s
    record = {
        "label": args.label,
        "model": args.served_name,
        "model_dir": args.model_dir,
        "requested_prompt_tokens": args.prompt_tokens,
        "prompt_tokens": prompt_token_count,
        "requested_completion_tokens": args.gen_tokens,
        "completion_tokens": completion_tokens,
        "completion_token_source": completion_token_source,
        "ignore_eos": args.ignore_eos,
        **result,
        "prefill_tok_s": (prompt_token_count / ttft) if ttft else None,
        "decode_tok_s": decode_tok_s,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
