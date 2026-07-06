#!/usr/bin/env python3
import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import requests
from transformers import AutoTokenizer


REPEATED_CHARS = re.compile(r"(.)\1{24,}", re.DOTALL)
REPEATED_PHRASE = re.compile(r"(.{8,80})\1{3,}", re.DOTALL)


def stream_completion(
    base_url: str,
    payload: dict[str, Any],
    *,
    timeout: float,
) -> dict[str, Any]:
    start = time.perf_counter()
    first = None
    status = None
    error = None
    stream_done = False
    chunks = 0
    text_parts: list[str] = []
    raw_preview: list[str] = []

    try:
        with requests.post(
            f"{base_url.rstrip('/')}/completions",
            json=payload,
            stream=True,
            timeout=(30, timeout),
        ) as resp:
            status = resp.status_code
            resp.raise_for_status()
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data: "):
                    continue
                data = raw[6:]
                if len(raw_preview) < 8:
                    raw_preview.append(data[:500])
                if data == "[DONE]":
                    stream_done = True
                    break
                now = time.perf_counter()
                if first is None:
                    first = now
                chunks += 1
                obj = json.loads(data)
                choice = (obj.get("choices") or [{}])[0]
                text = choice.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"

    elapsed = time.perf_counter() - start
    return {
        "http_status": status,
        "error": error,
        "stream_done": stream_done,
        "chunks": chunks,
        "text": "".join(text_parts),
        "raw_events_preview": raw_preview,
        "ttft_s": None if first is None else first - start,
        "elapsed_s": elapsed,
    }


def repeat_ids(unit_ids: list[int], count: int) -> list[int]:
    if count <= 0:
        return []
    repeats = math.ceil(count / len(unit_ids))
    return (unit_ids * repeats)[:count]


def build_niah_prompt(
    tokenizer: Any,
    *,
    target_tokens: int,
    depth: float,
    code: str,
) -> tuple[str, int, bool]:
    prefix = (
        "你正在做长上下文检索测试。下面的资料绝大多数都是无关背景。"
        "请只记住其中明确写出的秘密代码，最后只回答代码本身。\n\n"
        "资料开始。\n"
    )
    suffix = "\n资料结束。\n问题：上文中的秘密代码是什么？只输出秘密代码本身，不要解释。\n答案："
    needle = f"\n[关键事实] 秘密代码是 {code}。这是唯一正确答案。\n"
    filler_unit = (
        " This background sentence is irrelevant and contains no secret code."
        " It exists only to fill the long context."
    )

    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    needle_ids = tokenizer.encode(needle, add_special_tokens=False)
    unit_ids = tokenizer.encode(filler_unit, add_special_tokens=False)
    budget = target_tokens - len(prefix_ids) - len(suffix_ids) - len(needle_ids)
    if budget < 0:
        raise ValueError(
            f"target_tokens={target_tokens} is too small for the NIAH template"
        )

    before_count = max(0, min(budget, round(budget * depth)))
    after_count = budget - before_count
    ids = (
        prefix_ids
        + repeat_ids(unit_ids, before_count)
        + needle_ids
        + repeat_ids(unit_ids, after_count)
        + suffix_ids
    )
    prompt = tokenizer.decode(ids, skip_special_tokens=False)
    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    return prompt, prompt_tokens, code in prompt


def smoke_cases() -> list[tuple[str, str]]:
    return [
        (
            "zh_summary",
            "请用两句话说明为什么长上下文 KV 缓存优化必须同时验证速度和质量。不要重复。",
        ),
        (
            "zh_ordered_list",
            "请按顺序输出五行：甲、乙、丙、丁、戊。每行只包含一个字。",
        ),
    ]


def looks_degenerate(text: str) -> bool:
    return bool(REPEATED_CHARS.search(text) or REPEATED_PHRASE.search(text))


def run_one(
    tokenizer: Any,
    args: argparse.Namespace,
    *,
    case_label: str,
    prompt: str,
    prompt_tokens: int,
    expected: str | None,
    needle_present: bool | None,
    depth: float | None,
) -> dict[str, Any]:
    payload = {
        "model": args.served_name,
        "prompt": prompt,
        "max_tokens": args.gen_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    if args.ignore_eos:
        payload["ignore_eos"] = True

    result = stream_completion(args.base_url, payload, timeout=args.read_timeout)
    text = result.pop("text")
    completion_tokens = len(tokenizer.encode(text, add_special_tokens=False))
    correct = expected in text if expected is not None else None
    degenerate = looks_degenerate(text)
    ok = (
        result.get("error") is None
        and result.get("http_status") == 200
        and result.get("stream_done") is True
        and bool(text.strip())
        and not degenerate
        and (correct is not False)
    )

    record = {
        "label": args.label,
        "case": case_label,
        "model": args.served_name,
        "model_dir": args.model_dir,
        "mode": args.mode,
        "requested_context_tokens": args.context_tokens,
        "prompt_tokens": prompt_tokens,
        "depth": depth,
        "expected": expected,
        "needle_present_in_prompt": needle_present,
        "correct": correct,
        "ok": ok,
        "degenerate": degenerate,
        "completion_tokens": completion_tokens,
        "text_chars": len(text),
        "content_sample": text[:500],
        **result,
    }
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--served-name", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--mode", choices=("niah", "chinese-smoke"), required=True)
    parser.add_argument("--context-tokens", type=int, default=65536)
    parser.add_argument("--depths", default="0.50")
    parser.add_argument("--gen-tokens", type=int, default=64)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--read-timeout", type=float, default=1800.0)
    parser.add_argument("--ignore-eos", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    records: list[dict[str, Any]] = []

    if args.mode == "chinese-smoke":
        for case_label, prompt in smoke_cases():
            prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            records.append(
                run_one(
                    tokenizer,
                    args,
                    case_label=case_label,
                    prompt=prompt,
                    prompt_tokens=prompt_tokens,
                    expected=None,
                    needle_present=None,
                    depth=None,
                )
            )
    else:
        depths = [float(item) for item in args.depths.split(",") if item.strip()]
        for depth in depths:
            depth_pct = int(round(depth * 100))
            code = f"SM75-INT8-NIAH-{args.context_tokens}-{depth_pct:03d}"
            prompt, prompt_tokens, needle_present = build_niah_prompt(
                tokenizer,
                target_tokens=args.context_tokens,
                depth=depth,
                code=code,
            )
            records.append(
                run_one(
                    tokenizer,
                    args,
                    case_label=f"niah_{args.context_tokens}_{depth_pct:03d}",
                    prompt=prompt,
                    prompt_tokens=prompt_tokens,
                    expected=code,
                    needle_present=needle_present,
                    depth=depth,
                )
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(json.dumps({"ok": all(r["ok"] for r in records), "records": records}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
