#!/usr/bin/env python3
"""Plan fp16/int8 hybrid KV cache skip layers.

This helper does not benchmark a route. It produces a deterministic
``KV_CACHE_DTYPE_SKIP_LAYERS`` value and a coarse KV-memory ratio estimate so
the real benchmark starts from a route that can plausibly satisfy the capacity
gate.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ATTENTION_LAYER_TYPES = {
    "attention",
    "full_attention",
    "global_attention",
    "hybrid",
    "sliding_attention",
    "sliding_window",
}


def load_model_config(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"config.json not found under {model_dir}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def text_config(config: dict[str, Any]) -> dict[str, Any]:
    nested = config.get("text_config")
    return nested if isinstance(nested, dict) else config


def get_int(config: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = config.get(key)
        if isinstance(value, int):
            return value
    return None


def get_sequence(config: dict[str, Any], *keys: str) -> list[Any] | None:
    for key in keys:
        value = config.get(key)
        if isinstance(value, list):
            return value
    return None


def attention_layers(config: dict[str, Any]) -> list[int]:
    text = text_config(config)
    num_layers = get_int(text, "num_hidden_layers", "n_layer", "num_layers")
    if num_layers is None:
        num_layers = get_int(config, "num_hidden_layers", "n_layer", "num_layers")
    if num_layers is None:
        raise SystemExit("Could not infer num_hidden_layers from config.json")

    layer_types = get_sequence(
        text,
        "layer_types",
        "layers_block_type",
        "attn_type_list",
    )
    if layer_types is None:
        layer_types = get_sequence(
            config,
            "layer_types",
            "layers_block_type",
            "attn_type_list",
        )
    if layer_types is None:
        return list(range(num_layers))

    layers: list[int] = []
    for idx, layer_type in enumerate(layer_types):
        if isinstance(layer_type, int):
            if layer_type == 1:
                layers.append(idx)
            continue
        if str(layer_type).lower() in ATTENTION_LAYER_TYPES:
            layers.append(idx)
    if not layers:
        raise SystemExit("config.json has layer type metadata, but no attention layers matched")
    return layers


def infer_head_size(config: dict[str, Any], explicit_head_size: int | None) -> int:
    if explicit_head_size is not None:
        return explicit_head_size
    text = text_config(config)
    head_size = get_int(text, "head_dim", "head_size")
    if head_size is not None:
        return head_size
    hidden_size = get_int(text, "hidden_size", "n_embd", "d_model")
    num_heads = get_int(text, "num_attention_heads", "n_head")
    if hidden_size is None:
        hidden_size = get_int(config, "hidden_size", "n_embd", "d_model")
    if num_heads is None:
        num_heads = get_int(config, "num_attention_heads", "n_head")
    if hidden_size is None or num_heads is None or num_heads == 0:
        raise SystemExit("Could not infer head_size; pass --head-size")
    return hidden_size // num_heads


def quant_layer_ratio(kv_dtype: str, head_size: int, aligned_int8: bool) -> float:
    if kv_dtype in ("auto", "float16", "fp16", ""):
        return 1.0
    if kv_dtype in ("fp8", "fp8_e4m3", "fp8_e5m2"):
        return 0.5
    if kv_dtype in ("int8_per_token_head", "fp8_per_token_head"):
        if aligned_int8 and kv_dtype == "int8_per_token_head":
            padded = math.ceil((head_size + 4) / 16) * 16
            return (2 * padded) / (4 * head_size)
        return (2 * head_size + 8) / (4 * head_size)
    if kv_dtype == "turboquant_k8v4":
        slot_size = head_size + math.ceil(head_size * 4 / 8) + 4
        slot_size += slot_size % 2
        return slot_size / (4 * head_size)
    raise SystemExit(f"Unsupported kv dtype for estimate: {kv_dtype}")


def evenly_spaced(items: list[int], count: int) -> list[int]:
    if count <= 0:
        return []
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[len(items) // 2]]
    selected = {
        items[round(i * (len(items) - 1) / (count - 1))]
        for i in range(count)
    }
    idx = 0
    while len(selected) < count and idx < len(items):
        selected.add(items[idx])
        idx += 1
    return sorted(selected)


def boundary_layers(items: list[int], count: int) -> list[int]:
    if count <= 0:
        return []
    if count >= len(items):
        return list(items)
    left = (count + 1) // 2
    right = count // 2
    return sorted(set(items[:left] + items[-right:] if right else items[:left]))


def select_layers_by_count(items: list[int], count: int, policy: str) -> list[int]:
    count = max(0, min(len(items), count))
    if policy == "balanced":
        return evenly_spaced(items, count)
    if policy == "boundary":
        return boundary_layers(items, count)
    if policy == "alternate":
        stride = 2 if fraction >= 0.5 else max(2, round(1 / max(fraction, 1e-6)))
        selected = items[::stride]
        return evenly_spaced(selected, count) if len(selected) != count else selected
    raise SystemExit(f"Unknown policy: {policy}")


def max_fp16_count(attention_layer_count: int, q_ratio: float, target_ratio: float) -> int:
    if attention_layer_count <= 0:
        return 0
    if q_ratio >= 1:
        return attention_layer_count if target_ratio >= 1 else 0
    max_fraction = (target_ratio - q_ratio) / (1 - q_ratio)
    max_fraction = max(0.0, min(1.0, max_fraction))
    return math.floor(attention_layer_count * max_fraction + 1e-9)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--kv-dtype", default="int8_per_token_head")
    parser.add_argument(
        "--fp16-fraction",
        type=float,
        help=(
            "Explicit fraction of attention layers to keep in fp16. "
            "By default the tool picks the largest count under --target-kv-ratio."
        ),
    )
    parser.add_argument(
        "--policy",
        choices=("balanced", "boundary", "alternate"),
        default="balanced",
    )
    parser.add_argument("--target-kv-ratio", type=float, default=0.8)
    parser.add_argument("--head-size", type=int)
    parser.add_argument("--aligned-int8", action="store_true")
    args = parser.parse_args()

    if args.fp16_fraction is not None and not 0 <= args.fp16_fraction <= 1:
        raise SystemExit("--fp16-fraction must be between 0 and 1")

    config = load_model_config(args.model_dir)
    layers = attention_layers(config)
    head_size = infer_head_size(config, args.head_size)
    q_ratio = quant_layer_ratio(args.kv_dtype, head_size, args.aligned_int8)
    if args.fp16_fraction is None:
        fp16_count = max_fp16_count(len(layers), q_ratio, args.target_kv_ratio)
    else:
        fp16_count = round(len(layers) * args.fp16_fraction)
    selected = select_layers_by_count(layers, fp16_count, args.policy)
    hybrid_ratio = (len(selected) + (len(layers) - len(selected)) * q_ratio) / len(layers)
    max_count = max_fp16_count(len(layers), q_ratio, args.target_kv_ratio)
    max_fp16_fraction = max_count / len(layers)

    skip_layers = ",".join(str(i) for i in selected)
    print(f"attention_layers={','.join(str(i) for i in layers)}")
    print(f"attention_layer_count={len(layers)}")
    print(f"head_size={head_size}")
    print(f"kv_dtype={args.kv_dtype}")
    print(f"quant_layer_kv_ratio={q_ratio:.4f}")
    print(f"policy={args.policy}")
    print(f"fp16_skip_layer_count={len(selected)}")
    print(f"fp16_skip_fraction={len(selected) / len(layers):.4f}")
    print(f"estimated_hybrid_kv_ratio={hybrid_ratio:.4f}")
    print(f"target_kv_ratio={args.target_kv_ratio:.4f}")
    print(f"target_ok={str(hybrid_ratio <= args.target_kv_ratio).lower()}")
    print(f"max_fp16_count_for_target={max_count}")
    print(f"max_fp16_fraction_for_target={max_fp16_fraction:.4f}")
    print()
    print("profile_lines:")
    print(f"KV_CACHE_DTYPE={args.kv_dtype}")
    print(f"KV_CACHE_DTYPE_SKIP_LAYERS={skip_layers}")
    if args.aligned_int8 and args.kv_dtype == "int8_per_token_head":
        print("VLLM_INT8KV_ALIGNED_HEAD_STRIDE=1")


if __name__ == "__main__":
    main()
