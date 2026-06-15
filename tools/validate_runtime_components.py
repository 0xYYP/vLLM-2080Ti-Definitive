#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate runtime components required by the 2080 Ti build.

This is intentionally stricter than a plain import smoke. It is run by
``build.sh`` after installation so missing CUDA extensions fail during build
instead of later during model startup.
"""

from __future__ import annotations

import importlib
import importlib.metadata as metadata
import os
from pathlib import Path
import sys


FAILURES: list[str] = []


def ok(name: str, detail: object) -> None:
    print(f"OK {name}: {detail}")


def fail(name: str, detail: object) -> None:
    FAILURES.append(f"{name}: {detail}")
    print(f"FAIL {name}: {detail}")


def require(condition: bool, name: str, detail: object) -> None:
    if condition:
        ok(name, detail)
    else:
        fail(name, detail)


def version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not-installed"


def import_module(name: str):
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        fail(name, f"{type(exc).__name__}: {exc}")
        return None
    ok(name, getattr(module, "__file__", "built-in"))
    return module


def require_file(path: str | os.PathLike[str], name: str) -> Path | None:
    resolved = Path(path).expanduser().resolve()
    if resolved.is_file():
        ok(name, resolved)
        return resolved
    fail(name, f"missing file: {resolved}")
    return None


def require_dir(path: str | os.PathLike[str], name: str) -> Path | None:
    resolved = Path(path).expanduser().resolve()
    if resolved.is_dir():
        ok(name, resolved)
        return resolved
    fail(name, f"missing directory: {resolved}")
    return None


def validate_torch_cuda() -> None:
    torch = import_module("torch")
    if torch is None:
        return

    ok("torch.version", getattr(torch, "__version__", "unknown"))
    require(torch.cuda.is_available(), "torch.cuda", "CUDA must be available")
    if torch.cuda.is_available():
        ok("torch.cuda.device_count", torch.cuda.device_count())
        for idx in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(idx)
            capability = torch.cuda.get_device_capability(idx)
            ok(f"torch.cuda.device_{idx}", f"{name} sm_{capability[0]}{capability[1]}")


def validate_vllm_extensions() -> None:
    import_module("vllm")
    vllm_c = import_module("vllm._C")
    import_module("vllm._custom_ops")

    if vllm_c is not None:
        require_file(getattr(vllm_c, "__file__", ""), "vllm._C.shared_object")

    torch = import_module("torch")
    if torch is None:
        return

    core_ops = [
        "rms_norm",
        "fused_add_rms_norm",
        "gptq_gemm",
        "gptq_marlin_repack",
        "awq_gemm",
        "awq_dequantize",
        "cutlass_scaled_mm",
        "cutlass_scaled_mm_azp",
        "dynamic_scaled_int8_quant",
        "static_scaled_int8_quant",
    ]
    for op_name in core_ops:
        require(
            hasattr(torch.ops._C, op_name),
            f"torch.ops._C.{op_name}",
            "registered",
        )

    cache_ops = ["reshape_and_cache", "reshape_and_cache_flash"]
    for op_name in cache_ops:
        require(
            hasattr(torch.ops._C_cache_ops, op_name),
            f"torch.ops._C_cache_ops.{op_name}",
            "registered",
        )


def validate_flashinfer() -> None:
    flashinfer = import_module("flashinfer")
    if flashinfer is not None:
        ok("flashinfer.version", getattr(flashinfer, "__version__", version("flashinfer-python")))

    require(version("flashinfer-python") != "not-installed", "flashinfer-python", version("flashinfer-python"))
    require(version("flashinfer-cubin") != "not-installed", "flashinfer-cubin", version("flashinfer-cubin"))

    for module_name in [
        "flashinfer.prefill",
        "flashinfer.decode",
        "flashinfer.cascade",
        "flashinfer.sampling",
        "flashinfer.gdn_prefill",
        "flashinfer_cubin",
    ]:
        import_module(module_name)


def validate_flashqla_legacy() -> None:
    flashqla_dir = Path(os.environ.get("FLASHQLA_DIR", ""))
    if not flashqla_dir:
        fail("FlashQLA.dir", "FLASHQLA_DIR is not set")
        return
    flashqla_dir = flashqla_dir.expanduser().resolve()
    require_dir(flashqla_dir, "FlashQLA.dir")
    require_dir(flashqla_dir / "flash_qla", "FlashQLA.package")
    require_file(
        flashqla_dir
        / "flash_qla"
        / "ops"
        / "gated_delta_rule"
        / "legacy"
        / "sm_legacy.py",
        "FlashQLA.legacy.sm_legacy",
    )
    require_file(
        flashqla_dir
        / "flash_qla"
        / "ops"
        / "gated_delta_rule"
        / "legacy"
        / "csrc"
        / "gdn_forward.cu",
        "FlashQLA.legacy.gdn_forward",
    )

    if str(flashqla_dir) not in sys.path:
        sys.path.insert(0, str(flashqla_dir))
    legacy = import_module("flash_qla.ops.gated_delta_rule.legacy.sm_legacy")
    if legacy is None:
        return

    try:
        ext = legacy._load_ext()
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        fail("FlashQLA.legacy.extension", f"{type(exc).__name__}: {exc}")
        return

    so_path = Path(getattr(ext, "__file__", ""))
    if so_path.is_file():
        ok("FlashQLA.legacy.extension", so_path)
    else:
        fail("FlashQLA.legacy.extension", f"shared object missing: {so_path}")


def validate_source_trees() -> None:
    cutlass_dir = os.environ.get("VLLM_CUTLASS_SRC_DIR", "")
    triton_kernels_dir = os.environ.get("TRITON_KERNELS_SRC_DIR", "")

    if cutlass_dir:
        root = require_dir(cutlass_dir, "CUTLASS.source")
        if root is not None:
            require_file(root / "include" / "cutlass" / "cutlass.h", "CUTLASS.header")
    else:
        fail("CUTLASS.source", "VLLM_CUTLASS_SRC_DIR is not set")

    if triton_kernels_dir:
        root = require_dir(triton_kernels_dir, "TritonKernels.source")
        if root is not None:
            require_file(root / "__init__.py", "TritonKernels.package")
    else:
        fail("TritonKernels.source", "TRITON_KERNELS_SRC_DIR is not set")


def main() -> int:
    print("Runtime component validation")
    validate_torch_cuda()
    validate_vllm_extensions()
    validate_flashinfer()
    validate_flashqla_legacy()
    validate_source_trees()

    if FAILURES:
        print()
        print("Runtime component validation failed:")
        for item in FAILURES:
            print(f"  - {item}")
        return 1

    print()
    print("Runtime component validation OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
