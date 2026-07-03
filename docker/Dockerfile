# ============================================================
# vLLM 2080Ti FlashQLA - Multi-stage Docker Build
# ============================================================
# Builder:  nvidia/cuda:12.8.1-devel-ubuntu24.04  (includes nvcc)
# Runtime:  nvidia/cuda:12.8.1-runtime-ubuntu24.04 (smaller)
# ============================================================

# --------------- Builder stage ---------------
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_PATH=/usr/local/cuda
ENV CUDACXX=/usr/local/cuda/bin/nvcc
ENV PATH="/usr/local/cuda/bin:${PATH}"

# Install system dependencies (Python 3.11 via deadsnakes PPA on Ubuntu 24.04)
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev \
        git curl wget \
        build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install a pinned uv from PyPI instead of executing a remote installer script.
ARG UV_VERSION=0.5.31
RUN python3.11 -m venv /opt/uv-bootstrap \
    && /opt/uv-bootstrap/bin/pip install --no-cache-dir "uv==${UV_VERSION}" \
    && ln -sf /opt/uv-bootstrap/bin/uv /usr/local/bin/uv

WORKDIR /build

# Copy dependency metadata before source code so package-install layers remain
# cacheable when only application source changes.
COPY requirements requirements

# Build environment variables
ENV TORCH_CUDA_ARCH_LIST=7.5
ENV FLASHINFER_ENABLE_AOT=1
ENV FLASHQLA_DIR=/build/FlashQLA-SM70-SM75
ENV PYTHONPATH=/build/FlashQLA-SM70-SM75
# Use all available CPU cores for parallel compilation (max 16 to avoid OOM)
# GitHub Actions has 2-4 cores, local machines may have more
ARG MAX_JOBS=16
ENV MAX_JOBS=${MAX_JOBS}
ENV CMAKE_BUILD_TYPE=Release
ENV TORCH_EXTENSIONS_DIR=/opt/torch_extensions_cache
ENV VLLM_TARGET_DEVICE=cuda
ARG FLASHQLA_REPO=https://github.com/weicj/FlashQLA-SM70-SM75.git
ARG FLASHQLA_REF=3ab27d77d8ca01d7a4718903b726add1a8886c0e

# Create venv and install deps
RUN uv venv --python 3.11 /build/.venv
ENV PATH="/build/.venv/bin:${PATH}"

RUN uv pip install -U pip setuptools wheel
RUN uv pip install -r requirements/build/cuda.txt --torch-backend=cu128
RUN uv pip install -r requirements/cuda.txt --torch-backend=cu128

# Copy source code after dependency installation to preserve Docker layer cache.
COPY . .

# FlashQLA is fetched during Docker builds so the image does not depend on
# callers cloning this repository with submodules.
RUN git init "$FLASHQLA_DIR" \
    && git -C "$FLASHQLA_DIR" remote add origin "$FLASHQLA_REPO" \
    && git -C "$FLASHQLA_DIR" fetch --depth=1 origin "$FLASHQLA_REF" \
    && git -C "$FLASHQLA_DIR" checkout -q FETCH_HEAD
RUN test -f "$FLASHQLA_DIR/flash_qla/ops/gated_delta_rule/legacy/sm_legacy.py"
RUN python tools/patch_flashqla_sm75_imports.py "$FLASHQLA_DIR" tools/flashqla_sm75_patches
RUN uv pip install --no-deps -e "$FLASHQLA_DIR"
RUN python - <<'PY'
import importlib

import flash_qla

legacy = importlib.import_module("flash_qla.ops.gated_delta_rule.legacy.sm_legacy")
if legacy is None:
    raise SystemExit("FlashQLA legacy module import failed")
print(f"flash_qla={flash_qla.__file__}")
print(f"flash_qla_legacy={legacy.__file__}")
PY

# Build vLLM (editable install so __file__ resolves correctly)
# SETUPTOOLS_SCM_PRETEND_VERSION needed because .git is excluded by .dockerignore
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.1.12
RUN uv pip install --no-build-isolation -e .

# Fix editable install paths: /build -> /opt/vllm (for runtime stage)
RUN SITE=$(/build/.venv/bin/python -c "import site; print(site.getsitepackages()[0])") && \
    for f in "$SITE"/__editable__*finder*.py; do \
        [ -f "$f" ] && sed -i 's|/build|/opt/vllm|g' "$f"; \
    done

# --------------- Runtime stage ---------------
# Must use devel image for FlashQLA CUDA kernel compilation (requires nvcc)
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_PATH=/usr/local/cuda
ENV CUDACXX=/usr/local/cuda/bin/nvcc
ENV PATH="/usr/local/cuda/bin:${PATH}"

# Install runtime dependencies (Python 3.11 via deadsnakes PPA)
# python3.11-dev needed for FlashQLA CUDA kernel compilation (Python.h)
# CUDA stubs are exposed through LIBRARY_PATH for runtime kernel compilation.
# Do not put the stub on the runtime linker path; NVIDIA's injected driver
# libcuda.so.1 must win when GPUs are mounted.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev libpython3.11 \
        curl build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs

# Copy Python environment (with fixed editable paths)
COPY --from=builder /build/.venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Copy only necessary source files (not build cache)
COPY --from=builder /build/vllm /opt/vllm/vllm
COPY --from=builder /build/csrc /opt/vllm/csrc
COPY --from=builder /build/FlashQLA-SM70-SM75 /opt/vllm/FlashQLA-SM70-SM75
COPY --from=builder /build/profiles /opt/vllm/profiles
COPY --from=builder /build/launcher.sh /opt/vllm/
COPY --from=builder /build/PROJECT_RELEASE.env /opt/vllm/
COPY --from=builder /build/VERSION /opt/vllm/
COPY --from=builder /build/setup.py /opt/vllm/
COPY --from=builder /build/pyproject.toml /opt/vllm/
COPY --from=builder /build/CMakeLists.txt /opt/vllm/
COPY docker/docker-entrypoint.sh /opt/vllm/docker/docker-entrypoint.sh
COPY docker/chat_template_no_thinking.jinja /opt/vllm/docker/chat_template_no_thinking.jinja

RUN ln -s /opt/venv /opt/vllm/.venv \
    && chmod +x /opt/vllm/launcher.sh /opt/vllm/docker/docker-entrypoint.sh

WORKDIR /opt/vllm

# Runtime environment
ENV TORCH_CUDA_ARCH_LIST=7.5
ENV FLASHINFER_ENABLE_AOT=1
ENV FLASHQLA_DIR=/opt/vllm/FlashQLA-SM70-SM75
ENV TORCH_EXTENSIONS_DIR=/tmp/torch_extensions_cache
ENV PYTHONPATH="/opt/vllm:/opt/vllm/FlashQLA-SM70-SM75"
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/opt/vllm/docker/docker-entrypoint.sh"]
