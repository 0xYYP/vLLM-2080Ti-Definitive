#!/usr/bin/env bash
# Apply the vLLM 2080 Ti Definitive Edition flashinfer 0.6.16rc4 patches.
#
# 用法:
#   bash patches/flashinfer-0.6.16rc4/apply.sh <site-packages 目录>
# 例:
#   bash patches/flashinfer-0.6.16rc4/apply.sh .venv/lib/python3.11/site-packages
#
# 校验 flashinfer 版本为 0.6.16rc4 后，把补丁文件覆盖到安装目录的对应位置，
# 原文件备份为 .bak-<日期>。升级 flashinfer 前先卸载本补丁（还原 .bak）
# 或记录差异，升级后重新适配。
set -euo pipefail

SP="${1:?用法: apply.sh <site-packages 目录>}"
META="$SP/flashinfer_python-0.6.16rc4.dist-info/METADATA"

if [ ! -f "$META" ]; then
  echo "错误: 未找到 $META（需要 flashinfer-python 0.6.16rc4）"
  exit 1
fi
if ! grep -q "^Version: 0.6.16rc4" "$META"; then
  echo "错误: flashinfer 版本不匹配（需要 0.6.16rc4，实际 $(grep '^Version:' "$META")）"
  exit 1
fi

HERE="$(cd "$(dirname "$0")" && pwd)"
STAMP="$(date +%Y%m%d)"

apply() {
  local src_rel="$1"  # 相对本目录（patches/...）的路径
  local dst_rel="$2"  # 相对 site-packages 的路径
  local dst="$SP/$dst_rel"
  if [ ! -f "$dst" ]; then
    echo "错误: 未找到目标文件 $dst"
    exit 1
  fi
  cp "$dst" "$dst.bak-$STAMP"
  cp "$HERE/$src_rel" "$dst"
  echo "已应用: $dst_rel（原文件备份为 $dst_rel.bak-$STAMP）"
}

apply "flashinfer/__init__.py" "flashinfer/__init__.py"
apply "flashinfer/cute_dsl/utils.py" "flashinfer/cute_dsl/utils.py"
apply "flashinfer/jit/utils.py" "flashinfer/jit/utils.py"
apply "flashinfer/comm/fd_exchange.py" "flashinfer/comm/fd_exchange.py"
# CUDA 头文件安装在 flashinfer/data/include/ 下（与仓库内的 include/ 相对路径不同）
apply "include/flashinfer/utils.cuh" "flashinfer/data/include/flashinfer/utils.cuh"
apply "include/flashinfer/vec_dtypes.cuh" "flashinfer/data/include/flashinfer/vec_dtypes.cuh"
apply "include/flashinfer/attention/variant_helper.cuh" "flashinfer/data/include/flashinfer/attention/variant_helper.cuh"
apply "include/flashinfer/attention/decode.cuh" "flashinfer/data/include/flashinfer/attention/decode.cuh"
apply "include/flashinfer/attention/scheduler.cuh" "flashinfer/data/include/flashinfer/attention/scheduler.cuh"

echo "flashinfer 0.6.16rc4 补丁应用完成。"
echo "验证: grep -n 'vllm-2080ti' $SP/flashinfer/comm/fd_exchange.py; grep -n 'QO_LEN' $SP/flashinfer/data/include/flashinfer/attention/decode.cuh; grep -n 'vllm-2080ti' $SP/flashinfer/data/include/flashinfer/attention/scheduler.cuh"
