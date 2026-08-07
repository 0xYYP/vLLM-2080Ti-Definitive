#!/usr/bin/env bash
# Apply the vLLM 2080 Ti Definitive Edition flashinfer patches.
#
# 用法:
#   bash patches/flashinfer-0.6.8.post1/apply.sh <site-packages 目录>
# 例:
#   bash patches/flashinfer-0.6.8.post1/apply.sh .venv/lib/python3.11/site-packages
#
# 校验 flashinfer 版本为 0.6.8.post1 后，把 4 个补丁文件覆盖到
# flashinfer/data/include/flashinfer/，原文件备份为 .bak-<日期>。
# 升级 flashinfer 前先卸载本补丁（还原 .bak）或记录差异，升级后重新适配。
set -euo pipefail

SP="${1:?用法: apply.sh <site-packages 目录>}"
INC="$SP/flashinfer/data/include/flashinfer"
META="$SP/flashinfer-0.6.8.post1.dist-info/METADATA"

if [ ! -f "$META" ]; then
  echo "错误: 未找到 $META（需要 flashinfer 0.6.8.post1）"
  exit 1
fi
if ! grep -q "^Version: 0.6.8.post1" "$META"; then
  echo "错误: flashinfer 版本不匹配（需要 0.6.8.post1，实际 $(grep '^Version:' "$META")）"
  exit 1
fi

HERE="$(cd "$(dirname "$0")" && pwd)"
STAMP="$(date +%Y%m%d)"
for f in attention/decode.cuh attention/prefill.cuh attention/variant_helper.cuh vec_dtypes.cuh; do
  name="$(basename "$f")"
  dst="$INC/$f"
  if [ ! -f "$dst" ]; then
    echo "错误: 未找到目标文件 $dst"
    exit 1
  fi
  cp "$dst" "$dst.bak-$STAMP"
  cp "$HERE/$name" "$dst"
  echo "已应用: $f（原文件备份为 $f.bak-$STAMP）"
done
echo "flashinfer 0.6.8.post1 补丁应用完成。"
echo "验证: grep -n 'vLLM patch' $INC/attention/prefill.cuh; grep -n 'batch_idx = 0u' $INC/attention/decode.cuh"
