#!/usr/bin/env bash
# 安装/更新 qwen38-production systemd 服务。用法：sudo bash install_qwen38_production_service.sh
#
# **这是会中断服务的操作**：先停掉现存实例（不管是 nohup 起的还是 systemd 管的），
# 再交给 systemd 托管并等 /v1/models 就绪。预计停机 3-4 分钟（vLLM 冷启动约 200s）。
# 为什么不直接 kill 了再 nohup：只有 systemd 托管才有开机自启和崩溃重启，
# 而这也正是这份脚本要解决的问题（此前唯一的生产启动脚本在 /tmp 上，重启即丢）。
set -uo pipefail
BIN=/opt/vllm-2080ti-definitive/bin
SCRIPT=$BIN/start_qwen38_production.sh
UNIT=qwen38-production.service
PORT=${PORT:-8000}
WAIT_SECONDS=${WAIT_SECONDS:-600}

if [ "$(id -u)" != "0" ]; then
  echo "需要 root：sudo bash $0"
  exit 1
fi
[ -x "$SCRIPT" ] || {
  echo "缺少 $SCRIPT——先把 tools/start_qwen38_production.sh 部署到 $BIN/"
  exit 1
}
[ -f "$BIN/$UNIT" ] || {
  echo "缺少 $BIN/$UNIT——先把 tools/$UNIT 部署到 $BIN/"
  exit 1
}

echo "1/5 停掉 systemd 之外的旧实例（nohup 起的）"
bash "$SCRIPT" --stop --force || true

echo "2/5 停掉可能已存在的 systemd 服务"
systemctl stop "$UNIT" 2>/dev/null || true

echo "3/5 写默认的环境覆盖文件（若不存在）+ 安装 unit"
if [ ! -f "$BIN/qwen38-production.env" ]; then
  cat >"$BIN/qwen38-production.env" <<'EOF'
# qwen38-production.service 的可选环境覆盖（EnvironmentFile 语法：KEY=value）
# 默认值即生产值；只在你需要偏离生产配置时才取消注释。
#MTP_K=4
#MAX_MODEL_LEN=262144
#GPU_UTIL=0.98
#MAX_NUM_SEQS=1
#MAX_NUM_BATCHED_TOKENS=2048
#KV_CACHE_DTYPE=float16
#GPU_DEVICES=0,1
#PORT=8000
EOF
  echo "    已写入 $BIN/qwen38-production.env"
fi
install -m 0644 "$BIN/$UNIT" "/etc/systemd/system/$UNIT"
systemctl daemon-reload
systemctl enable "$UNIT"

echo "4/5 启动（等待 /v1/models 就绪，最多 ${WAIT_SECONDS}s）"
systemctl start "$UNIT"

echo "5/5 验收"
for ((i = 1; i <= WAIT_SECONDS / 5; i++)); do
  if curl -s -m 8 -o /dev/null "http://127.0.0.1:$PORT/v1/models"; then
    echo "✅ 已由 systemd 托管并通过 /v1/models 验收（约 $((i * 5))s）"
    systemctl status --no-pager -l "$UNIT" | head -14
    exit 0
  fi
  sleep 5
done

echo "⚠️ ${WAIT_SECONDS}s 内没就绪。回退到 nohup 方式："
echo "    systemctl stop $UNIT && bash $SCRIPT"
journalctl -u "$UNIT" --no-pager -n 30
exit 1
