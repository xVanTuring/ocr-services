#!/usr/bin/env bash
# 管理本地 OCR 服务的 launchd 后台任务（用户级 LaunchAgent，不需要 sudo）。
#
#   ./ocr-service.sh install     生成 plist、加载并启动（登录后自动启动）
#   ./ocr-service.sh uninstall   停止并删除 plist
#   ./ocr-service.sh start       启动（已 install 的前提下）
#   ./ocr-service.sh stop        停止（本次登录内不再自动拉起）
#   ./ocr-service.sh restart     重启
#   ./ocr-service.sh status      launchd 状态 + /healthz
#   ./ocr-service.sh logs        跟随日志
#   ./ocr-service.sh plist       只打印将要写入的 plist，不安装
#
# install 时读取的环境变量（不设则用默认值，写进 plist 后固定）：
#   OCR_PORT=8766  OCR_HOST=127.0.0.1  OCR_VARIANT=medium  OCR_IDLE_SECONDS=300
# 兼容 macOS 自带 bash 3.2。
set -euo pipefail

LABEL="tech.xvanturing.ocr-service"
ROOT="$(cd "$(dirname "$0")" && pwd)"
SERVICE_DIR="$ROOT/service"
BIN="$SERVICE_DIR/.venv/bin/ocr-service"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG_DIR="$HOME/Library/Logs/ocr-service"
DOMAIN="gui/$(id -u)"

PORT="${OCR_PORT:-8766}"
HOST="${OCR_HOST:-127.0.0.1}"
VARIANT="${OCR_VARIANT:-medium}"
IDLE="${OCR_IDLE_SECONDS:-300}"

die() { echo "错误：$*" >&2; exit 1; }

is_loaded() { launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; }

render_plist() {
  # HF_HUB_OFFLINE=1：launchd 不继承终端里的代理变量，联网检查 HuggingFace 会卡住；
  # 权重必须已经下载好（见 check_ready）。
  cat <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$BIN</string>
  </array>
  <key>WorkingDirectory</key>
  <string>$SERVICE_DIR</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>OCR_HOST</key>
    <string>$HOST</string>
    <key>OCR_PORT</key>
    <string>$PORT</string>
    <key>OCR_VARIANT</key>
    <string>$VARIANT</string>
    <key>OCR_IDLE_SECONDS</key>
    <string>$IDLE</string>
    <key>HF_HUB_OFFLINE</key>
    <string>1</string>
  </dict>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <dict>
    <key>SuccessfulExit</key>
    <false/>
  </dict>
  <key>ThrottleInterval</key>
  <integer>10</integer>
  <key>StandardOutPath</key>
  <string>$LOG_DIR/out.log</string>
  <key>StandardErrorPath</key>
  <string>$LOG_DIR/err.log</string>
</dict>
</plist>
EOF
}

check_ready() {
  [ -x "$BIN" ] || die "找不到 $BIN，请先执行 ./setup.sh 安装依赖"
  # 预训练权重缓存（路径相对服务工作目录）和 HF 权重缓存都要在，否则离线模式下加载会失败
  [ -f "$SERVICE_DIR/.cache/paddle_pretrained/PP-OCRv6_${VARIANT}_rec_pretrained.pdparams" ] \
    || die "还没有 $VARIANT 的权重缓存，请先在终端执行 ./run.sh 让它下载一次（出现 Uvicorn running 后 Ctrl+C）"
  ls -d "$HOME"/.cache/huggingface/hub/models--PaddlePaddle--PP-OCRv6_"${VARIANT}"_det_safetensors >/dev/null 2>&1 \
    || die "HuggingFace 缓存里没有 $VARIANT 检测模型，请先执行 ./run.sh 下载一次"
}

cmd_install() {
  check_ready
  mkdir -p "$LOG_DIR" "$(dirname "$PLIST")"
  if is_loaded; then
    launchctl bootout "$DOMAIN/$LABEL" || true
  fi
  render_plist >"$PLIST"
  plutil -lint "$PLIST" >/dev/null || die "生成的 plist 不合法：$PLIST"
  launchctl bootstrap "$DOMAIN" "$PLIST"
  echo "已安装并启动：$LABEL （http://$HOST:$PORT，空闲 ${IDLE}s 后卸载模型）"
  echo "日志：$LOG_DIR"
}

cmd_uninstall() {
  if is_loaded; then
    launchctl bootout "$DOMAIN/$LABEL" || true
  fi
  rm -f "$PLIST"
  echo "已卸载：$LABEL （日志保留在 $LOG_DIR）"
}

cmd_start() {
  [ -f "$PLIST" ] || die "尚未安装，请先执行 ./ocr-service.sh install"
  if is_loaded; then
    echo "已经在运行"
    return
  fi
  launchctl bootstrap "$DOMAIN" "$PLIST"
  echo "已启动"
}

cmd_stop() {
  if ! is_loaded; then
    echo "没有在运行"
    return
  fi
  launchctl bootout "$DOMAIN/$LABEL"
  echo "已停止（下次登录或执行 start 时会再启动）"
}

cmd_status() {
  if is_loaded; then
    launchctl print "$DOMAIN/$LABEL" | grep -E "^\s*(state|pid|last exit code|runs) =" || true
  else
    echo "launchd：未加载"
  fi
  local url="http://$HOST:$PORT/healthz"
  if curl -s -m 2 "$url"; then
    echo
  else
    echo "healthz：无响应（$url）"
  fi
}

case "${1:-}" in
  install) cmd_install ;;
  uninstall) cmd_uninstall ;;
  start) cmd_start ;;
  stop) cmd_stop ;;
  restart) cmd_stop; cmd_start ;;
  status) cmd_status ;;
  logs) mkdir -p "$LOG_DIR"; touch "$LOG_DIR/out.log" "$LOG_DIR/err.log"; exec tail -n 50 -F "$LOG_DIR/out.log" "$LOG_DIR/err.log" ;;
  plist) render_plist ;;
  *) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
