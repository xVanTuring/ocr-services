#!/usr/bin/env bash
# 启动本地 OCR 服务（默认 127.0.0.1:8766，模型 medium；可用 OCR_VARIANT / OCR_HOST / OCR_PORT 覆盖）
set -euo pipefail
cd "$(dirname "$0")/service"
exec uv run ocr-service
