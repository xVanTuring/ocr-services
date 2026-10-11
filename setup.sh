#!/usr/bin/env bash
# 安装 fork 的开发依赖并跑测试（需要联网下载 Python 3.12+ 与依赖）
set -euo pipefail
cd "$(dirname "$0")/mlx4ocr"
uv sync --group dev
uv run ruff check . && uv run ruff format --check .
uv run pytest -q
cd ../service && uv sync --group dev && uv run pytest -q
