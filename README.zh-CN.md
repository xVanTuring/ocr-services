# 本地 OCR 服务

[English](README.md) | 简体中文

在 Apple Silicon 的 Mac 上，用 PP-OCRv6（MLX）识别图片文字。接口与 PaddleOCR 云端的异步任务接口（`/api/v2/ocr/jobs`）一致，已有客户端（如 UniReader）只需要改地址。

- 模型在第一次请求时加载，空闲一段时间后自动卸载，释放内存。
- 自带一个简单的 Web 面板，可查看服务状态和最近的任务。

## 目录

| 路径 | 说明 |
|---|---|
| `mlx4ocr/` | 识别库（git 子模块，fork 自 mlx4ocr） |
| `service/` | FastAPI 服务（`src/ocr_service/`）及测试 |
| `docs/API.md` | 接口文档（给接入方看） |
| `setup.sh` | 安装依赖并跑测试 |
| `run.sh` | 前台启动服务（调试用） |
| `ocr-service.sh` | 管理 launchd 后台服务 |

## 安装

```bash
git clone --recurse-submodules https://github.com/xVanTuring/ocr-services.git
# 已克隆但没拉子模块：git submodule update --init
./setup.sh
```

`setup.sh` 用 uv 安装依赖，并跑 `mlx4ocr` 和 `service` 的测试，需要联网。

## 运行

前台启动（第一次运行会下载模型权重）：

```bash
./run.sh
```

默认监听 `http://127.0.0.1:8766`。

### 后台常驻（launchd）

先执行一次 `./run.sh`，等出现 `Uvicorn running` 后 Ctrl+C，让权重下载完。后台服务运行在离线模式，没有缓存会拒绝安装。

```bash
./ocr-service.sh install     # 安装并启动，登录后自动启动
./ocr-service.sh status      # 查看状态和 /healthz
./ocr-service.sh restart     # 重启
./ocr-service.sh stop        # 停止
./ocr-service.sh logs        # 跟随日志
./ocr-service.sh uninstall   # 卸载
```

日志在 `~/Library/Logs/ocr-service/`。

## 配置

环境变量。后台服务是在 `install` 时写进 plist 的，修改后需要重新 `install`。

| 变量 | 默认 | 说明 |
|---|---|---|
| `OCR_HOST` | `127.0.0.1` | 监听地址 |
| `OCR_PORT` | `8766` | 端口 |
| `OCR_VARIANT` | `medium` | 模型变体 |
| `OCR_IDLE_SECONDS` | `300` | 空闲多少秒后卸载模型，`0` 表示不卸载 |

## Web 面板

浏览器打开 `http://127.0.0.1:8766/`，每 3 秒自动刷新：

- 模型是否已加载、运行时长、模型变体、任务总数、失败数
- 最近 64 个任务：时间、文件名、状态、图片尺寸、文本行数、耗时、结果下载链接

同样的数据可从 `GET /api/stats` 取得（JSON）。任务记录只保存在内存里，重启后清空。

## 接口

详见 [docs/API.md](docs/API.md)。

```bash
curl -s -F file=@page.png http://127.0.0.1:8766/api/v2/ocr/jobs
curl -s http://127.0.0.1:8766/healthz
```

服务没有认证，默认只监听本机。改成监听局域网前请先考虑访问控制。

## 测试

```bash
cd service && uv run pytest -q
```

## 许可

GPL-3.0，见 [LICENSE](LICENSE)。
