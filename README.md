# Local OCR Service

English | [简体中文](README.zh-CN.md)

Recognise text in images on an Apple Silicon Mac with PP-OCRv6 (MLX). The API matches PaddleOCR's cloud async job API (`/api/v2/ocr/jobs`), so existing clients such as UniReader only need a new base URL.

- The model loads on the first request and unloads after an idle period to free memory.
- A small web dashboard shows the service status and recent jobs.

## Layout

| Path | Description |
|---|---|
| `mlx4ocr/` | Recognition library (git submodule, a fork of mlx4ocr) |
| `service/` | FastAPI service (`src/ocr_service/`) and tests |
| `docs/API.md` | API reference for integrators (in Chinese) |
| `setup.sh` | Install dependencies and run tests |
| `run.sh` | Start the service in the foreground (for debugging) |
| `ocr-service.sh` | Manage the launchd background service |

## Install

```bash
git clone --recurse-submodules https://github.com/xVanTuring/ocr-services.git
# Already cloned without submodules: git submodule update --init
./setup.sh
```

`setup.sh` installs dependencies with uv and runs the tests for `mlx4ocr` and `service`. It needs network access.

## Run

Foreground (the first run downloads the model weights):

```bash
./run.sh
```

It listens on `http://127.0.0.1:8766` by default.

### Background service (launchd)

Run `./run.sh` once first and press Ctrl+C after `Uvicorn running` appears, so the weights are downloaded. The background service runs in offline mode and `install` refuses to proceed without the cache.

```bash
./ocr-service.sh install     # install and start; starts again at login
./ocr-service.sh status      # launchd state and /healthz
./ocr-service.sh restart
./ocr-service.sh stop
./ocr-service.sh logs        # follow the logs
./ocr-service.sh uninstall
```

Logs are in `~/Library/Logs/ocr-service/`.

## Configuration

Environment variables. For the background service they are written into the plist at `install` time, so run `install` again after changing them.

| Variable | Default | Description |
|---|---|---|
| `OCR_HOST` | `127.0.0.1` | Listen address |
| `OCR_PORT` | `8766` | Port |
| `OCR_VARIANT` | `medium` | Model variant |
| `OCR_IDLE_SECONDS` | `300` | Seconds of inactivity before the model is unloaded; `0` keeps it loaded |

## Web dashboard

Open `http://127.0.0.1:8766/` in a browser. It refreshes every 3 seconds and shows:

- Whether the model is loaded, uptime, model variant, total and failed job counts
- The latest 64 jobs: time, file name, state, image size, text line count, duration and a link to the result

The same data is available as JSON from `GET /api/stats`. Job records are kept in memory only and are cleared on restart.

## API

See [docs/API.md](docs/API.md) for details.

```bash
curl -s -F file=@page.png http://127.0.0.1:8766/api/v2/ocr/jobs
curl -s http://127.0.0.1:8766/healthz
```

The service has no authentication and listens on localhost only by default. Add access control before exposing it to a network.

## Tests

```bash
cd service && uv run pytest -q
```

## License

GPL-3.0, see [LICENSE](LICENSE).
