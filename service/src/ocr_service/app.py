"""Local stand-in for the PaddleOCR cloud job API, backed by mlx4ocr (PP-OCRv6 on MLX).

UniReader submits a page image (``POST /api/v2/ocr/jobs``), polls the job, then downloads a
JSONL result. OCR runs inside the submit call, so a job is already ``done`` on first poll.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Annotated

import cv2
import mlx.core as mx
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response

from mlx4ocr import PP_OCRv6
from mlx4ocr.output import to_paddlex_res
from ocr_service.lazy_model import LazyModel

MAX_JOBS = 64
JOB_PATH = "/api/v2/ocr/jobs"


@dataclass
class JobRecord:
    """What the dashboard shows about one submitted job."""

    id: str
    filename: str
    submitted_at: float
    state: str = "done"  # "done" | "failed"
    width: int = 0
    height: int = 0
    text_count: int = 0
    elapsed_ms: int = 0
    error: str = ""
    payload: bytes = field(default=b"", repr=False)


class JobStore:
    """Bounded in-memory map of job id -> record (with result JSONL)."""

    def __init__(self, capacity: int = MAX_JOBS) -> None:
        self._capacity = capacity
        self._jobs: OrderedDict[str, JobRecord] = OrderedDict()
        self._lock = threading.Lock()
        self.total = 0
        self.failed = 0

    def add(self, record: JobRecord) -> None:
        with self._lock:
            self.total += 1
            if record.state == "failed":
                self.failed += 1
            self._jobs[record.id] = record
            while len(self._jobs) > self._capacity:
                self._jobs.popitem(last=False)

    def get(self, job_id: str) -> bytes | None:
        with self._lock:
            record = self._jobs.get(job_id)
        return record.payload if record and record.state == "done" else None

    def recent(self) -> list[JobRecord]:
        with self._lock:
            return list(reversed(self._jobs.values()))


def decode_image(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="cannot decode image")
    return image


def build_result_line(ocr: PP_OCRv6, image: np.ndarray, *, word_boxes: bool) -> tuple[bytes, int]:
    """Return the result JSONL line and the number of recognised text lines."""
    height, width = image.shape[:2]
    result = ocr.predict(image, char_boxes=word_boxes).result
    pruned = to_paddlex_res(result, word_boxes=word_boxes)
    pruned["width"] = width
    pruned["height"] = height
    line = {"result": {"ocrResults": [{"prunedResult": pruned}]}}
    payload = (json.dumps(line, ensure_ascii=False) + "\n").encode("utf-8")
    return payload, len(pruned.get("rec_texts", []))


DASHBOARD_HTML = """<!doctype html>
<html lang="zh-Hans"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OCR Service</title>
<style>
:root{color-scheme:light dark;font:14px/1.5 -apple-system,system-ui,sans-serif}
body{max-width:860px;margin:0 auto;padding:16px}
h1{font-size:18px;margin:0 0 12px}
.cards{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px}
.card{flex:1 1 120px;border:1px solid #8884;border-radius:8px;padding:8px 12px}
.card b{display:block;font-size:20px}
.card span{opacity:.65;font-size:12px}
table{width:100%;border-collapse:collapse}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid #8883;white-space:nowrap}
.failed{color:#d33}
.wrap{overflow-x:auto}
</style></head><body>
<h1>OCR Service</h1>
<div class="cards" id="cards"></div>
<div class="wrap"><table><thead><tr>
<th>时间</th><th>文件</th><th>状态</th><th>尺寸</th><th>文本行</th><th>耗时</th><th>结果</th>
</tr></thead><tbody id="rows"></tbody></table></div>
<script>
const $=id=>document.getElementById(id);
function dur(s){s=Math.floor(s);const h=Math.floor(s/3600),m=Math.floor(s%3600/60);
return h?`${h}时${m}分`:m?`${m}分${s%60}秒`:`${s}秒`}
function esc(t){return t.replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
async function load(){
 try{
  const d=await (await fetch('api/stats')).json();
  $('cards').innerHTML=[
   ['模型',d.loaded?'已加载':'未加载'],['运行时长',dur(d.uptime_seconds)],
   ['变体',d.variant],['空闲卸载',d.idle_seconds?d.idle_seconds+' 秒':'不卸载'],
   ['任务总数',d.total],['失败',d.failed]
  ].map(([k,v])=>`<div class="card"><b>${esc(String(v))}</b><span>${k}</span></div>`).join('');
  $('rows').innerHTML=d.jobs.map(j=>`<tr>
   <td>${new Date(j.submitted_at*1000).toLocaleString()}</td>
   <td>${esc(j.filename)}</td>
   <td class="${j.state}">${j.state==='done'?'完成':'失败'}${j.error?' · '+esc(j.error):''}</td>
   <td>${j.width}×${j.height}</td><td>${j.text_count}</td><td>${j.elapsed_ms} ms</td>
   <td>${j.state==='done'?`<a href="api/v2/ocr/jobs/${j.id}/result.jsonl">JSONL</a>`:''}</td></tr>`).join('');
 }catch(e){}
}
load();setInterval(load,3000);
</script></body></html>
"""


def default_loader() -> PP_OCRv6:
    return PP_OCRv6.from_hub(os.environ.get("OCR_VARIANT", "medium"))


def create_app(
    loader: Callable[[], PP_OCRv6] = default_loader,
    idle_seconds: float | None = None,
) -> FastAPI:
    """Build the app. The model loads on the first request and unloads after ``idle_seconds``.

    ``idle_seconds`` defaults to ``OCR_IDLE_SECONDS`` (300); 0 keeps the model loaded.
    """
    if idle_seconds is None:
        idle_seconds = float(os.environ.get("OCR_IDLE_SECONDS", "300"))
    store = JobStore()
    started_at = time.time()
    # Freed weights park in MLX's buffer cache; clear it so the memory really goes back.
    lazy = LazyModel(loader, idle_seconds, after_unload=mx.clear_cache)

    async def unload_when_idle() -> None:
        # Runs on the event loop thread, never during an inference (those block the loop),
        # which is also the thread the model was loaded on.
        interval = min(30.0, max(1.0, idle_seconds / 2))
        while True:
            await asyncio.sleep(interval)
            lazy.unload_if_idle()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        watcher = asyncio.create_task(unload_when_idle()) if idle_seconds > 0 else None
        yield
        if watcher is not None:
            watcher.cancel()
        lazy.unload()

    app = FastAPI(lifespan=lifespan)
    app.state.lazy = lazy

    @app.get("/healthz")
    def health() -> dict[str, object]:
        return {"loaded": lazy.loaded, "idle_seconds": idle_seconds}

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        return DASHBOARD_HTML

    @app.get("/api/stats")
    def stats() -> dict[str, object]:
        return {
            "loaded": lazy.loaded,
            "idle_seconds": idle_seconds,
            "variant": os.environ.get("OCR_VARIANT", "medium"),
            "uptime_seconds": time.time() - started_at,
            "total": store.total,
            "failed": store.failed,
            "jobs": [
                {k: v for k, v in asdict(job).items() if k != "payload"}
                for job in store.recent()
            ],
        }

    # `async def` on purpose: MLX streams are per-thread, so inference must run on the same
    # thread that loaded the model (the event loop's). Requests simply queue behind each other.
    @app.post(JOB_PATH)
    async def submit(
        request: Request,
        file: Annotated[UploadFile, File()],
        model: Annotated[str, Form()] = "PP-OCRv6",
        optionalPayload: Annotated[str, Form()] = "{}",  # noqa: N803 - cloud field name
    ) -> dict[str, object]:
        try:
            options = json.loads(optionalPayload or "{}")
        except json.JSONDecodeError as error:
            raise HTTPException(status_code=400, detail="optionalPayload is not JSON") from error
        record = JobRecord(
            id=uuid.uuid4().hex, filename=file.filename or "-", submitted_at=time.time()
        )
        started = time.monotonic()
        try:
            image = decode_image(await file.read())
            record.height, record.width = image.shape[:2]
            record.payload, record.text_count = build_result_line(
                lazy.get(),
                image,
                word_boxes=bool(options.get("returnWordBox", False)),
            )
        except Exception as error:
            record.state = "failed"
            record.error = str(getattr(error, "detail", error))[:120]
            raise
        finally:
            record.elapsed_ms = int((time.monotonic() - started) * 1000)
            store.add(record)
        lazy.touch()
        return {"data": {"jobId": record.id}}

    @app.get(JOB_PATH + "/{job_id}")
    def status(job_id: str, request: Request) -> dict[str, object]:
        if store.get(job_id) is None:
            raise HTTPException(status_code=404, detail="unknown job")
        url = f"{str(request.base_url).rstrip('/')}{JOB_PATH}/{job_id}/result.jsonl"
        return {"data": {"state": "done", "resultUrl": {"jsonUrl": url}}}

    @app.get(JOB_PATH + "/{job_id}/result.jsonl")
    def result(job_id: str) -> Response:
        payload = store.get(job_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="unknown job")
        return Response(payload, media_type="application/x-ndjson")

    return app


def main() -> None:
    import uvicorn

    # mlx4ocr caches the Paddle pretrained checkpoint under a path relative to the working
    # directory, so pin it: otherwise the weights are re-downloaded per launch directory.
    os.chdir(Path(__file__).resolve().parents[2])
    host = os.environ.get("OCR_HOST", "127.0.0.1")
    port = int(os.environ.get("OCR_PORT", "8766"))
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    main()
