"""Local stand-in for the PaddleOCR cloud job API, backed by mlx4ocr (PP-OCRv6 on MLX).

UniReader submits a page image (``POST /api/v2/ocr/jobs``), polls the job, then downloads a
JSONL result. Submitting only queues the job; one worker thread runs the OCR, so the API
stays responsive while a page is being recognised.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import queue
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
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

logger = logging.getLogger(__name__)

MAX_JOBS = 64
MAX_PENDING = 32
JOB_PATH = "/api/v2/ocr/jobs"


@dataclass
class JobRecord:
    """One submitted job. The worker thread fills in the outcome and sets ``state`` last."""

    id: str
    filename: str
    submitted_at: float
    state: str = "pending"  # "pending" | "running" | "done" | "failed"
    word_boxes: bool = False
    image: np.ndarray | None = field(default=None, repr=False)  # dropped once processed
    width: int = 0
    height: int = 0
    text_count: int = 0
    elapsed_ms: int = 0
    error: str = ""
    payload: bytes = field(default=b"", repr=False)

    def summary(self) -> dict[str, object]:
        """Fields the dashboard shows (no image, no result body)."""
        return {
            "id": self.id,
            "filename": self.filename,
            "submitted_at": self.submitted_at,
            "state": self.state,
            "width": self.width,
            "height": self.height,
            "text_count": self.text_count,
            "elapsed_ms": self.elapsed_ms,
            "error": self.error,
        }


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
            self._jobs[record.id] = record
            while len(self._jobs) > self._capacity:
                self._jobs.popitem(last=False)

    def count_failure(self) -> None:
        with self._lock:
            self.failed += 1

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def recent(self) -> list[JobRecord]:
        with self._lock:
            return list(reversed(self._jobs.values()))


class Worker:
    """Runs every model operation on one dedicated thread.

    MLX streams are per-thread, so the thread that loads the model must also run inference
    and unload it. Doing that off the event loop keeps the API responsive while a page is
    being recognised; jobs wait in a bounded FIFO queue.
    """

    def __init__(
        self,
        lazy: LazyModel[PP_OCRv6],
        store: JobStore,
        idle_seconds: float,
        max_pending: int,
    ) -> None:
        self._lazy = lazy
        self._store = store
        self._queue: queue.Queue[JobRecord | None] = queue.Queue(maxsize=max_pending)
        self._stopping = threading.Event()
        self._interval = min(30.0, max(1.0, idle_seconds / 2)) if idle_seconds > 0 else 30.0
        self._thread = threading.Thread(target=self._run, name="ocr-worker", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stopping.set()
        with contextlib.suppress(queue.Full):
            self._queue.put_nowait(None)  # wake the thread if it is idle
        self._thread.join()

    def submit(self, record: JobRecord) -> None:
        """Queue a job; raises ``queue.Full`` when too many are waiting."""
        self._queue.put_nowait(record)

    def _run(self) -> None:
        while not self._stopping.is_set():
            try:
                record = self._queue.get(timeout=self._interval)
            except queue.Empty:
                self._lazy.unload_if_idle()
                continue
            if record is not None and not self._stopping.is_set():
                self._process(record)
        self._lazy.unload()

    def _process(self, record: JobRecord) -> None:
        record.state = "running"
        started = time.monotonic()
        outcome = "done"
        try:
            assert record.image is not None
            record.payload, record.text_count = build_result_line(
                self._lazy.get(), record.image, word_boxes=record.word_boxes
            )
        except Exception as error:
            logger.exception("job %s failed", record.id)
            record.error = str(error)[:120] or type(error).__name__
            outcome = "failed"
            self._store.count_failure()
        finally:
            record.image = None
            record.elapsed_ms = int((time.monotonic() - started) * 1000)
            self._lazy.touch()
            record.state = outcome


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
   ['处理中/排队',d.queued],['任务总数',d.total],['失败',d.failed]
  ].map(([k,v])=>`<div class="card"><b>${esc(String(v))}</b><span>${k}</span></div>`).join('');
  $('rows').innerHTML=d.jobs.map(j=>`<tr>
   <td>${new Date(j.submitted_at*1000).toLocaleString()}</td>
   <td>${esc(j.filename)}</td>
   <td class="${j.state}">${({pending:'排队中',running:'识别中',done:'完成',failed:'失败'})[j.state]}${j.error?' · '+esc(j.error):''}</td>
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

    max_pending = int(os.environ.get("OCR_MAX_PENDING", str(MAX_PENDING)))
    worker = Worker(lazy, store, idle_seconds, max_pending)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        worker.start()
        yield
        await asyncio.to_thread(worker.stop)

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
        jobs = store.recent()
        return {
            "queued": sum(job.state in ("pending", "running") for job in jobs),
            "loaded": lazy.loaded,
            "idle_seconds": idle_seconds,
            "variant": os.environ.get("OCR_VARIANT", "medium"),
            "uptime_seconds": time.time() - started_at,
            "total": store.total,
            "failed": store.failed,
            "jobs": [job.summary() for job in jobs],
        }

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
            id=uuid.uuid4().hex,
            filename=file.filename or "-",
            submitted_at=time.time(),
            word_boxes=bool(options.get("returnWordBox", False)),
        )
        try:
            record.image = await asyncio.to_thread(decode_image, await file.read())
        except HTTPException as error:
            record.state, record.error = "failed", str(error.detail)
            store.add(record)
            store.count_failure()
            raise
        record.height, record.width = record.image.shape[:2]
        try:
            worker.submit(record)
        except queue.Full as error:
            raise HTTPException(
                status_code=503, detail="too many pending jobs", headers={"Retry-After": "5"}
            ) from error
        store.add(record)
        return {"data": {"jobId": record.id}}

    @app.get(JOB_PATH + "/{job_id}")
    def status(job_id: str, request: Request) -> dict[str, object]:
        record = store.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="unknown job")
        state = record.state
        if state == "done":
            url = f"{str(request.base_url).rstrip('/')}{JOB_PATH}/{job_id}/result.jsonl"
            return {"data": {"state": state, "resultUrl": {"jsonUrl": url}}}
        if state == "failed":
            return {"data": {"state": state, "errorMsg": record.error}}
        return {"data": {"state": state}}

    @app.get(JOB_PATH + "/{job_id}/result.jsonl")
    def result(job_id: str) -> Response:
        record = store.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="unknown job")
        if record.state != "done":
            raise HTTPException(status_code=409, detail=f"job is {record.state}")
        return Response(record.payload, media_type="application/x-ndjson")

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
