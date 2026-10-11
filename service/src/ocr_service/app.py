"""Local stand-in for the PaddleOCR cloud job API, backed by mlx4ocr (PP-OCRv6 on MLX).

UniReader submits a page image (``POST /api/v2/ocr/jobs``), polls the job, then downloads a
JSONL result. OCR runs inside the submit call, so a job is already ``done`` on first poll.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import cv2
import mlx.core as mx
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response

from mlx4ocr import PP_OCRv6
from mlx4ocr.output import to_paddlex_res
from ocr_service.lazy_model import LazyModel

MAX_JOBS = 64
JOB_PATH = "/api/v2/ocr/jobs"


class JobStore:
    """Bounded in-memory map of job id -> result JSONL."""

    def __init__(self, capacity: int = MAX_JOBS) -> None:
        self._capacity = capacity
        self._jobs: OrderedDict[str, bytes] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, payload: bytes) -> str:
        job_id = uuid.uuid4().hex
        with self._lock:
            self._jobs[job_id] = payload
            while len(self._jobs) > self._capacity:
                self._jobs.popitem(last=False)
        return job_id

    def get(self, job_id: str) -> bytes | None:
        with self._lock:
            return self._jobs.get(job_id)


def decode_image(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="cannot decode image")
    return image


def build_result_line(ocr: PP_OCRv6, image: np.ndarray, *, word_boxes: bool) -> bytes:
    height, width = image.shape[:2]
    result = ocr.predict(image, char_boxes=word_boxes).result
    pruned = to_paddlex_res(result, word_boxes=word_boxes)
    pruned["width"] = width
    pruned["height"] = height
    line = {"result": {"ocrResults": [{"prunedResult": pruned}]}}
    return (json.dumps(line, ensure_ascii=False) + "\n").encode("utf-8")


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
        image = decode_image(await file.read())
        payload = build_result_line(
            lazy.get(),
            image,
            word_boxes=bool(options.get("returnWordBox", False)),
        )
        lazy.touch()
        return {"data": {"jobId": store.put(payload)}}

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
