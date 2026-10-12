"""API shape tests with a stub OCR pipeline (no model download)."""

from __future__ import annotations

import io
import json
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
from fastapi.testclient import TestClient

from mlx4ocr.pipeline.crop import char_boxes_in_image
from mlx4ocr.types import BoundingBox, OCRResult, TextDetection, TextRecognition
from ocr_service.app import JOB_PATH, create_app


class StubOCR:
    def predict(self, image: np.ndarray, *, char_boxes: bool = False) -> SimpleNamespace:
        detection = TextDetection(
            box=BoundingBox(points=((10.0, 10.0), (110.0, 10.0), (110.0, 30.0), (10.0, 30.0))),
            score=0.9,
        )
        recognition = TextRecognition(
            text="你好", score=0.9, chars=("你", "好"), char_centers=(0.25, 0.75), char_width=0.1
        )
        boxes = (char_boxes_in_image(detection, recognition),) if char_boxes else None
        result = OCRResult.from_ppocrv6((detection,), (recognition,), char_boxes=boxes)
        return SimpleNamespace(result=result)

    def close(self) -> None:
        pass


def _png() -> bytes:
    ok, buffer = cv2.imencode(".png", np.full((40, 120, 3), 255, dtype=np.uint8))
    assert ok
    return buffer.tobytes()


def _submit(client: TestClient) -> str:
    response = client.post(
        JOB_PATH, files={"file": ("page.png", io.BytesIO(_png()), "image/png")}
    )
    assert response.status_code == 200, response.text
    return response.json()["data"]["jobId"]


def wait_done(client: TestClient, job_id: str, timeout: float = 5.0) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        data = client.get(f"{JOB_PATH}/{job_id}").json()["data"]
        if data["state"] in ("done", "failed"):
            return data
        time.sleep(0.01)
    raise AssertionError("job did not finish")


def _run(client: TestClient, options: dict[str, object]) -> dict[str, object]:
    submit = client.post(
        JOB_PATH,
        data={"model": "PP-OCRv6", "optionalPayload": json.dumps(options)},
        files={"file": ("page.png", io.BytesIO(_png()), "image/png")},
    )
    assert submit.status_code == 200
    status = wait_done(client, submit.json()["data"]["jobId"])
    assert status["state"] == "done"
    body = client.get(status["resultUrl"]["jsonUrl"]).text
    return json.loads(body.strip().splitlines()[0])["result"]["ocrResults"][0]["prunedResult"]


def test_job_roundtrip_with_word_boxes() -> None:
    with TestClient(create_app(StubOCR, idle_seconds=0)) as client:  # type: ignore[arg-type]
        pruned = _run(client, {"returnWordBox": True})

    assert pruned["width"] == 120 and pruned["height"] == 40
    assert pruned["rec_texts"] == ["你好"]
    assert pruned["rec_boxes"] == [[10.0, 10.0, 110.0, 30.0]]
    assert pruned["text_word"] == [["你", "好"]]
    assert pruned["text_word_boxes"] == [[[30.0, 10.0, 40.0, 30.0], [80.0, 10.0, 90.0, 30.0]]]


def test_job_without_word_boxes_omits_them() -> None:
    with TestClient(create_app(StubOCR, idle_seconds=0)) as client:  # type: ignore[arg-type]
        pruned = _run(client, {})
    assert "text_word" not in pruned


def test_dashboard_lists_jobs() -> None:
    with TestClient(create_app(StubOCR, idle_seconds=0)) as client:  # type: ignore[arg-type]
        assert "OCR Service" in client.get("/").text
        _run(client, {})
        client.post(JOB_PATH, files={"file": ("bad.png", io.BytesIO(b"x"), "image/png")})
        stats = client.get("/api/stats").json()

    assert stats["total"] == 2 and stats["failed"] == 1
    failed, done = stats["jobs"]
    assert failed["state"] == "failed" and failed["filename"] == "bad.png"
    assert done["state"] == "done" and done["text_count"] == 1
    assert done["width"] == 120 and "payload" not in done


class GatedOCR(StubOCR):
    """Blocks inside predict until released, to observe the API mid-inference."""

    started = threading.Event()
    release = threading.Event()

    def predict(self, image: np.ndarray, *, char_boxes: bool = False) -> SimpleNamespace:
        type(self).started.set()
        assert type(self).release.wait(5)
        return super().predict(image, char_boxes=char_boxes)


def test_api_stays_responsive_while_a_job_is_running() -> None:
    GatedOCR.started.clear()
    GatedOCR.release.clear()
    with TestClient(create_app(GatedOCR, idle_seconds=0)) as client:  # type: ignore[arg-type]
        first, second = _submit(client), _submit(client)
        assert GatedOCR.started.wait(5)

        assert client.get("/healthz").status_code == 200
        assert client.get(f"{JOB_PATH}/{first}").json()["data"]["state"] == "running"
        assert client.get(f"{JOB_PATH}/{second}").json()["data"]["state"] == "pending"
        assert client.get(f"{JOB_PATH}/{first}/result.jsonl").status_code == 409
        assert client.get("/api/stats").json()["queued"] == 2

        GatedOCR.release.set()
        assert wait_done(client, first)["state"] == "done"
        assert wait_done(client, second)["state"] == "done"
        assert client.get("/api/stats").json()["queued"] == 0


def test_full_queue_returns_503(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("OCR_MAX_PENDING", "1")
    GatedOCR.started.clear()
    GatedOCR.release.clear()
    with TestClient(create_app(GatedOCR, idle_seconds=0)) as client:  # type: ignore[arg-type]
        _submit(client)
        assert GatedOCR.started.wait(5)  # running, queue empty again
        _submit(client)  # fills the queue
        rejected = client.post(
            JOB_PATH, files={"file": ("page.png", io.BytesIO(_png()), "image/png")}
        )
        assert rejected.status_code == 503
        GatedOCR.release.set()


def test_model_error_marks_job_failed() -> None:
    class Boom(StubOCR):
        def predict(self, image: np.ndarray, *, char_boxes: bool = False) -> SimpleNamespace:
            raise RuntimeError("boom")

    with TestClient(create_app(Boom, idle_seconds=0)) as client:  # type: ignore[arg-type]
        job_id = _submit(client)
        status = wait_done(client, job_id)
        assert status == {"state": "failed", "errorMsg": "boom"}
        assert client.get(f"{JOB_PATH}/{job_id}/result.jsonl").status_code == 409
        assert client.get("/api/stats").json()["failed"] == 1


def test_unknown_job_is_404() -> None:
    with TestClient(create_app(StubOCR, idle_seconds=0)) as client:  # type: ignore[arg-type]
        assert client.get(f"{JOB_PATH}/nope").status_code == 404
