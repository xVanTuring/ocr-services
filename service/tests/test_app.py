"""API shape tests with a stub OCR pipeline (no model download)."""

from __future__ import annotations

import io
import json
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


def _run(client: TestClient, options: dict[str, object]) -> dict[str, object]:
    submit = client.post(
        JOB_PATH,
        data={"model": "PP-OCRv6", "optionalPayload": json.dumps(options)},
        files={"file": ("page.png", io.BytesIO(_png()), "image/png")},
    )
    assert submit.status_code == 200
    job_id = submit.json()["data"]["jobId"]
    status = client.get(f"{JOB_PATH}/{job_id}").json()["data"]
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


def test_unknown_job_is_404() -> None:
    with TestClient(create_app(StubOCR, idle_seconds=0)) as client:  # type: ignore[arg-type]
        assert client.get(f"{JOB_PATH}/nope").status_code == 404
