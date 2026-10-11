"""Lazy load / idle unload behaviour, with a fake clock and a counting stub."""

from __future__ import annotations

from fastapi.testclient import TestClient

from ocr_service.app import create_app
from ocr_service.lazy_model import LazyModel
from tests.test_app import StubOCR, _run


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class CountingModel:
    loads = 0
    closes = 0

    def __init__(self) -> None:
        type(self).loads += 1

    def close(self) -> None:
        type(self).closes += 1


def _holder(idle: float) -> tuple[LazyModel[CountingModel], FakeClock]:
    CountingModel.loads = CountingModel.closes = 0
    clock = FakeClock()
    return LazyModel(CountingModel, idle, clock=clock), clock


def test_loads_only_on_first_get() -> None:
    holder, _ = _holder(60)
    assert not holder.loaded and CountingModel.loads == 0
    first = holder.get()
    assert holder.get() is first
    assert CountingModel.loads == 1


def test_unloads_after_idle_and_reloads_on_next_use() -> None:
    holder, clock = _holder(60)
    holder.get()

    clock.now = 59
    assert not holder.unload_if_idle() and holder.loaded

    clock.now = 60
    assert holder.unload_if_idle()
    assert not holder.loaded and CountingModel.closes == 1

    holder.get()
    assert CountingModel.loads == 2


def test_use_and_touch_restart_the_idle_timer() -> None:
    holder, clock = _holder(60)
    holder.get()
    clock.now = 50
    holder.touch()
    clock.now = 100
    assert not holder.unload_if_idle()
    clock.now = 110
    assert holder.unload_if_idle()


def test_zero_idle_never_unloads() -> None:
    holder, clock = _holder(0)
    holder.get()
    clock.now = 10**9
    assert not holder.unload_if_idle() and holder.loaded


def test_unload_without_model_is_noop() -> None:
    holder, _ = _holder(60)
    holder.unload()
    assert CountingModel.closes == 0


def test_app_loads_on_first_request_not_at_startup() -> None:
    loads: list[int] = []

    def loader() -> StubOCR:
        loads.append(1)
        return StubOCR()

    with TestClient(create_app(loader, idle_seconds=0)) as client:
        assert loads == []
        assert client.get("/healthz").json()["loaded"] is False
        _run(client, {})
        _run(client, {})
        assert loads == [1]
        assert client.get("/healthz").json()["loaded"] is True


def test_after_unload_runs_once_per_unload() -> None:
    calls: list[int] = []
    CountingModel.loads = CountingModel.closes = 0
    holder = LazyModel(CountingModel, 60, after_unload=lambda: calls.append(1))
    holder.unload()  # nothing loaded yet
    assert calls == []
    holder.get()
    holder.unload()
    assert calls == [1]
