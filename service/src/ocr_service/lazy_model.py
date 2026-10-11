"""Load the OCR model on first use and drop it again after a quiet period.

Everything here must run on one thread: MLX streams are per-thread, so the thread that
loads the model is the one that has to run inference and unload it. The service calls it
only from the event loop thread.
"""

from __future__ import annotations

import gc
import logging
import time
from collections.abc import Callable
from typing import Protocol

logger = logging.getLogger(__name__)


class Closable(Protocol):
    def close(self) -> None: ...


class LazyModel[M: Closable]:
    """Holds at most one loaded model; ``idle_seconds <= 0`` keeps it loaded forever."""

    def __init__(
        self,
        loader: Callable[[], M],
        idle_seconds: float,
        after_unload: Callable[[], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._loader = loader
        self._idle_seconds = idle_seconds
        self._after_unload = after_unload
        self._clock = clock
        self._model: M | None = None
        self._last_used = 0.0

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def get(self) -> M:
        """Return the model, loading it if needed. Counts as use."""
        if self._model is None:
            started = self._clock()
            self._model = self._loader()
            logger.info("model loaded in %.1fs", self._clock() - started)
        self._last_used = self._clock()
        return self._model

    def touch(self) -> None:
        """Restart the idle timer, e.g. after a long inference."""
        self._last_used = self._clock()

    def unload_if_idle(self) -> bool:
        """Unload when unused for ``idle_seconds``; returns whether it unloaded."""
        if self._model is None or self._idle_seconds <= 0:
            return False
        if self._clock() - self._last_used < self._idle_seconds:
            return False
        self.unload()
        return True

    def unload(self) -> None:
        if self._model is None:
            return
        model, self._model = self._model, None
        model.close()
        del model
        gc.collect()
        if self._after_unload is not None:
            self._after_unload()
        logger.info("model unloaded")
