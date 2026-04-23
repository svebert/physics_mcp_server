from __future__ import annotations

import time
from collections import defaultdict

from fastapi import Request


class RateLimitHook:
    """Tiny middleware-ready counter (no blocking yet, just telemetry hooks)."""

    def __init__(self) -> None:
        self._counter: dict[str, int] = defaultdict(int)

    async def __call__(self, request: Request, call_next):
        client = request.client.host if request.client else "unknown"
        self._counter[client] += 1
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1_000
        response.headers["X-Request-Time-Ms"] = f"{elapsed_ms:.2f}"
        response.headers["X-RateLimit-Hook"] = "enabled"
        return response

    def snapshot(self) -> dict[str, int]:
        return dict(self._counter)
