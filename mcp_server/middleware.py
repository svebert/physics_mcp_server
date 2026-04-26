from __future__ import annotations

import logging
import json
import time
from collections import defaultdict

from fastapi import Request
from starlette.responses import Response


class RateLimitHook:
    """Tiny middleware-ready counter (no blocking yet, just telemetry hooks)."""

    def __init__(self, logger_name: str = "physics-mcp") -> None:
        self._counter: dict[str, int] = defaultdict(int)
        self._logger = logging.getLogger(logger_name)

    async def _read_request_body(self, request: Request) -> bytes:
        original_receive = request._receive  # type: ignore[attr-defined]
        buffered_messages: list[dict[str, object]] = []
        collected_body = bytearray()

        while True:
            message = await original_receive()
            buffered_messages.append(message)

            if message.get("type") != "http.request":
                break

            chunk = message.get("body", b"")
            if isinstance(chunk, (bytes, bytearray)):
                collected_body.extend(chunk)

            if not message.get("more_body", False):
                break

        async def receive() -> dict[str, object]:
            if buffered_messages:
                return buffered_messages.pop(0)
            return await original_receive()

        request._receive = receive  # type: ignore[attr-defined]
        return bytes(collected_body)

    def _decode_body(self, body: bytes) -> str:
        if not body:
            return ""
        try:
            parsed = json.loads(body.decode("utf-8"))
            return json.dumps(parsed, ensure_ascii=False)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return body.decode("utf-8", errors="replace")

    async def __call__(self, request: Request, call_next):
        client = request.client.host if request.client else "unknown"
        self._counter[client] += 1
        start = time.perf_counter()
        debug_enabled = self._logger.isEnabledFor(logging.DEBUG)

        request_body = b""
        if debug_enabled:
            request_body = await self._read_request_body(request)

        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1_000
        response.headers["X-Request-Time-Ms"] = f"{elapsed_ms:.2f}"
        response.headers["X-RateLimit-Hook"] = "enabled"

        if debug_enabled:
            content_type = response.headers.get("content-type", "")
            if "text/event-stream" in content_type:
                self._logger.debug(
                    "HTTP debug | method=%s path=%s client=%s status=%s elapsed_ms=%.2f request=%s response=<skipped stream>",
                    request.method,
                    request.url.path,
                    client,
                    response.status_code,
                    elapsed_ms,
                    self._decode_body(request_body),
                )
            else:
                body_chunks = [chunk async for chunk in response.body_iterator]
                response_body = b"".join(body_chunks)
                self._logger.debug(
                    "HTTP debug | method=%s path=%s client=%s status=%s elapsed_ms=%.2f request=%s response=%s",
                    request.method,
                    request.url.path,
                    client,
                    response.status_code,
                    elapsed_ms,
                    self._decode_body(request_body),
                    self._decode_body(response_body),
                )
                response = Response(
                    content=response_body,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.media_type,
                )
        return response

    def snapshot(self) -> dict[str, int]:
        return dict(self._counter)
