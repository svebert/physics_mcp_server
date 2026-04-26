from __future__ import annotations

import logging
import json
import os
import time
import uuid
from collections import defaultdict

from fastapi import Request
from starlette.responses import Response


class RateLimitHook:
    """Tiny middleware-ready counter (no blocking yet, just telemetry hooks)."""

    def __init__(self, logger_name: str = "physics-mcp") -> None:
        self._counter: dict[str, int] = defaultdict(int)
        self._logger = logging.getLogger(logger_name)
        self._max_payload_chars = int(os.getenv("PHYSICS_MCP_MAX_LOG_PAYLOAD_CHARS", "2000"))

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

    def _truncate(self, payload: str) -> str:
        if len(payload) <= self._max_payload_chars:
            return payload
        return f"{payload[:self._max_payload_chars]}...<truncated {len(payload) - self._max_payload_chars} chars>"

    async def __call__(self, request: Request, call_next):
        client = request.client.host if request.client else "unknown"
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        user_agent = request.headers.get("user-agent", "unknown")
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
        response.headers["X-Request-Id"] = request_id

        if debug_enabled:
            content_type = response.headers.get("content-type", "")
            inbound_summary = (
                "HTTP inbound | request_id=%s direction=client->physics-postdoc-mcp "
                "peer=%s method=%s path=%s status=%s elapsed_ms=%.2f user_agent=%s"
            )
            if "text/event-stream" in content_type:
                request_payload = self._truncate(self._decode_body(request_body))
                self._logger.debug(
                    inbound_summary,
                    request_id,
                    client,
                    request.method,
                    request.url.path,
                    response.status_code,
                    elapsed_ms,
                    user_agent,
                )
                self._logger.debug("  ├─ request_payload=%s", request_payload)
                self._logger.debug("  └─ response_payload=<skipped stream>")
            else:
                body_chunks = [chunk async for chunk in response.body_iterator]
                response_body = b"".join(body_chunks)
                request_payload = self._truncate(self._decode_body(request_body))
                response_payload = self._truncate(self._decode_body(response_body))
                self._logger.debug(
                    inbound_summary,
                    request_id,
                    client,
                    request.method,
                    request.url.path,
                    response.status_code,
                    elapsed_ms,
                    user_agent,
                )
                self._logger.debug("  ├─ request_payload=%s", request_payload)
                self._logger.debug("  └─ response_payload=%s", response_payload)
                response = Response(
                    content=response_body,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.media_type,
                )
        return response

    def snapshot(self) -> dict[str, int]:
        return dict(self._counter)
