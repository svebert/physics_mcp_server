from __future__ import annotations

import logging
from datetime import datetime, timezone

from mcp_server.logging_config import configure_logging


def test_configure_logging_uses_daily_file_name(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PHYSICS_MCP_LOG_DIR", str(tmp_path))

    configure_logging(service_name="physics-mcp", app_logger_name="physics-mcp", level=logging.INFO)

    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    assert (tmp_path / f"physics-mcp-{day}.log").exists()
