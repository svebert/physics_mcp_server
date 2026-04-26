from __future__ import annotations

import logging
import logging.config
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Any


def _build_uvicorn_log_config(log_file: Path, level: int, app_logger_name: str) -> dict[str, Any]:
    log_level_name = logging.getLevelName(level)
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(asctime)s %(levelprefix)s %(message)s",
                "datefmt": "%Y-%m-%dT%H:%M:%S%z",
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(asctime)s %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
                "datefmt": "%Y-%m-%dT%H:%M:%S%z",
            },
        },
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "stream": "ext://sys.stdout",
            },
            "access": {
                "class": "logging.StreamHandler",
                "formatter": "access",
                "stream": "ext://sys.stdout",
            },
            "file_default": {
                "class": "logging.FileHandler",
                "formatter": "default",
                "filename": str(log_file),
                "encoding": "utf-8",
            },
            "file_access": {
                "class": "logging.FileHandler",
                "formatter": "access",
                "filename": str(log_file),
                "encoding": "utf-8",
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["default", "file_default"], "level": log_level_name, "propagate": False},
            "uvicorn.error": {"handlers": ["default", "file_default"], "level": log_level_name, "propagate": False},
            "uvicorn.access": {"handlers": ["access", "file_access"], "level": log_level_name, "propagate": False},
            "mcp": {"handlers": ["default", "file_default"], "level": log_level_name, "propagate": False},
            "httpx": {"handlers": ["default", "file_default"], "level": log_level_name, "propagate": False},
            "httpcore": {"handlers": ["default", "file_default"], "level": log_level_name, "propagate": False},
            app_logger_name: {"handlers": ["default", "file_default"], "level": log_level_name, "propagate": False},
        },
    }


def configure_logging(
    *,
    service_name: str = "physics-mcp",
    app_logger_name: str = "physics-mcp",
    level: int = logging.INFO,
) -> dict[str, Any]:
    log_dir = Path(os.getenv("PHYSICS_MCP_LOG_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    current_day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_file = log_dir / f"{service_name}-{current_day}.log"

    logging_config = _build_uvicorn_log_config(log_file=log_file, level=level, app_logger_name=app_logger_name)
    logging.config.dictConfig(logging_config)
    return logging_config
