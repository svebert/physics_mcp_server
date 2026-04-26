from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_project_env(service_dir: Path) -> None:
    """Load shared root env plus optional service-local overrides.

    Precedence (highest to lowest):
    1) existing process environment variables
    2) <service_dir>/.env (optional)
    3) <project>/.env (shared defaults)
    """
    preexisting = set(os.environ.keys())

    root_env = PROJECT_ROOT / ".env"
    if root_env.exists():
        load_dotenv(root_env, override=False)

    service_env = service_dir / ".env"
    if service_env.exists():
        for key, value in dotenv_values(service_env).items():
            if value is None:
                continue
            if key in preexisting:
                continue
            os.environ[key] = value
