from __future__ import annotations

from pathlib import Path

from mcp_server import env_loader


def test_load_project_env_loads_root_then_service_override(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    service_dir = project_root / "smart_mcp_server"
    service_dir.mkdir(parents=True)

    (project_root / ".env").write_text("LLM_API_KEY=root-key\nPHYSICS_MCP_URL=http://root\n")
    (service_dir / ".env").write_text("LLM_API_KEY=service-key\nSMART_ONLY=value\n")

    monkeypatch.setattr(env_loader, "PROJECT_ROOT", project_root)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("PHYSICS_MCP_URL", raising=False)
    monkeypatch.delenv("SMART_ONLY", raising=False)

    env_loader.load_project_env(service_dir)

    assert env_loader.os.getenv("LLM_API_KEY") == "service-key"
    assert env_loader.os.getenv("PHYSICS_MCP_URL") == "http://root"
    assert env_loader.os.getenv("SMART_ONLY") == "value"


def test_load_project_env_keeps_explicit_environment(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    service_dir = project_root / "mcp_server"
    service_dir.mkdir(parents=True)

    (project_root / ".env").write_text("LLM_API_KEY=root-key\n")
    (service_dir / ".env").write_text("LLM_API_KEY=service-key\n")

    monkeypatch.setattr(env_loader, "PROJECT_ROOT", project_root)
    monkeypatch.setenv("LLM_API_KEY", "explicit-key")

    env_loader.load_project_env(service_dir)

    assert env_loader.os.getenv("LLM_API_KEY") == "explicit-key"
