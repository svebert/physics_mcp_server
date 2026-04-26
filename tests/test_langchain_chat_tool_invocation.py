from __future__ import annotations

from client.langchain_chat import _invoke_tool


class _AsyncOnlyTool:
    async def ainvoke(self, args):
        return {"ok": True, "mode": "async", "args": args}

    def invoke(self, _args):  # pragma: no cover - should not be called
        raise AssertionError("sync invoke should not be used for async-capable tools")


class _SyncOnlyTool:
    def invoke(self, args):
        return {"ok": True, "mode": "sync", "args": args}


def test_invoke_tool_prefers_async_invoke_when_available() -> None:
    payload = {"question": "test"}
    result = _invoke_tool(_AsyncOnlyTool(), payload)
    assert result == {"ok": True, "mode": "async", "args": payload}


def test_invoke_tool_uses_sync_invoke_when_async_missing() -> None:
    payload = {"question": "test"}
    result = _invoke_tool(_SyncOnlyTool(), payload)
    assert result == {"ok": True, "mode": "sync", "args": payload}
