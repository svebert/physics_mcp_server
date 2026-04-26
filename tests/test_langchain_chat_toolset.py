from __future__ import annotations

from client.langchain_chat import TOOLSET_OPTIONS


EXPECTED_MCP_TOOLS = {
    "solve_beam_case_tool",
    "get_supported_cases_tool",
    "get_model_assumptions_tool",
}


def test_mcp_toolsets_use_registered_mcp_tool_names() -> None:
    for key in ("1", "3", "4", "5"):
        tools = set(TOOLSET_OPTIONS[key]["tools"])
        assert EXPECTED_MCP_TOOLS.issubset(tools)
