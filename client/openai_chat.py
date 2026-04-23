from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from openai import OpenAI
from prompt_toolkit import prompt
from prompt_toolkit.key_binding import KeyBindings

def _load_env() -> None:
    """Load env vars from local .env files for easier CLI usage."""
    cwd_env = Path.cwd() / ".env"
    repo_env = Path(__file__).resolve().parent.parent / ".env"
    if cwd_env.exists():
        load_dotenv(cwd_env, override=False)
    elif repo_env.exists():
        load_dotenv(repo_env, override=False)


_load_env()
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_SERVER = os.getenv("PHYSICS_MCP_URL", "http://127.0.0.1:8080")

MCP_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "solve_beam_case",
        "description": "Solve one supported beam case and return reactions, maxima, and curves.",
        "parameters": {
            "type": "object",
            "properties": {
                "case": {"type": "string"},
                "length_m": {"type": "number"},
                "youngs_modulus_pa": {"type": "number"},
                "second_moment_m4": {"type": "number"},
                "point_load_n": {"type": "number"},
                "point_load_position_m": {"type": "number"},
                "udl_n_per_m": {"type": "number"},
                "samples": {"type": "integer"},
            },
            "required": ["case", "length_m", "youngs_modulus_pa", "second_moment_m4"],
        },
    },
    {
        "type": "function",
        "name": "get_supported_cases",
        "description": "Get all case names supported by physics_core v0.1.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "get_model_assumptions",
        "description": "Get assumptions and limits of the beam model.",
        "parameters": {"type": "object", "properties": {}},
    },
]

WEB_SEARCH_TOOLS: list[dict[str, Any]] = [{"type": "web_search_preview"}]

TOOLSET_OPTIONS: dict[str, dict[str, Any]] = {
    "0": {"label": "Kein Tool", "tools": [], "needs_mcp": False},
    "1": {"label": "Nur physics-mcp", "tools": MCP_TOOLS, "needs_mcp": True},
    "2": {"label": "Nur Websearch", "tools": WEB_SEARCH_TOOLS, "needs_mcp": False},
    "3": {
        "label": "Websearch + physics-mcp",
        "tools": [*WEB_SEARCH_TOOLS, *MCP_TOOLS],
        "needs_mcp": True,
    },
}

MCP_TOOL_NAMES = {tool["name"] for tool in MCP_TOOLS if tool["type"] == "function"}


def invoke_server_tool(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    with httpx.Client(timeout=30) as client:
        try:
            response = client.post(f"{DEFAULT_SERVER}/dev/tools/{tool_name}", json=args)
        except httpx.HTTPError as exc:
            return {
                "ok": False,
                "error": {
                    "status_code": None,
                    "tool": tool_name,
                    "input": args,
                    "details": f"Cannot reach physics MCP server at {DEFAULT_SERVER}: {exc}",
                },
            }
        if response.is_success:
            return {"ok": True, "result": response.json()}
        try:
            details: Any = response.json()
        except ValueError:
            details = response.text
        return {
            "ok": False,
            "error": {
                "status_code": response.status_code,
                "tool": tool_name,
                "input": args,
                "details": details,
            },
        }


def _ensure_server_is_reachable() -> None:
    try:
        with httpx.Client(timeout=5) as client:
            response = client.get(f"{DEFAULT_SERVER}/health")
    except httpx.HTTPError as exc:
        raise RuntimeError(
            "Physics MCP server is not reachable. "
            f"Start it first (e.g. `physics-mcp-server`) or set PHYSICS_MCP_URL correctly. "
            f"Current URL: {DEFAULT_SERVER}. Original error: {exc}"
        ) from exc
    if not response.is_success:
        raise RuntimeError(
            "Physics MCP server health check failed with "
            f"HTTP {response.status_code} at {DEFAULT_SERVER}/health. "
            "Start/restart the server and try again."
        )


def _read_question() -> str:
    key_bindings = KeyBindings()

    @key_bindings.add("enter")
    def _(event: Any) -> None:
        event.current_buffer.validate_and_handle()

    @key_bindings.add("escape", "enter")
    def _(event: Any) -> None:
        event.current_buffer.insert_text("\n")

    @key_bindings.add("c-t")
    def _(event: Any) -> None:
        event.current_buffer.text = "/tools"
        event.current_buffer.validate_and_handle()

    return prompt(
        "Frage eingeben (Enter senden, Alt+Enter Zeilenumbruch, Ctrl+T Tool-Set):\n",

        multiline=True,
        key_bindings=key_bindings,
    ).strip()


def _select_toolset_interactive() -> str:
    print("\nTool-Set auswählen:")
    for key, option in TOOLSET_OPTIONS.items():
        print(f"  {key}: {option['label']}")

    selection = prompt("Auswahl [0-3, Default 1]: ").strip() or "1"
    if selection not in TOOLSET_OPTIONS:
        print(f"Ungültige Auswahl '{selection}', nehme Default 1 (Nur physics-mcp).")
        return "1"
    return selection


def _invoke_tools_for_response(client: OpenAI, response: Any, tools: list[dict[str, Any]]) -> Any:
    current_response = response
    while True:
        function_calls = [item for item in current_response.output if item.type == "function_call"]
        if not function_calls:
            return current_response

        tool_outputs: list[dict[str, Any]] = []
        for call in function_calls:
            name = call.name
            args = json.loads(call.arguments or "{}")
            if name in MCP_TOOL_NAMES:
                outcome = invoke_server_tool(name, args)
            else:
                outcome = {
                    "ok": False,
                    "error": {
                        "status_code": None,
                        "tool": name,
                        "input": args,
                        "details": f"Unknown tool '{name}'",
                    },
                }

            tool_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(outcome),
                }
            )

        current_response = client.responses.create(
            model=DEFAULT_MODEL,
            input=tool_outputs,
            tools=tools,
            previous_response_id=current_response.id,
        )


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Set it in your environment or in .env (see .env.example)."
        )

    client = OpenAI(api_key=api_key)
    toolset_key = _select_toolset_interactive()
    toolset = TOOLSET_OPTIONS[toolset_key]
    if toolset["needs_mcp"]:
        _ensure_server_is_reachable()
    previous_response_id: str | None = None

    print(
        f"\nAktives Tool-Set: {toolset['label']}. "
        "Befehle: /exit beendet, /tools wechselt Tool-Set."
    )
    while True:
        question = _read_question()
        if not question:
            continue
        if question.lower() in {"/exit", "exit", "quit"}:
            break
        if question.lower() == "/tools":
            toolset_key = _select_toolset_interactive()
            toolset = TOOLSET_OPTIONS[toolset_key]
            if toolset["needs_mcp"]:
                _ensure_server_is_reachable()
            print(f"\nAktives Tool-Set: {toolset['label']}")
            continue

        response = client.responses.create(
            model=DEFAULT_MODEL,
            input=question,
            tools=toolset["tools"],
            previous_response_id=previous_response_id,
        )
        response = _invoke_tools_for_response(client, response, toolset["tools"])
        previous_response_id = response.id
        print("\nAntwort:\n")
        print(response.output_text)


if __name__ == "__main__":
    main()
