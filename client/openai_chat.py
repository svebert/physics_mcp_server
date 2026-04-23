from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from openai import OpenAI


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


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
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
    },
    {
        "type": "function",
        "function": {
            "name": "get_supported_cases",
            "description": "Get all case names supported by physics_core v0.1.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_model_assumptions",
            "description": "Get assumptions and limits of the beam model.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def invoke_server_tool(tool_name: str, args: dict[str, Any]) -> Any:
    with httpx.Client(timeout=30) as client:
        response = client.post(f"{DEFAULT_SERVER}/dev/tools/{tool_name}", json=args)
        response.raise_for_status()
        return response.json()


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Set it in your environment or in .env (see .env.example)."
        )


    client = OpenAI(api_key=api_key)
    question = input("Ask an engineering question: ").strip()

    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    while True:
        completion = client.chat.completions.create(model=DEFAULT_MODEL, messages=messages, tools=TOOLS)
        msg = completion.choices[0].message
        tool_calls = msg.tool_calls or []

        if not tool_calls:
            print("\nAnswer:\n")
            print(msg.content)
            break

        messages.append(msg.model_dump(exclude_none=True))
        for call in tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments or "{}")
            result = invoke_server_tool(name, args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": name,
                    "content": json.dumps(result),
                }
            )


if __name__ == "__main__":
    main()
