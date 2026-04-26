from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
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
DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
DEFAULT_MODEL = os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_SERVER = os.getenv("PHYSICS_MCP_URL", "http://127.0.0.1:8080")

PHYSICS_POSTDOC_CONTEXT = """
Du bist ein Physik- und Ingenieurwesen-Postdoc und bearbeitest MINT-Anfragen in jeder Sprache.
Arbeite strikt so:
1) Prüfe zuerst, ob alle nötigen Informationen für eine belastbare Lösung vorhanden sind.
2) Falls Informationen fehlen, frage kurz nach oder recherchiere zunächst selbst über verfügbare Tools.
3) Prüfe Einheiten systematisch und vereinheitliche intern auf SI, bevor du rechnest.
4) Gib Ergebnisse mit Einheit, Plausibilitätsprüfung und klaren Annahmen aus.
""".strip()

PHYSICS_MCP_TOOL_INSTRUCTION = """
Nutzung der mcp-physics Tools:
- Verwende `get_supported_cases` bevor du unsicher über Lastfall-Namen bist.
- Verwende `get_model_assumptions`, wenn Grenzen/Annahmen relevant sind.
- Verwende `solve_beam_case` nur mit diesen SI-Feldern:
  case, length_m, youngs_modulus_pa, second_moment_m4, optional point_load_n,
  point_load_position_m, udl_n_per_m, samples.
- Wenn Nutzer in anderer Sprache schreibt: mappe Begriffe robust auf die Tool-Felder.
- Wenn Nutzer in nicht-SI schreibt (z.B. kN, GPa, mm^4), konvertiere vor Tool-Call nach SI
  und erwähne Konvertierung in der Antwort.
- Nutze nur physikalisch konsistente Einheiten; bei Widersprüchen zuerst klären.
""".strip()


TOOLSET_OPTIONS: dict[str, dict[str, Any]] = {
    "0": {"label": "no tools", "tools": [], "needs_mcp": False, "system_prompt": None},
    "1": {
        "label": "mcp-physics",
        "tools": ["solve_beam_case", "get_supported_cases", "get_model_assumptions"],
        "needs_mcp": True,
        "system_prompt": None,
    },
    "2": {
        "label": "websearch",
        "tools": ["websearch"],
        "needs_mcp": False,
        "system_prompt": None,
    },
    "3": {
        "label": "mcp-physics+websearch",
        "tools": ["solve_beam_case", "get_supported_cases", "get_model_assumptions", "websearch"],
        "needs_mcp": True,
        "system_prompt": None,
    },
    "4": {
        "label": "physics-postdoc",
        "tools": ["solve_beam_case", "get_supported_cases", "get_model_assumptions"],
        "needs_mcp": True,
        "system_prompt": f"{PHYSICS_POSTDOC_CONTEXT}\n\n{PHYSICS_MCP_TOOL_INSTRUCTION}",
    },
    "5": {
        "label": "physics-postdoc+websearch",
        "tools": ["solve_beam_case", "get_supported_cases", "get_model_assumptions", "websearch"],
        "needs_mcp": True,
        "system_prompt": (
            f"{PHYSICS_POSTDOC_CONTEXT}\n\n{PHYSICS_MCP_TOOL_INSTRUCTION}\n\n"
            "Nutze zusätzlich Web-Recherche, wenn Wissen nicht sicher oder aktuell ist."
        ),
    },
}


def _resolve_api_key(provider: str) -> str | None:
    """Resolve a provider API key with backward-compatible env var names."""
    generic_key = os.getenv("LLM_API_KEY")
    if generic_key:
        return generic_key

    provider_key_var = f"{provider.upper()}_API_KEY"
    provider_key = os.getenv(provider_key_var)
    if provider_key:
        return provider_key

    if provider == "openai":
        return os.getenv("OPENAI_API_KEY")

    return None


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


def _build_chat_model() -> Any:
    provider = DEFAULT_PROVIDER
    api_key = _resolve_api_key(provider)

    if not api_key:
        raise RuntimeError(
            "No API key found. Set LLM_API_KEY (provider-agnostic) or "
            f"{provider.upper()}_API_KEY. For OpenAI, OPENAI_API_KEY remains supported."
        )

    try:
        from langchain.chat_models import init_chat_model
    except ImportError as exc:
        raise RuntimeError(
            "LangChain is required for the client. Install project dependencies, e.g. "
            "`pip install -e '.[dev]'`."
        ) from exc

    try:
        return init_chat_model(
            model=DEFAULT_MODEL,
            model_provider=provider,
            api_key=api_key,
            temperature=0,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not initialize chat model. "
            f"provider={provider!r}, model={DEFAULT_MODEL!r}. "
            "Ensure matching provider package is installed (e.g. langchain-openai or "
            "langchain-anthropic) and env vars are set correctly."
        ) from exc


def _web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    max_results = max(1, min(max_results, 10))
    try:
        with httpx.Client(timeout=20) as client:
            response = client.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            )
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        return {"ok": False, "error": {"details": f"Websearch failed: {exc}"}}

    results: list[dict[str, str]] = []

    abstract = str(payload.get("AbstractText") or "").strip()
    abstract_url = str(payload.get("AbstractURL") or "").strip()
    heading = str(payload.get("Heading") or "").strip()
    if abstract:
        results.append(
            {
                "title": heading or "DuckDuckGo Abstract",
                "url": abstract_url,
                "snippet": abstract,
            }
        )

    for item in payload.get("RelatedTopics", []) or []:
        if isinstance(item, dict) and "Text" in item:
            results.append(
                {
                    "title": str(item.get("FirstURL") or "Related result"),
                    "url": str(item.get("FirstURL") or ""),
                    "snippet": str(item.get("Text") or ""),
                }
            )
        if isinstance(item, dict) and isinstance(item.get("Topics"), list):
            for nested in item["Topics"]:
                if isinstance(nested, dict) and "Text" in nested:
                    results.append(
                        {
                            "title": str(nested.get("FirstURL") or "Related result"),
                            "url": str(nested.get("FirstURL") or ""),
                            "snippet": str(nested.get("Text") or ""),
                        }
                    )

    return {"ok": True, "query": query, "results": results[:max_results]}


def _build_web_search_tool() -> Any:
    from langchain_core.tools import tool

    @tool
    def websearch(query: str, max_results: int = 5) -> dict[str, Any]:
        """Search the web for current context and return short snippets with URLs."""
        return _web_search(query=query, max_results=max_results)

    return websearch


async def _build_mcp_tools() -> tuple[Any, list[Any]]:
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError as exc:
        detail = str(exc)
        if "langchain_core.messages.content" in detail:
            raise RuntimeError(
                "Incompatible LangChain packages detected: `langchain-mcp-adapters` "
                "expects a newer `langchain-core` than the one installed. \
Install matching versions (for this project: `langchain-mcp-adapters>=0.2,<0.3`) with \
`pip install -e '.[dev]'` or pin `langchain-mcp-adapters>=0.2,<0.3`."
            ) from exc

        raise RuntimeError(
            "Missing MCP adapter dependency. Install `langchain-mcp-adapters`, "
            "for example with `pip install -e '.[dev]'`."
        ) from exc

    client = MultiServerMCPClient(
        {
            "physics": {
                "url": f"{DEFAULT_SERVER}/mcp",
                "transport": "streamable_http",
            }
        }
    )
    tools = await client.get_tools()
    return client, tools


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

    choices = f"{min(TOOLSET_OPTIONS.keys())}-{max(TOOLSET_OPTIONS.keys())}"
    selection = prompt(f"Auswahl [{choices}, Default 1]: ").strip() or "1"
    if selection not in TOOLSET_OPTIONS:
        print(f"Ungültige Auswahl '{selection}', nehme Default 1 (mcp-physics).")
        return "1"
    return selection


def _extract_text(response: Any) -> str:
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                chunks.append(str(item.get("text", "")))
            elif hasattr(item, "text"):
                chunks.append(str(item.text))
        return "\n".join(chunk for chunk in chunks if chunk).strip()
    return str(content)


def main() -> None:
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

    model = _build_chat_model()
    web_tool = _build_web_search_tool()
    mcp_client: Any | None = None
    mcp_tools: list[Any] = []

    toolset_key = _select_toolset_interactive()
    toolset = TOOLSET_OPTIONS[toolset_key]
    if toolset["needs_mcp"]:
        _ensure_server_is_reachable()
        mcp_client, mcp_tools = asyncio.run(_build_mcp_tools())

    history: list[Any] = []

    print(
        f"\nAktiver Provider: {DEFAULT_PROVIDER}, Modell: {DEFAULT_MODEL}, "
        f"Tool-Set: {toolset['label']}. "
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
            if toolset["needs_mcp"] and not mcp_tools:
                _ensure_server_is_reachable()
                mcp_client, mcp_tools = asyncio.run(_build_mcp_tools())
            print(f"\nAktives Tool-Set: {toolset['label']}")
            continue

        history.append(HumanMessage(content=question))

        tool_pool = [*mcp_tools, web_tool]
        tool_lookup = {tool.name: tool for tool in tool_pool}
        selected_tools = [tool_lookup[name] for name in toolset["tools"] if name in tool_lookup]
        current_model = model.bind_tools(selected_tools) if selected_tools else model
        request_messages = list(history)
        if toolset.get("system_prompt"):
            request_messages = [SystemMessage(content=toolset["system_prompt"]), *request_messages]

        response = current_model.invoke(request_messages)

        while getattr(response, "tool_calls", None):
            history.append(response)
            for call in response.tool_calls:
                tool_name = call["name"]
                tool_args = call.get("args", {})
                tool_impl = tool_lookup.get(tool_name)
                if tool_impl is None:
                    tool_result: dict[str, Any] = {
                        "ok": False,
                        "error": {
                            "status_code": None,
                            "tool": tool_name,
                            "input": tool_args,
                            "details": f"Unknown tool '{tool_name}'",
                        },
                    }
                else:
                    tool_result = tool_impl.invoke(tool_args)
                history.append(
                    ToolMessage(
                        content=json.dumps(tool_result, ensure_ascii=False),
                        tool_call_id=call["id"],
                    )
                )
            request_messages = list(history)
            if toolset.get("system_prompt"):
                request_messages = [SystemMessage(content=toolset["system_prompt"]), *request_messages]
            response = current_model.invoke(request_messages)

        history.append(response)
        print("\nAntwort:\n")
        print(_extract_text(response))

    if mcp_client is not None:
        close_fn = getattr(mcp_client, "aclose", None)
        if callable(close_fn):
            asyncio.run(close_fn())


if __name__ == "__main__":
    main()
