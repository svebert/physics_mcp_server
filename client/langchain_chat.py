from __future__ import annotations

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
    "0": {"label": "Kein Tool", "tools": [], "needs_mcp": False, "system_prompt": None},
    "1": {
        "label": "Nur physics-mcp",
        "tools": ["solve_beam_case", "get_supported_cases", "get_model_assumptions"],
        "needs_mcp": True,
        "system_prompt": None,
    },
    "2": {
        "label": "physics post doc",
        "tools": ["solve_beam_case", "get_supported_cases", "get_model_assumptions"],
        "needs_mcp": True,
        "system_prompt": f"{PHYSICS_POSTDOC_CONTEXT}\n\n{PHYSICS_MCP_TOOL_INSTRUCTION}",
    },
    "3": {
        "label": "websearch+physik post doc",
        "tools": ["solve_beam_case", "get_supported_cases", "get_model_assumptions", "web_search"],
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

def _build_mcp_tools() -> list[Any]:
    from langchain_core.tools import tool

    @tool
    def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
        """Search the web for current context and return short snippets with URLs."""
        return _web_search(query=query, max_results=max_results)

    @tool
    def solve_beam_case(
        case: str,
        length_m: float,
        youngs_modulus_pa: float,
        second_moment_m4: float,
        point_load_n: float | None = None,
        point_load_position_m: float | None = None,
        udl_n_per_m: float | None = None,
        samples: int | None = None,
    ) -> dict[str, Any]:
        """Solve one supported beam case and return reactions, maxima, and curves."""
        payload = {
            "case": case,
            "length_m": length_m,
            "youngs_modulus_pa": youngs_modulus_pa,
            "second_moment_m4": second_moment_m4,
        }
        optional_fields = {
            "point_load_n": point_load_n,
            "point_load_position_m": point_load_position_m,
            "udl_n_per_m": udl_n_per_m,
            "samples": samples,
        }
        payload.update({key: value for key, value in optional_fields.items() if value is not None})
        return invoke_server_tool("solve_beam_case", payload)

    @tool
    def get_supported_cases() -> dict[str, Any]:
        """Get all case names supported by physics_core v0.1."""
        return invoke_server_tool("get_supported_cases", {})

    @tool
    def get_model_assumptions() -> dict[str, Any]:
        """Get assumptions and limits of the beam model."""
        return invoke_server_tool("get_model_assumptions", {})

    return [web_search, solve_beam_case, get_supported_cases, get_model_assumptions]


def _build_web_search_tool() -> Any:
    from langchain_core.tools import tool

    @tool
    def web_search(query: str) -> dict[str, Any]:
        """Run a lightweight public web search and return short evidence snippets."""
        with httpx.Client(timeout=15) as client:
            response = client.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            )
            response.raise_for_status()
            body = response.json()

        snippets: list[str] = []
        if body.get("AbstractText"):
            snippets.append(str(body["AbstractText"]))
        for item in body.get("RelatedTopics", [])[:5]:
            if isinstance(item, dict) and item.get("Text"):
                snippets.append(str(item["Text"]))
        return {"ok": True, "query": query, "snippets": snippets[:5]}

    return web_search


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

    choices = "-".join([min(TOOLSET_OPTIONS.keys()), max(TOOLSET_OPTIONS.keys())])
    selection = prompt(f"Auswahl [{choices}, Default 1]: ").strip() or "1"
    if selection not in TOOLSET_OPTIONS:
        print(f"Ungültige Auswahl '{selection}', nehme Default 1 (Nur physics-mcp).")
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
    mcp_tools = _build_mcp_tools()
    web_tool = _build_web_search_tool()
    all_tools = mcp_tools + [web_tool]
    tool_lookup = {tool.name: tool for tool in all_tools}

    toolset_key = _select_toolset_interactive()
    toolset = TOOLSET_OPTIONS[toolset_key]
    if toolset["needs_mcp"]:
        _ensure_server_is_reachable()

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
            if toolset["needs_mcp"]:
                _ensure_server_is_reachable()
            print(f"\nAktives Tool-Set: {toolset['label']}")
            continue

        history.append(HumanMessage(content=question))

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


if __name__ == "__main__":
    main()
