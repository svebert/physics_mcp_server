from __future__ import annotations

import asyncio
import importlib.metadata
import json
import logging
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
LOG_DIR = Path(os.getenv("PHYSICS_MCP_LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
CLIENT_LOG_FILE = LOG_DIR / "physics-mcp-client.log"

logger = logging.getLogger("physics-mcp-client")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(CLIENT_LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(file_handler)
    logger.propagate = False

DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
DEFAULT_MODEL = os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_PHYSICS_SERVER = os.getenv("PHYSICS_MCP_URL", "http://127.0.0.1:8080")
DEFAULT_POSTDOC_SERVER = os.getenv("SMART_MCP_URL", "http://127.0.0.1:8090")

TOOLSET_OPTIONS: dict[str, dict[str, Any]] = {
    "0": {"label": "none", "tools": [], "mcp_servers": [], "system_prompt": None},
    "1": {
        "label": "mcp-physics",
        "tools": ["solve_beam_case_tool", "get_supported_cases_tool", "get_model_assumptions_tool"],
        "mcp_servers": ["physics"],
        "system_prompt": None,
    },
    "2": {
        "label": "websearch",
        "tools": ["websearch"],
        "mcp_servers": [],
        "system_prompt": None,
    },
    "3": {
        "label": "mcp-physics + websearch",
        "tools": ["solve_beam_case_tool", "get_supported_cases_tool", "get_model_assumptions_tool", "websearch"],
        "mcp_servers": ["physics"],
        "system_prompt": None,
    },
    "4": {
        "label": "physik-postdoc",
        "tools": ["ask_postdoc_tool"],
        "mcp_servers": ["postdoc"],
        "system_prompt": None,
    },
    "5": {
        "label": "physik-postdoc + websearch",
        "tools": ["ask_postdoc_tool", "websearch"],
        "mcp_servers": ["postdoc"],
        "system_prompt": None,
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


def _ensure_server_is_reachable(server_url: str, server_name: str) -> None:
    try:
        with httpx.Client(timeout=5) as client:
            response = client.get(f"{server_url}/health")
    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"{server_name} server is not reachable. "
            "Start it first and check your URL env vars. "
            f"Current URL: {server_url}. Original error: {exc}"
        ) from exc
    if not response.is_success:
        raise RuntimeError(
            f"{server_name} server health check failed with "
            f"HTTP {response.status_code} at {server_url}/health. "
            "Start/restart the server and try again."
        )
    logger.info("Server reachable: %s (%s)", server_name, server_url)


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


def _invoke_tool(tool_impl: Any, tool_args: dict[str, Any]) -> Any:
    """Invoke a LangChain tool, supporting both async-only and sync implementations."""
    if hasattr(tool_impl, "ainvoke"):
        return asyncio.run(tool_impl.ainvoke(tool_args))
    return tool_impl.invoke(tool_args)


def _parse_semver(version: str) -> tuple[int, int, int]:
    normalized = version.split("+", 1)[0].split("-", 1)[0]
    parts = normalized.split(".")
    major = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 0
    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    patch = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
    return major, minor, patch


def _validate_langchain_mcp_versions() -> None:
    try:
        core_version = importlib.metadata.version("langchain-core")
    except importlib.metadata.PackageNotFoundError:
        core_version = "not-installed"
    try:
        adapter_version = importlib.metadata.version("langchain-mcp-adapters")
    except importlib.metadata.PackageNotFoundError:
        adapter_version = "not-installed"

    if adapter_version == "not-installed":
        raise RuntimeError(
            "Missing MCP adapter dependency (`langchain-mcp-adapters`). "
            "Install project dependencies with `pip install -e '.[dev]'`."
        )

    if _parse_semver(adapter_version) < (0, 2, 0):
        raise RuntimeError(
            "Installed `langchain-mcp-adapters` is too old for this client "
            f"(found {adapter_version}, required >=0.2,<0.3). "
            "Upgrade with: "
            "`pip install -U \"langchain-mcp-adapters>=0.2,<0.3\" "
            "\"langchain-core>=0.3.78,<0.4\"`."
        )

    if core_version == "not-installed" or _parse_semver(core_version) < (0, 3, 78):
        raise RuntimeError(
            "Installed `langchain-core` is incompatible with MCP adapters "
            f"(found {core_version}, required >=0.3.78,<0.4). "
            "Upgrade with: "
            "`pip install -U \"langchain-core>=0.3.78,<0.4\" "
            "\"langchain-mcp-adapters>=0.2,<0.3\"`."
        )


async def _build_mcp_tools(selected_servers: list[str]) -> tuple[Any, list[Any]]:
    _validate_langchain_mcp_versions()
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import MCP adapter client even though dependencies look installed. "
            "Reinstall matching versions with "
            "`pip install -U \"langchain-core>=0.3.78,<0.4\" "
            "\"langchain-mcp-adapters>=0.2,<0.3\"`."
        ) from exc

    server_map: dict[str, dict[str, str]] = {}
    if "physics" in selected_servers:
        server_map["physics"] = {
            "url": f"{DEFAULT_PHYSICS_SERVER}/mcp",
            "transport": "streamable_http",
        }
    if "postdoc" in selected_servers:
        server_map["postdoc"] = {
            "url": f"{DEFAULT_POSTDOC_SERVER}/mcp",
            "transport": "streamable_http",
        }
    client = MultiServerMCPClient(server_map)
    try:
        tools = await client.get_tools()
    except Exception as exc:
        close_fn = getattr(client, "aclose", None)
        if callable(close_fn):
            await close_fn()
        raise RuntimeError(
            "Failed to initialize MCP tools. Ensure the selected MCP server is running and "
            f"supports streamable HTTP. selected_servers={selected_servers!r}, error={exc}"
        ) from exc
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
    loaded_mcp_servers: set[str] = set()

    logger.info("Starting client | provider=%s model=%s", DEFAULT_PROVIDER, DEFAULT_MODEL)
    toolset_key = _select_toolset_interactive()
    toolset = TOOLSET_OPTIONS[toolset_key]
    required_servers = set(toolset["mcp_servers"])
    if required_servers:
        if "physics" in required_servers:
            _ensure_server_is_reachable(DEFAULT_PHYSICS_SERVER, "mcp-physics")
        if "postdoc" in required_servers:
            _ensure_server_is_reachable(DEFAULT_POSTDOC_SERVER, "physik-postdoc")
        try:
            mcp_client, mcp_tools = asyncio.run(_build_mcp_tools(sorted(required_servers)))
            loaded_mcp_servers = required_servers
        except RuntimeError as exc:
            print(f"\nFehler beim Laden des Tool-Sets '{toolset['label']}': {exc}")
            logger.exception("Failed to load initial toolset '%s'", toolset["label"])
            toolset = TOOLSET_OPTIONS["0"]
            mcp_tools = []
            loaded_mcp_servers = set()

    history: list[Any] = []

    print(
        f"\nAktiver Provider: {DEFAULT_PROVIDER}, Modell: {DEFAULT_MODEL}, "
        f"Tool-Set: {toolset['label']}. "
        "Befehle: /exit beendet, /tools wechselt Tool-Set."
    )
    while True:
        try:
            question = _read_question()
        except KeyboardInterrupt:
            print("\nBeendet (KeyboardInterrupt).")
            break
        if not question:
            continue
        if question.lower() in {"/exit", "exit", "quit"}:
            break
        if question.lower() == "/tools":
            toolset_key = _select_toolset_interactive()
            toolset = TOOLSET_OPTIONS[toolset_key]
            required_servers = set(toolset["mcp_servers"])
            if required_servers and required_servers != loaded_mcp_servers:
                if mcp_client is not None:
                    close_fn = getattr(mcp_client, "aclose", None)
                    if callable(close_fn):
                        asyncio.run(close_fn())
                if "physics" in required_servers:
                    _ensure_server_is_reachable(DEFAULT_PHYSICS_SERVER, "mcp-physics")
                if "postdoc" in required_servers:
                    _ensure_server_is_reachable(DEFAULT_POSTDOC_SERVER, "physik-postdoc")
                try:
                    mcp_client, mcp_tools = asyncio.run(_build_mcp_tools(sorted(required_servers)))
                    loaded_mcp_servers = required_servers
                except RuntimeError as exc:
                    print(f"\nFehler beim Laden des Tool-Sets '{toolset['label']}': {exc}")
                    logger.exception("Failed to switch toolset to '%s'", toolset["label"])
                    toolset = TOOLSET_OPTIONS["0"]
                    mcp_tools = []
                    loaded_mcp_servers = set()
            if not required_servers:
                loaded_mcp_servers = set()
            print(f"\nAktives Tool-Set: {toolset['label']}")
            continue

        history.append(HumanMessage(content=question))
        logger.info("User question received (len=%d) using toolset='%s'", len(question), toolset["label"])

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
                    try:
                        logger.info("Invoking tool '%s' with args=%s", tool_name, tool_args)
                        tool_result = _invoke_tool(tool_impl, tool_args)
                    except Exception as exc:  # pragma: no cover - defensive wrapper for interactive loop
                        logger.exception("Tool execution failed for '%s'", tool_name)
                        tool_result = {
                            "ok": False,
                            "error": {
                                "status_code": None,
                                "tool": tool_name,
                                "input": tool_args,
                                "details": f"Tool execution failed: {exc}",
                            },
                        }
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
    logger.info("Client shutdown")


if __name__ == "__main__":
    main()
