from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
import os
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from mcp_server.logging_config import configure_logging

logger = logging.getLogger("physics-postdoc-mcp")
smart_mcp = FastMCP("physics-postdoc-mcp")

LOG_CONFIG = configure_logging(service_name="physics-postdoc-mcp", app_logger_name="physics-postdoc-mcp")
PHYSICS_MCP_URL = os.getenv("PHYSICS_MCP_URL", "http://127.0.0.1:8080")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
LLM_MODEL = os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

POSTDOC_CONTEXT = """
Du bist ein Physik- und Ingenieurwesen-Postdoc und bearbeitest MINT-Anfragen in jeder Sprache.
Arbeite strikt so:
1) Prüfe zuerst, ob alle nötigen Informationen für eine belastbare Lösung vorhanden sind.
2) Falls Informationen fehlen, frage kurz nach oder recherchiere zunächst selbst über verfügbare Tools.
3) Prüfe Einheiten systematisch und vereinheitliche intern auf SI, bevor du rechnest.
4) Gib Ergebnisse mit Einheit, Plausibilitätsprüfung und klaren Annahmen aus.

Tool-Anweisung für mcp-physics:
- Nutze `get_supported_cases_tool` bei Unsicherheit über Lastfälle.
- Nutze `get_model_assumptions_tool` für Modellgrenzen.
- Nutze `solve_beam_case_tool` mit einem payload-Objekt und SI-Feldern:
  case, length_m, youngs_modulus_pa,
  second_moment_m4, optional point_load_n, point_load_position_m, udl_n_per_m, samples.
- Bei nicht-SI Eingaben (kN, GPa, mm^4, cm, ...) immer vor Tool-Call nach SI konvertieren.
- Bei mehrsprachigen Eingaben Begriffe robust auf Tool-Felder mappen.
- Bei inkonsistenten Einheiten zuerst aktiv klären, dann rechnen.
""".strip()


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    enable_web_search: bool = True


class AskResponse(BaseModel):
    answer: str


def _resolve_api_key(provider: str) -> str | None:
    generic_key = os.getenv("LLM_API_KEY")
    if generic_key:
        return generic_key
    provider_key = os.getenv(f"{provider.upper()}_API_KEY")
    if provider_key:
        return provider_key
    if provider == "openai":
        return os.getenv("OPENAI_API_KEY")
    return None


def _build_chat_model() -> Any:
    api_key = _resolve_api_key(LLM_PROVIDER)
    if not api_key:
        raise RuntimeError("No API key configured for smart MCP chat model.")
    from langchain.chat_models import init_chat_model

    return init_chat_model(
        model=LLM_MODEL,
        model_provider=LLM_PROVIDER,
        api_key=api_key,
        temperature=0,
    )


def _build_tools(enable_web_search: bool) -> list[Any]:
    from langchain_core.tools import tool

    tools: list[Any] = []

    if enable_web_search:

        @tool
        def web_search(query: str) -> dict[str, Any]:
            try:
                with httpx.Client(timeout=15) as client:
                    response = client.get(
                        "https://api.duckduckgo.com/",
                        params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
                    )
                    response.raise_for_status()
                    body = response.json()
            except (httpx.HTTPError, ValueError) as exc:
                return {"ok": False, "error": {"tool": "web_search", "input": {"query": query}, "details": f"Web search failed: {exc}"}}
            snippets: list[str] = []
            if body.get("AbstractText"):
                snippets.append(str(body["AbstractText"]))
            for topic in body.get("RelatedTopics", [])[:5]:
                if isinstance(topic, dict) and topic.get("Text"):
                    snippets.append(str(topic["Text"]))
            return {"ok": True, "query": query, "snippets": snippets[:5]}

        tools.append(web_search)

    return tools


async def _build_physics_mcp_tools() -> tuple[Any, list[Any]]:
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError as exc:
        raise RuntimeError(
            "Missing MCP adapter dependency. Install `langchain-mcp-adapters`, "
            "for example with `pip install -e '.[dev]'`."
        ) from exc

    client = MultiServerMCPClient(
        {
            "physics": {
                "url": f"{PHYSICS_MCP_URL}/mcp",
                "transport": "streamable_http",
            }
        }
    )
    return client, await client.get_tools()


def _extract_text(response: Any) -> str:
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        output: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                output.append(str(item.get("text", "")))
            elif hasattr(item, "text"):
                output.append(str(item.text))
        return "\n".join(chunk for chunk in output if chunk).strip()
    return str(content)


async def run_postdoc_agent(question: str, enable_web_search: bool = True) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

    model = _build_chat_model()
    mcp_client, mcp_tools = await _build_physics_mcp_tools()
    local_tools = _build_tools(enable_web_search=enable_web_search)
    tools = [*mcp_tools, *local_tools]
    tool_lookup = {tool.name: tool for tool in tools}

    history: list[Any] = [HumanMessage(content=question)]
    request_messages = [SystemMessage(content=POSTDOC_CONTEXT), *history]
    current_model = model.bind_tools(tools)
    try:
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
                        "error": {"status_code": None, "tool": tool_name, "input": tool_args},
                    }
                else:
                    try:
                        tool_result = tool_impl.invoke(tool_args)
                    except Exception as exc:
                        tool_result = {
                            "ok": False,
                            "error": {
                                "status_code": None,
                                "tool": tool_name,
                                "input": tool_args,
                                "details": f"Tool execution failed: {exc}",
                            },
                        }
                history.append(ToolMessage(content=json.dumps(tool_result, ensure_ascii=False), tool_call_id=call["id"]))

            request_messages = [SystemMessage(content=POSTDOC_CONTEXT), *history]
            response = current_model.invoke(request_messages)
    finally:
        close_fn = getattr(mcp_client, "aclose", None)
        if callable(close_fn):
            await close_fn()

    return _extract_text(response)


@smart_mcp.tool()
async def ask_postdoc_tool(question: str) -> dict[str, str]:
    """Answer natural-language STEM questions using mcp-physics and optional web research."""
    return {"answer": await run_postdoc_agent(question, enable_web_search=True)}


mcp_http_app = smart_mcp.streamable_http_app()


@asynccontextmanager
async def lifespan(_: FastAPI):
    async with smart_mcp.session_manager.run():
        yield


app = FastAPI(title="physics-postdoc-mcp", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "physics-postdoc-mcp", "version": "0.1.0"}


@app.post("/v1/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    try:
        answer = await run_postdoc_agent(
            request.question,
            enable_web_search=request.enable_web_search,
        )
    except Exception as exc:
        logger.exception("postdoc agent failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return AskResponse(answer=answer)


app.mount("/", mcp_http_app)


def run() -> None:
    host = os.getenv("SMART_PHYSICS_MCP_HOST", "0.0.0.0")
    port = int(os.getenv("SMART_PHYSICS_MCP_PORT", "8090"))
    logger.info("Starting physics-postdoc-mcp", extra={"host": host, "port": port})
    uvicorn.run("smart_mcp_server.app:app", host=host, port=port, log_level="info", log_config=LOG_CONFIG)


if __name__ == "__main__":
    run()
