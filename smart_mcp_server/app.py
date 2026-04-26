from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from mcp_server.env_loader import load_project_env
from mcp_server.logging_config import configure_logging
from mcp_server.middleware import RateLimitHook

load_project_env(Path(__file__).resolve().parent)

logger = logging.getLogger("physics-postdoc-mcp")
smart_mcp = FastMCP("physics-postdoc-mcp")

LOG_LEVEL_NAME = os.getenv("SMART_PHYSICS_MCP_LOG_LEVEL", os.getenv("PHYSICS_MCP_LOG_LEVEL", "info")).strip().lower()
LOG_LEVEL = getattr(logging, LOG_LEVEL_NAME.upper(), logging.INFO)
LOG_CONFIG = configure_logging(service_name="physics-postdoc-mcp", app_logger_name="physics-postdoc-mcp", level=LOG_LEVEL)
PHYSICS_MCP_URL = os.getenv("PHYSICS_MCP_URL", "http://127.0.0.1:8080")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
LLM_MODEL = os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_TOOL_ROUNDS = int(os.getenv("SMART_MCP_MAX_TOOL_ROUNDS", "12"))
MAX_LOG_PAYLOAD_CHARS = int(os.getenv("SMART_MCP_MAX_LOG_PAYLOAD_CHARS", "2000"))

POSTDOC_CONTEXT = """
Du bist ein Physik- und Ingenieurwesen-Postdoc und bearbeitest MINT-Anfragen in jeder Sprache.
Arbeite strikt so:
1) Prüfe zuerst, ob alle nötigen Informationen für eine belastbare Lösung vorhanden sind.
2) Falls Informationen fehlen, frage kurz nach oder recherchiere zunächst selbst über verfügbare Tools.
3) Prüfe Einheiten systematisch und vereinheitliche intern auf SI, bevor du rechnest.
4) Gib Ergebnisse mit Einheit, Plausibilitätsprüfung und klaren Annahmen aus.

Tool-Anweisung für mcp-physics:
- Nutze `get_supported_cases_tool` nur bei echter Unsicherheit über Lastfälle (maximal 1x pro Frage).
- Nutze `get_model_assumptions_tool` für Modellgrenzen.
- Nutze `solve_beam_case_tool` bevorzugt genau 1x pro Frage und dann direkt die Ergebnisformulierung.
- Verwende für `solve_beam_case_tool` bevorzugt dieses Format:
  {"payload": {"case": "...", "length_m": ..., "youngs_modulus_pa": ..., "second_moment_m4": ..., "point_load_n": ..., "point_load_position_m": ...}}
  (Flat-Argumente ohne `payload` sind nur Fallback.)
- Unterstützte Fälle: simply_supported_point, simply_supported_udl, cantilever_point, cantilever_udl.
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
            """Search the public web and return short evidence snippets for grounding answers."""
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


def _summarize_messages(messages: list[Any], max_chars: int = MAX_LOG_PAYLOAD_CHARS) -> str:
    summary: list[dict[str, Any]] = []
    for message in messages:
        role = message.__class__.__name__
        content = getattr(message, "content", "")
        if isinstance(content, list):
            content = json.dumps(content, ensure_ascii=False)
        summary.append({"role": role, "content": str(content)[:300]})
    payload = json.dumps(summary, ensure_ascii=False)
    if len(payload) <= max_chars:
        return payload
    return f"{payload[:max_chars]}...<truncated {len(payload)-max_chars} chars>"


async def run_postdoc_agent(question: str, enable_web_search: bool = True) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

    async def _invoke_tool(tool_impl: Any, tool_args: dict[str, Any]) -> Any:
        if hasattr(tool_impl, "ainvoke"):
            return await tool_impl.ainvoke(tool_args)
        if hasattr(tool_impl, "invoke"):
            return tool_impl.invoke(tool_args)
        raise TypeError(f"Tool {getattr(tool_impl, 'name', '<unknown>')} is not invokable.")

    model = _build_chat_model()
    mcp_client, mcp_tools = await _build_physics_mcp_tools()
    local_tools = _build_tools(enable_web_search=enable_web_search)
    tools = [*mcp_tools, *local_tools]
    tool_lookup = {tool.name: tool for tool in tools}

    history: list[Any] = [HumanMessage(content=question)]
    request_messages = [SystemMessage(content=POSTDOC_CONTEXT), *history]
    current_model = model.bind_tools(tools)
    guardrail_message = (
        "Ich breche hier ab, weil das Modell wiederholt dieselben Tool-Aufrufe erzeugt hat. "
        "Bitte formuliere die Frage präziser oder nutze direkt das mcp-physics Tool-Set."
    )
    trace_id = uuid.uuid4().hex[:12]
    started = time.perf_counter()
    logger.debug(
        "postdoc-trace start trace_id=%s enable_web_search=%s question=%s",
        trace_id,
        enable_web_search,
        question[:MAX_LOG_PAYLOAD_CHARS],
    )
    logger.debug("postdoc-trace selected-tools trace_id=%s tools=%s", trace_id, [tool.name for tool in tools])
    try:
        logger.debug("postdoc-trace model-request trace_id=%s messages=%s", trace_id, _summarize_messages(request_messages))
        response = await current_model.ainvoke(request_messages)
        logger.debug("postdoc-trace model-response trace_id=%s content=%s tool_calls=%s", trace_id, _extract_text(response)[:MAX_LOG_PAYLOAD_CHARS], json.dumps(getattr(response, "tool_calls", []), ensure_ascii=False)[:MAX_LOG_PAYLOAD_CHARS])
        tool_round = 0
        previous_round_signatures: tuple[str, ...] = ()
        repeated_rounds = 0

        while getattr(response, "tool_calls", None):
            tool_round += 1
            logger.debug(
                "postdoc-trace model-tool-round trace_id=%s round=%d tool_calls=%s",
                trace_id,
                tool_round,
                json.dumps(response.tool_calls, ensure_ascii=False)[:MAX_LOG_PAYLOAD_CHARS],
            )
            if tool_round > MAX_TOOL_ROUNDS:
                logger.warning("Stopping postdoc agent after reaching max tool rounds", extra={"max_rounds": MAX_TOOL_ROUNDS})
                return guardrail_message

            current_round_signatures = tuple(
                json.dumps({"name": call["name"], "args": call.get("args", {})}, ensure_ascii=False, sort_keys=True)
                for call in response.tool_calls
            )
            if current_round_signatures == previous_round_signatures:
                repeated_rounds += 1
            else:
                repeated_rounds = 0
            previous_round_signatures = current_round_signatures

            if repeated_rounds >= 2:
                logger.warning(
                    "Stopping postdoc agent due to repeated tool-call rounds",
                    extra={"tool_signatures": current_round_signatures},
                )
                return guardrail_message

            history.append(response)
            for call in response.tool_calls:
                tool_name = call["name"]
                tool_args = call.get("args", {})
                logger.debug(
                    "postdoc-trace tool-request trace_id=%s round=%d tool=%s args=%s",
                    trace_id,
                    tool_round,
                    tool_name,
                    json.dumps(tool_args, ensure_ascii=False)[:MAX_LOG_PAYLOAD_CHARS],
                )
                tool_impl = tool_lookup.get(tool_name)
                if tool_impl is None:
                    tool_result: dict[str, Any] = {
                        "ok": False,
                        "error": {"status_code": None, "tool": tool_name, "input": tool_args},
                    }
                else:
                    try:
                        tool_result = await _invoke_tool(tool_impl, tool_args)
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
                logger.debug(
                    "postdoc-trace tool-response trace_id=%s round=%d tool=%s result=%s",
                    trace_id,
                    tool_round,
                    tool_name,
                    json.dumps(tool_result, ensure_ascii=False)[:MAX_LOG_PAYLOAD_CHARS],
                )
                history.append(ToolMessage(content=json.dumps(tool_result, ensure_ascii=False), tool_call_id=call["id"]))

            request_messages = [SystemMessage(content=POSTDOC_CONTEXT), *history]
            logger.debug("postdoc-trace model-request trace_id=%s messages=%s", trace_id, _summarize_messages(request_messages))
            response = await current_model.ainvoke(request_messages)
            logger.debug("postdoc-trace model-response trace_id=%s content=%s tool_calls=%s", trace_id, _extract_text(response)[:MAX_LOG_PAYLOAD_CHARS], json.dumps(getattr(response, "tool_calls", []), ensure_ascii=False)[:MAX_LOG_PAYLOAD_CHARS])
    finally:
        close_fn = getattr(mcp_client, "aclose", None)
        if callable(close_fn):
            await close_fn()

    answer = _extract_text(response)
    logger.debug(
        "postdoc-trace finish trace_id=%s elapsed_ms=%.2f answer=%s",
        trace_id,
        (time.perf_counter() - started) * 1_000,
        answer[:MAX_LOG_PAYLOAD_CHARS],
    )
    return answer


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
rate_limit_hook = RateLimitHook(logger_name="physics-postdoc-mcp")
app.middleware("http")(rate_limit_hook)


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
    uvicorn.run("smart_mcp_server.app:app", host=host, port=port, log_level=LOG_LEVEL_NAME, log_config=LOG_CONFIG)


if __name__ == "__main__":
    run()
