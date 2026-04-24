# physics-mcp (v0.1)

Minimal, production-clean pilot project exposing simple Euler-Bernoulli beam calculations as a remote MCP service over **Streamable HTTP**.

## What it does

- Solves 4 beam cases:
  1. simply supported + one point load
  2. simply supported + full-span UDL
  3. cantilever + tip point load
  4. cantilever + full-span UDL
- Returns:
  - support reactions
  - maximum bending moment
  - maximum deflection
  - sampled arrays: `x`, `shear`, `moment`, `deflection`
  - warnings and model assumptions

## Architecture overview

- `physics_core/`: deterministic, pure physics logic (no MCP dependency).
- `mcp_server/`: MCP tool wrapper + FastAPI app.
  - MCP endpoint mounted at `/mcp` (Streamable HTTP transport).
  - Health endpoint at `/health`.
  - Dev helper endpoint at `/dev/tools/{tool_name}` for quick local testing.
- `client/`: LangChain-based CLI tool-calling chat client with provider-agnostic model support.
- `tests/`: unit, validation, integration-like, and smoke tests.

## Repository tree

```text
physics-mcp/
├── client/
│   └── langchain_chat.py
├── mcp_server/
│   ├── app.py
│   ├── logging_config.py
│   └── middleware.py
├── physics_core/
│   ├── assumptions.py
│   ├── models.py
│   └── solver.py
├── scripts/
│   ├── local_smoke.sh
│   └── sample_request.json
├── tests/
│   ├── test_mcp_tools.py
│   ├── test_physics_core.py
│   ├── test_smoke_local.py
│   └── test_validation.py
├── .env.example
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
└── README.md
```

## Local setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
# default client setup (OpenAI provider)
pip install -e '.[dev,client-openai]'
cp .env.example .env
# put LLM_API_KEY (recommended) or OPENAI_API_KEY (backward-compatible) into .env
```

For Claude/Anthropic support install:

```bash
pip install -e '.[dev,client-anthropic]'
```

## Environment variables

- `LLM_PROVIDER`: optional provider selector for the LangChain client (default: `openai`).
- `LLM_MODEL`: optional model name for the client (default fallback: `OPENAI_MODEL`, then `gpt-4o-mini`).
- `LLM_API_KEY`: recommended, provider-agnostic API key variable for the client.
- `OPENAI_API_KEY`: backward-compatible fallback for OpenAI usage.
- `OPENAI_MODEL`: backward-compatible fallback model variable.
- `PHYSICS_MCP_URL`: optional for client (default: `http://127.0.0.1:8080`).
- `PHYSICS_MCP_HOST`: server bind host (default: `0.0.0.0`).
- `PHYSICS_MCP_PORT`: server port (default: `8080`).

## API key setup (provider-agnostic)

Recommended naming scheme:

```bash
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4o-mini
export LLM_API_KEY='...'
```

Backward compatibility for existing OpenAI setups remains:

```bash
export OPENAI_API_KEY='sk-...'
export OPENAI_MODEL=gpt-4o-mini
```

For Claude/Anthropic, for example:

```bash
export LLM_PROVIDER=anthropic
export LLM_MODEL=claude-3-5-sonnet-latest
# choose one of the following:
export LLM_API_KEY='...'
# or
export ANTHROPIC_API_KEY='...'
```

## Run tests

```bash
pytest
```


If `physics-mcp-client` is not found, reinstall in your active venv:

```bash
python3 -m pip install -e '.[dev,client-openai]'
```

## Run the MCP server locally

```bash
physics-mcp-server
# or
python3 -m mcp_server.app
```

### Quick local smoke test

With server running:

```bash
./scripts/local_smoke.sh
```

## Example JSON request

`POST /dev/tools/solve_beam_case`

```json
{
  "case": "simply_supported_point",
  "length_m": 6.0,
  "point_load_n": 12000.0,
  "point_load_position_m": 3.0,
  "youngs_modulus_pa": 210000000000.0,
  "second_moment_m4": 0.0000085,
  "samples": 51
}
```

Equivalent `curl` command:

```bash
curl -sS -X POST "http://127.0.0.1:8080/dev/tools/solve_beam_case" \
  -H 'content-type: application/json' \
  -d '{
    "case": "simply_supported_point",
    "length_m": 6.0,
    "point_load_n": 12000.0,
    "point_load_position_m": 3.0,
    "youngs_modulus_pa": 210000000000.0,
    "second_moment_m4": 0.0000085,
    "samples": 51
  }'
```

## LangChain test client

Run with MCP server available locally:

```bash
# uses LLM_* vars (recommended) or OPENAI_* fallback vars
physics-mcp-client
# fallback without console script:
python3 -m client.langchain_chat
```

Input UX in the client:
- Arrow keys navigate input history and cursor position.
- `Alt+Enter` inserts a newline.
- `Enter` submits the message.
- `Ctrl+T` opens Tool-Set selection while chatting.
- `/tools` also switches Tool-Sets, `/exit` beendet den Client.


The client uses **LangChain chat models** and supports these Tool-Sets:
- `0`: kein Tool
- `1`: nur `physics-mcp`

If a Tool-Set includes `physics-mcp`, the client checks `PHYSICS_MCP_URL` via `/health` and prints a clear startup error if the server is offline.

If the model produces invalid tool arguments, the client returns structured tool errors back to the model so it can retry with corrected parameters instead of crashing.
Der Dev-Tool-Endpunkt normalisiert außerdem häufige Kurzschreibweisen aus LLM-Tool-Calls (z. B. `l`/`e`/`i`, `point_load_kn`, `udl_kn_per_m`) und kann den Lastfall bei fehlendem `case` aus den Lastparametern ableiten.


Beispiel-Prompts (Deutsch):

- "Für einen 6-m-Einfeldträger (E=210e9 Pa, I=8.5e-6 m^4) mit 12 kN Mittellast: Lagerreaktionen, maximales Biegemoment und maximale Durchbiegung."
- "Welche Annahmen macht das Beam-Modell und wo liegen die Grenzen für reale Stahlträger?"
- "Vergleiche einfach gelagerten Träger mit Einzellast vs. UDL bei gleicher Gesamtlast und gib die Deflektionsmaxima an."
- "Erzeuge mir eine JSON-Anfrage für `/dev/tools/solve_beam_case` für einen Kragträger mit UDL und 81 Samples."

## Deploy on Alpine Linux mini PC

```bash
cp .env.example .env
# fill keys as needed
docker compose up -d --build
```

This creates a small single-service deployment suitable for home-server use. Put a reverse proxy (Caddy/Nginx/Traefik) in front later for TLS and public exposure.

## Public exposure later (recommended)

1. Keep `physics-mcp` on private LAN at `:8080`.
2. Add reverse proxy with TLS certificates.
3. Forward `/mcp` and `/health`.
4. Add auth and real rate limiting at proxy level.
5. Keep app-level middleware hooks for future policies.

## Physics v0.1 limitations

- Euler-Bernoulli assumptions only.
- No shear deformation (no Timoshenko beam).
- No variable section/material.
- Cantilever point load only at free tip (defaults to x=length if omitted).
- Simply supported point load defaults to midspan if load position is omitted.
- UDL assumed full-length only.
- No unit conversion layer (SI input/output only).

## Suggested v0.2 priorities

- Add partial-span loads and off-tip cantilever point loads.
- Add `solve_catenary_case` with compatible schema.
- Add optional unit conversion at API boundary.
- Add auth + robust rate limiting.
- Add caching for repeated parameter sets.
