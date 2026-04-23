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
- `client/`: minimal OpenAI CLI tool-calling chat client.
- `tests/`: unit, validation, integration-like, and smoke tests.

## Repository tree

```text
physics-mcp/
├── client/
│   └── openai_chat.py
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
pip install -e '.[dev]'
cp .env.example .env
# put OPENAI_API_KEY into .env for the client
```

## Environment variables

- `OPENAI_API_KEY`: required for test chat client.
- `OPENAI_MODEL`: optional (default: `gpt-4o-mini`).
- `PHYSICS_MCP_URL`: optional for client (default: `http://127.0.0.1:8080`).
- `PHYSICS_MCP_HOST`: server bind host (default: `0.0.0.0`).
- `PHYSICS_MCP_PORT`: server port (default: `8080`).

## How to get an OpenAI API key

1. Create or log into your OpenAI account at https://platform.openai.com/.
2. Open **API keys** in the dashboard: https://platform.openai.com/api-keys.
3. Create a new secret key and copy it once (it is shown only at creation).
4. Add it to your local environment:

```bash
export OPENAI_API_KEY='sk-...'
```

## Run tests

```bash
pytest
```


If `physics-mcp-client` is not found, reinstall in your active venv:

```bash
python3 -m pip install -e '.[dev]'
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

## OpenAI test client

Run with MCP server available locally:

```bash
# uses OPENAI_API_KEY from .env or current shell
physics-mcp-client
# fallback without console script:
python3 -m client.openai_chat
```

Ask something like:

- "For a 6 m simply supported steel beam (E=210e9 Pa, I=8.5e-6 m^4) with a 12 kN center point load, what are the reactions, maximum moment, and maximum deflection?"

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
- Cantilever point load only at free tip.
- UDL assumed full-length only.
- No unit conversion layer (SI input/output only).

## Suggested v0.2 priorities

- Add partial-span loads and off-tip cantilever point loads.
- Add `solve_catenary_case` with compatible schema.
- Add optional unit conversion at API boundary.
- Add auth + robust rate limiting.
- Add caching for repeated parameter sets.
