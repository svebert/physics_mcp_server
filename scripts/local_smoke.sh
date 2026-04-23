#!/usr/bin/env sh
set -eu

BASE_URL="${1:-http://127.0.0.1:8080}"

echo "== health =="
curl -sS "$BASE_URL/health" | jq .

echo "\n== supported cases =="
curl -sS -X POST "$BASE_URL/dev/tools/get_supported_cases" -H 'content-type: application/json' -d '{}' | jq .

echo "\n== solve sample =="
curl -sS -X POST "$BASE_URL/dev/tools/solve_beam_case" \
  -H 'content-type: application/json' \
  -d @scripts/sample_request.json | jq '{case, max_bending_moment_nm, max_deflection_m, reactions_n}'
