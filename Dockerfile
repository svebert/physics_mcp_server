FROM python:3.11-alpine

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apk add --no-cache build-base

COPY pyproject.toml README.md /app/
COPY physics_core /app/physics_core
COPY mcp_server /app/mcp_server
COPY smart_mcp_server /app/smart_mcp_server
COPY client /app/client

RUN pip install --no-cache-dir .

EXPOSE 8080

CMD ["physics-mcp-server"]
