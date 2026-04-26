# Public Exposure Guide (Cloudflare Tunnel + Nginx)

Diese Anleitung beschreibt eine produktive Exposition von `physics-mcp` unter deiner Domain `svebert.org` mit bestehendem Nginx, TLS-Zertifikat und Cloudflare Tunnel.

## Zielbild

- `physics-mcp` läuft intern im Docker-Netz (kein direkter Public Port notwendig).
- Nginx terminiert TLS (dein vorhandenes Zertifikat).
- Cloudflare Tunnel veröffentlicht nur den gewünschten Hostnamen nach außen.
- Extern verfügbar:
  - `https://mcp.svebert.org/health`
  - `https://mcp.svebert.org/mcp`

## 1) physics-mcp im selben Docker-Netz wie Nginx bereitstellen

Beispiel `docker-compose.yml` für `physics-mcp`:

```yaml
services:
  physics-mcp:
    build: .
    container_name: physics-mcp
    restart: unless-stopped
    env_file:
      - .env
    environment:
      - PHYSICS_MCP_HOST=0.0.0.0
      - PHYSICS_MCP_PORT=8080
      - PHYSICS_MCP_LOG_LEVEL=info
    expose:
      - "8080"
    networks:
      - shared_network

networks:
  shared_network:
    external: true
```

Hinweis: `ports:` ist nicht nötig, wenn nur Nginx den Service erreichen soll.

## 2) Nginx vHost für mcp.svebert.org ergänzen

In `nginx/conf.d/default.conf` einen neuen `server`-Block hinzufügen:

```nginx
server {
    listen 443 ssl;
    server_name mcp.svebert.org;

    ssl_certificate /etc/nginx/ssl/live/svebert.org/fullchain.pem;
    ssl_certificate_key /etc/nginx/ssl/live/svebert.org/privkey.pem;

    # Optional: einfache Schutzmaßnahmen
    limit_req  zone=global_req  burst=20  nodelay;
    limit_conn global_conn 10;

    location /health {
        proxy_pass http://physics-mcp:8080/health;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_redirect off;
    }

    location /mcp {
        proxy_http_version 1.1;
        proxy_pass http://physics-mcp:8080/mcp;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $connection_upgrade;
        proxy_read_timeout 3600;
        proxy_redirect off;
    }
}
```

Anschließend Nginx neu laden:

```bash
docker exec nginx nginx -t
# wenn ok:
docker exec nginx nginx -s reload
```

## 3) Cloudflare Tunnel Route konfigurieren

Im Cloudflare Zero Trust Dashboard (oder per `cloudflared` config) für den Tunnel eine Public Hostname Route setzen:

- Hostname: `mcp.svebert.org`
- Service: `https://nginx` (wenn `cloudflared` im gleichen Docker-Netz läuft)
  - alternativ auf den internen Host/Port, auf dem Nginx erreichbar ist.

Wenn dein Tunnel-Container im `tunnel_network` hängt, stelle sicher, dass Nginx ebenfalls dort hängt (bei dir bereits der Fall).

## 4) DNS und SSL Mode in Cloudflare

- DNS: CNAME für `mcp` auf den Tunnel (wird i. d. R. automatisch bei Public Hostname erzeugt).
- SSL/TLS Mode in Cloudflare: **Full (strict)**, da du ein gültiges Zertifikat auf Nginx hast.

## 5) Validierung

Von extern testen:

```bash
curl -i https://mcp.svebert.org/health
```

Erwartet: HTTP 200 und JSON mit `status: ok`.

MCP Endpoint testen (Beispiel):

```bash
curl -i https://mcp.svebert.org/mcp
```

(Je nach MCP-Handshake kann die Antwort nicht einfach ein 200-JSON sein; wichtig ist, dass der Endpoint erreichbar ist.)

## 6) Client-Konfiguration

Für deinen Client:

```bash
export PHYSICS_MCP_URL="https://mcp.svebert.org"
physics-mcp-client
```

## 7) Härtungsempfehlungen

- Optional HTTP Basic Auth oder Access-Policy vor `/mcp` (Cloudflare Access).
- Zusätzliche IP-Allowlist (z. B. nur Cloudflare egress ranges) falls gewünscht.
- Rate Limits (hast du bereits global) und Monitoring (5xx/latency).
- Regelmäßige Updates von `cloudflared`, Nginx und Python Base Image.
