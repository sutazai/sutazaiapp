# Local Development Guide (Public Images + Overrides)

To run the stack locally without private `sutazai-*-secure` images:

- Ensure Docker network: `docker network create sutazai-network || true`
- Provide required secrets in `.env` (see `.env.example`, `.env.production.secure`):
  - `POSTGRES_PASSWORD`, `JWT_SECRET_KEY`, `SECRET_KEY`, `NEO4J_PASSWORD`, `GRAFANA_PASSWORD`, `RABBITMQ_DEFAULT_PASS`, etc.
- Start with public image override:

```
docker compose \
  -f docker-compose.yml \
  -f docker-compose.public-images.override.yml \
  up -d
```

Notes:
- Backend expects strict auth env; set `JWT_SECRET_KEY` (>=32 chars) and `SECRET_KEY`.
- Service DNS is consistent: `sutazai-postgres`, `sutazai-redis`, `sutazai-neo4j`, etc.
- MCP configuration is not modified here (Rule 20).

## Troubleshooting
- If backend build installs heavy ML deps slowly, consider building once and reusing the image tag.
- Use `docker compose logs -f backend` to view auth/env validation messages.
- If ports collide, adjust exposed ports in an additional local override.
