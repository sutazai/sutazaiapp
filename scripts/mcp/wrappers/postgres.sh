#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/../_common.sh"

DB_URI="${DATABASE_URL:-${DATABASE_URI:-}}"
NET="${DOCKER_NETWORK:-sutazai-network}"
CONT_NAME="${POSTGRES_CONTAINER:-sutazai-postgres}"

if ! has_cmd docker; then
  err "Docker is required for postgres MCP (crystaldba/postgres-mcp)."
  exit 127
fi

# Resolve DATABASE_URL/DATABASE_URI from env or env files
resolve_db_uri() {
  if [ -n "${DATABASE_URL:-}" ]; then echo "$DATABASE_URL"; return 0; fi
  if [ -n "${DATABASE_URI:-}" ]; then echo "$DATABASE_URI"; return 0; fi
  local ROOT="/opt/sutazaiapp"
  local candidates=(
    "$ROOT/.env"
    "$ROOT/.env.production"
    "$ROOT/.env.production.secure"
    "$ROOT/.env.secure"
    "$ROOT/.env.example"
  )
  for f in "${candidates[@]}"; do
    if [ -f "$f" ]; then
      local uri
      uri=$(grep -E '^DATABASE_URI=' "$f" | tail -n1 | cut -d'=' -f2- || true)
      if [ -n "$uri" ]; then echo "$uri"; return 0; fi
      uri=$(grep -E '^DATABASE_URL=' "$f" | tail -n1 | cut -d'=' -f2- || true)
      if [ -n "$uri" ]; then echo "$uri"; return 0; fi
      # Construct from POSTGRES_* if possible
      set +u
      local U P D H
      U=$(grep -E '^POSTGRES_USER=' "$f" | tail -n1 | cut -d'=' -f2- || true)
      P=$(grep -E '^POSTGRES_PASSWORD=' "$f" | tail -n1 | cut -d'=' -f2- || true)
      D=$(grep -E '^POSTGRES_DB=' "$f" | tail -n1 | cut -d'=' -f2- || true)
      H=$(grep -E '^POSTGRES_HOST=' "$f" | tail -n1 | cut -d'=' -f2- || true)
      set -u
      U="${U:-sutazai}"; D="${D:-sutazai}"; H="${H:-postgres}"
      if [ -n "$P" ]; then echo "postgresql://${U}:${P}@${H}:5432/${D}"; return 0; fi
    fi
  done
  echo ""
}

# Build DB URI if not provided
if [ -z "$DB_URI" ]; then
  DB_URI="$(resolve_db_uri)"
fi

# Try to populate POSTGRES_* values for selfcheck convenience
POSTGRES_USER_VAL="${POSTGRES_USER:-}"
POSTGRES_PASSWORD_VAL="${POSTGRES_PASSWORD:-}"
POSTGRES_DB_VAL="${POSTGRES_DB:-}"
if [ -z "$POSTGRES_USER_VAL$POSTGRES_PASSWORD_VAL$POSTGRES_DB_VAL" ]; then
  for f in "/opt/sutazaiapp/.env" "/opt/sutazaiapp/.env.production" "/opt/sutazaiapp/.env.production.secure" "/opt/sutazaiapp/.env.secure" "/opt/sutazaiapp/.env.example"; do
    if [ -f "$f" ]; then
      POSTGRES_USER_VAL=${POSTGRES_USER_VAL:-$(grep -E '^POSTGRES_USER=' "$f" | tail -n1 | cut -d'=' -f2- || true)}
      POSTGRES_PASSWORD_VAL=${POSTGRES_PASSWORD_VAL:-$(grep -E '^POSTGRES_PASSWORD=' "$f" | tail -n1 | cut -d'=' -f2- || true)}
      POSTGRES_DB_VAL=${POSTGRES_DB_VAL:-$(grep -E '^POSTGRES_DB=' "$f" | tail -n1 | cut -d'=' -f2- || true)}
    fi
  done
fi

if [ "${1:-}" = "--selfcheck" ]; then
  section "Postgres MCP selfcheck $(ts)"
  if ! has_cmd docker; then err_line "docker not found"; exit 127; else ok_line "docker present"; fi
  if docker network ls --format '{{.Name}}' | grep -qx "$NET"; then ok_line "network $NET exists"; else err_line "network $NET missing"; exit 127; fi
  if docker ps --format '{{.Names}}' | grep -qx "$CONT_NAME"; then ok_line "container $CONT_NAME running"; else err_line "container $CONT_NAME not running"; exit 127; fi
  if docker exec "$CONT_NAME" pg_isready -U "${POSTGRES_USER_VAL:-${POSTGRES_USER:-sutazai}}" -d "${POSTGRES_DB_VAL:-${POSTGRES_DB:-sutazai}}" >/dev/null 2>&1; then ok_line "pg_isready OK"; else warn_line "pg_isready not ready"; fi
  if docker exec -e PGPASSWORD="${POSTGRES_PASSWORD_VAL:-${POSTGRES_PASSWORD:-}}" "$CONT_NAME" psql -U "${POSTGRES_USER_VAL:-${POSTGRES_USER:-sutazai}}" -d "${POSTGRES_DB_VAL:-${POSTGRES_DB:-sutazai}}" -c 'SELECT 1;' >/dev/null 2>&1; then ok_line "psql SELECT 1 OK"; else warn_line "psql SELECT 1 failed"; fi
  if [ -n "$DB_URI" ]; then ok_line "DATABASE_URL/URI resolved"; else err_line "DATABASE_URL/URI empty"; exit 127; fi
  exit 0
fi

# Validate network exists
if ! docker network ls --format '{{.Name}}' | grep -qx "$NET"; then
  err "Docker network '$NET' not found. Create it: docker network create $NET"
  exit 127
fi

# Validate container is running
if ! docker ps --format '{{.Names}}' | grep -qx "$CONT_NAME"; then
  err "Postgres container '$CONT_NAME' is not running. Start it: docker compose up -d postgres"
  exit 127
fi

# Readiness via docker exec (no image pull)
if docker exec "$CONT_NAME" pg_isready -U "${POSTGRES_USER:-sutazai}" -d "${POSTGRES_DB:-sutazai}" >/dev/null 2>&1; then
  ok "Postgres is accepting connections"
else
  warn "Postgres not ready yet; MCP will still start (may retry internally)"
fi

exec docker run --network "$NET" --rm -i -e DATABASE_URI="$DB_URI" crystaldba/postgres-mcp --access-mode=restricted
