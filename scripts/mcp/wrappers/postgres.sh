#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/../_common.sh"

DB_URI="${DATABASE_URI:-}"
NET="${DOCKER_NETWORK:-sutazai-network}"
CONT_NAME="${POSTGRES_CONTAINER:-sutazai-postgres}"

if ! has_cmd docker; then
  err "Docker is required for postgres MCP (crystaldba/postgres-mcp)."
  exit 127
fi

# Build DB URI from .env if not provided
if [ -z "$DB_URI" ] && [ -f "/opt/sutazaiapp/.env" ]; then
  # shellcheck disable=SC2046
  set +u
  # Read only the required keys to avoid exporting all
  POSTGRES_USER_VAL=$(grep -E '^POSTGRES_USER=' /opt/sutazaiapp/.env | tail -n1 | cut -d'=' -f2- || true)
  POSTGRES_PASSWORD_VAL=$(grep -E '^POSTGRES_PASSWORD=' /opt/sutazaiapp/.env | tail -n1 | cut -d'=' -f2- || true)
  POSTGRES_DB_VAL=$(grep -E '^POSTGRES_DB=' /opt/sutazaiapp/.env | tail -n1 | cut -d'=' -f2- || true)
  HOST_VAL=$(grep -E '^POSTGRES_HOST=' /opt/sutazaiapp/.env | tail -n1 | cut -d'=' -f2- || echo postgres)
  set -u
  DB_URI="postgresql://${POSTGRES_USER_VAL:-sutazai}:${POSTGRES_PASSWORD_VAL}@${HOST_VAL:-postgres}:5432/${POSTGRES_DB_VAL:-sutazai}"
fi

if [ "${1:-}" = "--selfcheck" ]; then
  section "Postgres MCP selfcheck $(ts)"
  if ! has_cmd docker; then err_line "docker not found"; exit 127; else ok_line "docker present"; fi
  if docker network ls --format '{{.Name}}' | grep -qx "$NET"; then ok_line "network $NET exists"; else err_line "network $NET missing"; exit 127; fi
  if docker ps --format '{{.Names}}' | grep -qx "$CONT_NAME"; then ok_line "container $CONT_NAME running"; else err_line "container $CONT_NAME not running"; exit 127; fi
  if docker exec "$CONT_NAME" pg_isready -U "${POSTGRES_USER:-sutazai}" -d "${POSTGRES_DB:-sutazai}" >/dev/null 2>&1; then ok_line "pg_isready OK"; else warn_line "pg_isready not ready"; fi
  if docker exec -e PGPASSWORD="${POSTGRES_PASSWORD:-${POSTGRES_PASSWORD_VAL:-}}" "$CONT_NAME" psql -U "${POSTGRES_USER:-${POSTGRES_USER_VAL:-sutazai}}" -d "${POSTGRES_DB:-${POSTGRES_DB_VAL:-sutazai}}" -c 'SELECT 1;' >/dev/null 2>&1; then ok_line "psql SELECT 1 OK"; else warn_line "psql SELECT 1 failed"; fi
  if [ -n "$DB_URI" ]; then ok_line "DATABASE_URI set"; else err_line "DATABASE_URI empty"; exit 127; fi
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
