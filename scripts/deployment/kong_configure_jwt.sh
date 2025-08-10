#!/usr/bin/env bash
set -euo pipefail

# Config

# Signal handlers for graceful shutdown
cleanup_and_exit() {
    local exit_code="${1:-0}"
    echo "Script interrupted, cleaning up..." >&2
    # Clean up any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    exit "$exit_code"
}

trap 'cleanup_and_exit 130' INT
trap 'cleanup_and_exit 143' TERM
trap 'cleanup_and_exit 1' ERR

KONG_ADMIN_URL=${KONG_ADMIN_URL:-http://localhost:10007}
BACKEND_URL=${BACKEND_URL:-http://backend:10010}
BACKEND_SERVICE=${BACKEND_SERVICE:-backend}
BACKEND_ROUTE=${BACKEND_ROUTE:-backend}
BACKEND_PATH=${BACKEND_PATH:-/api/}

KONG_CONSUMER=${KONG_CONSUMER:-sutazai}
KONG_JWT_KEY=${KONG_JWT_KEY:-sutazai}
KONG_JWT_ALG=${KONG_JWT_ALG:-HS256}
JWT_SECRET_ENV=${JWT_SECRET:-}

DRY_RUN=${DRY_RUN:-0}
HTTP(){ if [ "$DRY_RUN" = "1" ]; then echo "DRY: curl -sS $*"; else curl -sS "$@"; fi }
HTTPW(){ if [ "$DRY_RUN" = "1" ]; then echo "DRY: curl -sS -w '%{http_code}' $*"; else curl -sS -w '%{http_code}' "$@"; fi }

log(){ echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')] $*"; }

require(){ command -v "$1" >/dev/null 2>&1 || { echo "Missing dependency: $1"; exit 1; }; }

require curl

log "Kong Admin: $KONG_ADMIN_URL"
code=$(HTTPW -o /dev/null "$KONG_ADMIN_URL/")
if [ "$code" != "200" ] && [ "$code" != "302" ]; then
  echo "Kong Admin not reachable (code $code) at $KONG_ADMIN_URL" >&2
  exit 1
fi

# Upsert backend service
log "Upsert service: $BACKEND_SERVICE -> $BACKEND_URL"
HTTP -X PUT "$KONG_ADMIN_URL/services/$BACKEND_SERVICE" \
  -d "url=$BACKEND_URL" | jq -r '.name // "ok"' >/dev/null || true

# Upsert backend route
log "Upsert route: $BACKEND_ROUTE paths=$BACKEND_PATH"
HTTP -X PUT "$KONG_ADMIN_URL/services/$BACKEND_SERVICE/routes/$BACKEND_ROUTE" \
  -d "strip_path=false" \
  -d "paths[]=$BACKEND_PATH" | jq -r '.name // "ok"' >/dev/null || true

# Enable JWT plugin on the service (idempotent: use PUT with name)
log "Enable JWT plugin on service: $BACKEND_SERVICE"
# Try to find existing jwt plugin id
PLUGIN_ID=$(HTTP "$KONG_ADMIN_URL/services/$BACKEND_SERVICE/plugins" | jq -r '.data[]?|select(.name=="jwt")|.id' | head -n1 || true)
if [ -z "$PLUGIN_ID" ]; then
  HTTP -X POST "$KONG_ADMIN_URL/services/$BACKEND_SERVICE/plugins" -d "name=jwt" >/dev/null
else
  log "JWT plugin already present: $PLUGIN_ID"
fi

# Create consumer
log "Upsert consumer: $KONG_CONSUMER"
# Kong lacks PUT for consumers; POST is idempotent on username
HTTP -X POST "$KONG_ADMIN_URL/consumers" -d "username=$KONG_CONSUMER" >/dev/null || true

# Create JWT credential for consumer
# If JWT_SECRET env present, use it; else generate random
if [ -z "$JWT_SECRET_ENV" ]; then
  if command -v openssl >/dev/null 2>&1; then
    JWT_SECRET_ENV=$(openssl rand -base64 64 | tr -d '\n')
  else
    JWT_SECRET_ENV=$(head -c 48 /dev/urandom | base64 | tr -d '\n')
  fi
  log "Generated JWT secret for consumer (not printed)"
else
  log "Using JWT_SECRET from environment"
fi

# Check if credential exists
HAS_CRED=$(HTTP "$KONG_ADMIN_URL/consumers/$KONG_CONSUMER/jwt" | jq -r '.data[]?|select(.key=="'$KONG_JWT_KEY'")|.id' | head -n1 || true)
if [ -z "$HAS_CRED" ]; then
  log "Create JWT credential (alg=$KONG_JWT_ALG, key=$KONG_JWT_KEY)"
  HTTP -X POST "$KONG_ADMIN_URL/consumers/$KONG_CONSUMER/jwt" \
    -d "algorithm=$KONG_JWT_ALG" \
    -d "key=$KONG_JWT_KEY" \
    -d "secret=$JWT_SECRET_ENV" >/dev/null
else
  log "JWT credential already exists for key=$KONG_JWT_KEY"
fi

log "Done. Test via: curl -i $KONG_ADMIN_URL/services/$BACKEND_SERVICE | jq ."
