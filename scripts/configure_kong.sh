#!/usr/bin/env bash
# Idempotent Kong configuration helper
#
# Usage:
#   scripts/configure_kong.sh <service_name> <path_prefix>
# Example:
#   scripts/configure_kong.sh jarvis-task-controller /task
#
# Behavior:
# - Creates/updates a Kong Service that targets the Consul DNS for the service
#   e.g., http://<service_name>.service.consul:8000
# - Creates a Kong Route with the provided path prefix
# - Idempotent: safe to run repeatedly without duplicating entities

set -euo pipefail

KONG_ADMIN_URL=${KONG_ADMIN_URL:-"http://localhost:8001"}

log() {
  echo "[${1}] $(date '+%Y-%m-%d %H:%M:%S') ${2}" >&2
}

die() {
  log ERROR "$1"; exit 1
}

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <service_name> <path_prefix>" >&2
  exit 2
fi

SERVICE_NAME="$1"
PATH_PREFIX="$2"

if [[ -z "$SERVICE_NAME" || -z "$PATH_PREFIX" ]]; then
  die "service_name and path_prefix are required"
fi

SERVICE_URL="http://${SERVICE_NAME}.service.consul:8000"

# Create or update Service
log INFO "Ensuring Kong Service '${SERVICE_NAME}' -> ${SERVICE_URL}"
HTTP_CODE=$(curl -sS -o /dev/null -w "%{http_code}" "${KONG_ADMIN_URL}/services/${SERVICE_NAME}") || die "Failed to reach Kong Admin API"
if [[ "$HTTP_CODE" == "200" ]]; then
  # Update existing service if needed
  curl -sS -X PATCH "${KONG_ADMIN_URL}/services/${SERVICE_NAME}" \
       -H 'Content-Type: application/json' \
       -d "{\"url\": \"${SERVICE_URL}\"}" \
       >/dev/null && log INFO "Service '${SERVICE_NAME}' updated"
elif [[ "$HTTP_CODE" == "404" ]]; then
  curl -sS -X POST "${KONG_ADMIN_URL}/services" \
       -H 'Content-Type: application/json' \
       -d "{\"name\": \"${SERVICE_NAME}\", \"url\": \"${SERVICE_URL}\"}" \
       >/dev/null && log INFO "Service '${SERVICE_NAME}' created"
else
  die "Unexpected HTTP ${HTTP_CODE} when querying service"
fi

# Ensure Route for path prefix exists
log INFO "Ensuring Route for '${SERVICE_NAME}' with path '${PATH_PREFIX}'"
ROUTES_JSON=$(curl -sS "${KONG_ADMIN_URL}/routes?service.name=${SERVICE_NAME}") || die "Failed to list routes"
if echo "$ROUTES_JSON" | grep -q '"data"'; then
  if echo "$ROUTES_JSON" | grep -q "\"paths\":\[\"${PATH_PREFIX}\"\]"; then
    log INFO "Route already exists for path '${PATH_PREFIX}'"
  else
    curl -sS -X POST "${KONG_ADMIN_URL}/routes" \
         -H 'Content-Type: application/json' \
         -d "{\"name\": \"${SERVICE_NAME}-route\", \"service\":{\"name\":\"${SERVICE_NAME}\"}, \"paths\":[\"${PATH_PREFIX}\"]}" \
         >/dev/null && log INFO "Route created for path '${PATH_PREFIX}'"
  fi
else
  die "Unexpected routes response from Kong"
fi

log INFO "Kong configuration complete for '${SERVICE_NAME}'"

