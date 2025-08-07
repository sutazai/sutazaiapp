#!/usr/bin/env bash
# Health verification and smoke tests for core services
#
# Usage:
#   scripts/devops/check_services_health.sh \
#     --ollama-host localhost --ollama-port 10104 \
#     --kong-host localhost --kong-port 10005 \
#     --consul-host localhost --consul-port 10006 \
#     --vector-start 10100 --vector-end 10103
#
# Notes:
# - Avoids hardcoded values via CLI flags (getopts parsing)
# - Safe to run repeatedly; prints detailed, timestamped logs
# - Uses curl and bash /dev/tcp checks to validate reachability and latency

set -euo pipefail

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[${1}] $(ts) ${2}"; }

OLLAMA_HOST=localhost
OLLAMA_PORT=10104
KONG_HOST=localhost
KONG_PORT=10005
CONSUL_HOST=localhost
CONSUL_PORT=10006
VECTOR_START=10100
VECTOR_END=10103

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ollama-host) OLLAMA_HOST="$2"; shift 2;;
    --ollama-port) OLLAMA_PORT="$2"; shift 2;;
    --kong-host) KONG_HOST="$2"; shift 2;;
    --kong-port) KONG_PORT="$2"; shift 2;;
    --consul-host) CONSUL_HOST="$2"; shift 2;;
    --consul-port) CONSUL_PORT="$2"; shift 2;;
    --vector-start) VECTOR_START="$2"; shift 2;;
    --vector-end) VECTOR_END="$2"; shift 2;;
    --help|-h)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
    *) log ERROR "Unknown option: $1"; exit 2;;
  esac
done

check_tcp() {
  local host="$1" port="$2" name="$3"
  local start end
  start=$(date +%s%3N || date +%s)
  if (echo > "/dev/tcp/${host}/${port}") >/dev/null 2>&1; then
    end=$(date +%s%3N || date +%s)
    local ms=$((end-start))
    log INFO "${name} TCP reachable at ${host}:${port} (latency ~${ms}ms)"
    return 0
  else
    log ERROR "${name} not reachable at ${host}:${port}"
    return 1
  fi
}

check_http() {
  local url="$1" name="$2"
  local out
  out=$(curl -sS -o /dev/null -w 'code=%{http_code} time=%{time_total}' "$url" || true)
  log INFO "${name} ${url} -> ${out}"
  [[ "$out" == code=200* ]] || return 1
}

overall=0

# Ollama + TinyLlama (HTTP API)
check_tcp "$OLLAMA_HOST" "$OLLAMA_PORT" "Ollama" || overall=1
check_http "http://${OLLAMA_HOST}:${OLLAMA_PORT}/" "Ollama-root" || true

# Kong Admin/Gateway
check_tcp "$KONG_HOST" "$KONG_PORT" "Kong" || overall=1
check_http "http://${KONG_HOST}:${KONG_PORT}/" "Kong-root" || true

# Consul HTTP
check_tcp "$CONSUL_HOST" "$CONSUL_PORT" "Consul" || overall=1
check_http "http://${CONSUL_HOST}:${CONSUL_PORT}/v1/status/leader" "Consul-leader" || true

# Vector DB services (range)
for p in $(seq "$VECTOR_START" "$VECTOR_END"); do
  check_tcp localhost "$p" "VectorDB:${p}" || overall=1
done

if [[ "$overall" -eq 0 ]]; then
  log INFO "All critical services reachable"
else
  log ERROR "One or more services failed reachability checks"
fi

exit "$overall"

