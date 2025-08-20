#!/usr/bin/env bash
set -euo pipefail

# Health Sweep Script
# - Summarizes docker compose service states
# - Highlights unhealthy/restarting/exited services
# - Shows last log lines for failing services
# - Pings key HTTP health endpoints

echo "=== SutazAI Health Sweep ==="

# Ensure we're in repo root (script can be called from anywhere)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Compose files: allow override via COMPOSE_FILE, else default to public override
if [[ -z "${COMPOSE_FILE:-}" ]]; then
  export COMPOSE_FILE="docker-compose.yml:docker-compose.public-images.override.yml"
fi

# Ensure network exists (idempotent)
docker network create sutazai-network >/dev/null 2>&1 || true

echo "Compose files: $COMPOSE_FILE"
echo "Project: $(basename "$REPO_ROOT")"
echo

echo "--- docker compose ps ---"
docker compose ps || true
echo

echo "--- Detecting problematic services (unhealthy/restarting/exited) ---"
PROBLEM_SERVICES=()
while IFS= read -r svc; do
  [[ -z "$svc" ]] && continue
  cid=$(docker compose ps -q "$svc" 2>/dev/null || true)
  if [[ -z "$cid" ]]; then
    # Not created or stopped before creation
    echo "[WARN] $svc: container not created"
    PROBLEM_SERVICES+=("$svc")
    continue
  fi
  status=$(docker inspect --format='{{.State.Status}}' "$cid" 2>/dev/null || echo unknown)
  health=$(docker inspect --format='{{if .State.Health}}{{.State.Health.Status}}{{end}}' "$cid" 2>/dev/null || true)
  effective=${health:-$status}
  case "$effective" in
    healthy|running)
      :
      ;;
    *)
      echo "[ISSUE] $svc: status=$status health=${health:-none}"
      PROBLEM_SERVICES+=("$svc")
      ;;
  esac
done < <(docker compose ps --services 2>/dev/null || true)

if (( ${#PROBLEM_SERVICES[@]} )); then
  echo
  echo "--- Recent logs for problematic services (tail 200) ---"
  for s in "${PROBLEM_SERVICES[@]}"; do
    echo
    echo "===== $s ====="
    docker compose logs --tail=200 "$s" 2>&1 | tail -n 200 || echo "[logs unavailable]"
  done
else
  echo "All services healthy."
fi

echo
echo "--- HTTP health probes (best effort) ---"
probe() {
  local name="$1" url="$2"
  printf "%-12s %s\n" "$name:" "$url"
  curl -fsS --max-time 5 "$url" | head -c 200 || echo "[unreachable]"
  echo
}

probe Backend   "http://localhost:10010/health"
probe Frontend  "http://localhost:10011/"
probe Ollama    "http://localhost:10104/api/tags"
probe Chroma    "http://localhost:10100/api/v1/heartbeat"
probe Qdrant    "http://localhost:10101/healthz"
probe Prometheus "http://localhost:10200/-/healthy"
probe Grafana   "http://localhost:10201/api/health"
probe Jaeger    "http://localhost:10210/"

echo "=== Health sweep complete ==="

