#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/../_common.sh"

# Unified Development Service MCP Wrapper
# Consolidates ultimatecoder, language-server, and sequentialthinking
# Target memory: 512MB

SERVICE_NAME="unified-dev"
SERVICE_PORT="${MCP_UNIFIED_DEV_PORT:-4000}"
SERVICE_HOST="${MCP_UNIFIED_DEV_HOST:-127.0.0.1}"
MAX_MEMORY_MB="${MCP_UNIFIED_DEV_MAX_MB:-512}"
MAX_INSTANCES="${MCP_UNIFIED_DEV_MAX_INSTANCES:-2}"

# Service paths
DOCKER_IMAGE="sutazai-mcp-unified:latest"
CONTAINER_NAME="mcp-unified-dev"
SERVICE_DIR="/opt/sutazaiapp/docker/mcp-services/unified-dev"
LOG_FILE="/var/log/mcp/unified-dev.log"

if [ "${1:-}" = "--selfcheck" ]; then
  section "Unified Development Service selfcheck $(ts)"
  
  # Check Docker image
  if docker image inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
    ok_line "Docker image present: $DOCKER_IMAGE"
  else
    warn_line "Docker image missing: $DOCKER_IMAGE"
    info_line "Building image from $SERVICE_DIR"
    
    if [ -f "$SERVICE_DIR/Dockerfile" ]; then
      docker build -t "$DOCKER_IMAGE" "$SERVICE_DIR" || {
        err_line "Failed to build Docker image"
        exit 127
      }
      ok_line "Docker image built successfully"
    else
      err_line "Dockerfile missing at $SERVICE_DIR/Dockerfile"
      exit 127
    fi
  fi
  
  # Check service directory
  if [ -d "$SERVICE_DIR" ]; then
    ok_line "Service directory present: $SERVICE_DIR"
  else
    err_line "Service directory missing: $SERVICE_DIR"
    exit 127
  fi
  
  # Check Node.js source
  if [ -f "$SERVICE_DIR/src/unified-dev-server.js" ]; then
    ok_line "Main service file present"
  else
    err_line "Main service file missing: $SERVICE_DIR/src/unified-dev-server.js"
    exit 127
  fi
  
  # Check port availability
  if ! nc -z "$SERVICE_HOST" "$SERVICE_PORT" 2>/dev/null; then
    ok_line "Port $SERVICE_PORT available"
  else
    warn_line "Port $SERVICE_PORT already in use"
  fi
  
  # Check memory requirements
  AVAILABLE_MB=$(free -m | awk 'NR==2{print $7}')
  if [ "$AVAILABLE_MB" -gt "$MAX_MEMORY_MB" ]; then
    ok_line "Sufficient memory available: ${AVAILABLE_MB}MB > ${MAX_MEMORY_MB}MB"
  else
    warn_line "Low memory: ${AVAILABLE_MB}MB available, ${MAX_MEMORY_MB}MB required"
  fi
  
  info_line "Configuration:"
  info_line "  Service: $SERVICE_NAME"
  info_line "  Port: $SERVICE_PORT"
  info_line "  Memory Limit: ${MAX_MEMORY_MB}MB"
  info_line "  Max Instances: $MAX_INSTANCES"
  info_line "  Container: $CONTAINER_NAME"
  
  exit 0
fi

if [ "${1:-}" = "health" ]; then
  # Health check for Docker container
  if docker ps --filter "name=$CONTAINER_NAME" --filter "status=running" | grep -q "$CONTAINER_NAME"; then
    # Container is running, check HTTP health
    if curl -f -s "http://$SERVICE_HOST:$SERVICE_PORT/health" >/dev/null 2>&1; then
      echo "healthy"
      exit 0
    else
      echo "unhealthy: HTTP health check failed"
      exit 1
    fi
  else
    echo "unhealthy: container not running"
    exit 1
  fi
fi

prune_old_instances() {
  # Clean up old/stale unified-dev containers
  local running_containers
  running_containers=$(docker ps --filter "name=mcp-unified-dev" --format "{{.Names}}" | wc -l)
  
  if [ "$running_containers" -gt "$MAX_INSTANCES" ]; then
    warn "Too many unified-dev containers running: $running_containers (max: $MAX_INSTANCES)"
    
    # Get oldest containers
    docker ps --filter "name=mcp-unified-dev" --format "{{.Names}} {{.CreatedAt}}" | \
      sort -k2 | \
      head -n $((running_containers - MAX_INSTANCES)) | \
      while read -r container_name _; do
        warn_line "Stopping old container: $container_name"
        docker stop "$container_name" >/dev/null 2>&1 || true
        docker rm "$container_name" >/dev/null 2>&1 || true
      done
  fi
  
  # Clean up exited containers
  docker ps -a --filter "name=mcp-unified-dev" --filter "status=exited" -q | \
    xargs -r docker rm >/dev/null 2>&1 || true
}

start_service() {
  # Ensure log directory exists
  mkdir -p "$(dirname "$LOG_FILE")"
  
  # Stop existing container if running
  if docker ps --filter "name=$CONTAINER_NAME" --filter "status=running" | grep -q "$CONTAINER_NAME"; then
    info "Stopping existing container: $CONTAINER_NAME"
    docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
    docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
  fi
  
  # Prune old instances
  prune_old_instances
  
  # Start new container
  info "Starting unified development service on port $SERVICE_PORT"
  
  docker run \
    --name "$CONTAINER_NAME" \
    --detach \
    --restart unless-stopped \
    --publish "$SERVICE_PORT:4000" \
    --memory "${MAX_MEMORY_MB}m" \
    --memory-swap "${MAX_MEMORY_MB}m" \
    --cpus "2.0" \
    --env NODE_ENV=production \
    --env MCP_SERVICE="$SERVICE_NAME" \
    --env MCP_HOST=0.0.0.0 \
    --env MCP_PORT=4000 \
    --env MAX_INSTANCES="$MAX_INSTANCES" \
    --env NODE_OPTIONS="--max-old-space-size=$MAX_MEMORY_MB" \
    --volume mcp-unified-dev-data:/var/lib/mcp \
    --volume mcp-logs:/var/log/mcp \
    --network mcp-bridge \
    --log-driver json-file \
    --log-opt max-size=10m \
    --log-opt max-file=3 \
    "$DOCKER_IMAGE" || {
      err "Failed to start container: $CONTAINER_NAME"
      exit 1
    }
  
  # Wait for service to be ready
  local max_wait=30
  local wait_count=0
  
  while [ $wait_count -lt $max_wait ]; do
    if curl -f -s "http://$SERVICE_HOST:$SERVICE_PORT/health" >/dev/null 2>&1; then
      ok "Unified development service ready on port $SERVICE_PORT"
      break
    fi
    
    sleep 1
    wait_count=$((wait_count + 1))
  done
  
  if [ $wait_count -eq $max_wait ]; then
    err "Service failed to become ready within ${max_wait} seconds"
    docker logs "$CONTAINER_NAME" --tail 20
    exit 1
  fi
}

# Check if we should use Docker or direct execution
if has_cmd docker; then
  start_service
else
  # Fallback to direct Node.js execution (development mode)
  if [ ! -f "$SERVICE_DIR/src/unified-dev-server.js" ]; then
    err "Service file not found: $SERVICE_DIR/src/unified-dev-server.js"
    exit 127
  fi
  
  cd "$SERVICE_DIR"
  
  # Set environment variables
  export NODE_ENV=production
  export MCP_SERVICE="$SERVICE_NAME"
  export MCP_HOST="$SERVICE_HOST"
  export MCP_PORT="$SERVICE_PORT"
  export NODE_OPTIONS="--max-old-space-size=$MAX_MEMORY_MB"
  export MAX_INSTANCES="$MAX_INSTANCES"
  
  info "Starting unified development service directly (port $SERVICE_PORT)"
  exec node src/unified-dev-server.js
fi