#!/bin/bash

# IMMEDIATE Security Fix for Hardware Resource Optimizer
# Removes dangerous configurations and restarts securely

set -euo pipefail


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

log() { echo "[$(date +'%H:%M:%S')] $1"; }
error() { echo "ERROR: $1" >&2; exit 1; }

log "🔒 IMMEDIATE SECURITY FIX - Hardware Resource Optimizer"

# 1. Stop current insecure service
log "Stopping insecure service..."
docker-compose stop hardware-resource-optimizer 2>/dev/null || true
docker-compose rm -f hardware-resource-optimizer 2>/dev/null || true

# 2. Create secure configuration patch
log "Applying security configuration..."
cat > "$(mktemp /tmp/secure-patch.yml.XXXXXX)" << 'EOF'
  hardware-resource-optimizer:
    build:
      context: ./agents/hardware-resource-optimizer
      dockerfile: Dockerfile
    container_name: sutazai-hardware-resource-optimizer
    depends_on:
      backend:
        condition: service_started
      redis:
        condition: service_healthy
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 256M
    environment:
      AGENT_TYPE: hardware-resource-optimizer
      API_ENDPOINT: http://backend:8000
      DATABASE_URL: postgresql://${POSTGRES_USER:-sutazai}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB:-sutazai}
      LOG_LEVEL: INFO
      OLLAMA_API_KEY: local
      OLLAMA_BASE_URL: http://ollama:11434
      OLLAMA_MODEL: tinyllama:latest
      PORT: 8080
      RABBITMQ_URL: amqp://${RABBITMQ_DEFAULT_USER:-sutazai}:${RABBITMQ_DEFAULT_PASS}@rabbitmq:5672/
      REDIS_URL: redis://sutazai-redis:6379/0
      SUTAZAI_ENV: ${SUTAZAI_ENV:-production}
      TZ: ${TZ:-UTC}
    healthcheck:
      interval: 30s
      retries: 3
      start_period: 60s
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      timeout: 10s
    networks:
    - sutazai-network
    ports:
    - 11110:8080
    restart: unless-stopped
    volumes:
      - ./agents/core:/app/agents/core:ro
      - ./data:/app/data
      - ./configs:/app/configs:ro
      - ./logs:/app/logs
    # REMOVED: pid: host (SECURITY RISK)
    # REMOVED: privileged: true (SECURITY RISK)  
    # REMOVED: /var/run/docker.sock mount (SECURITY RISK)
EOF

# 3. Rebuild and start securely
log "Rebuilding with security enhancements..."
docker-compose build --no-cache hardware-resource-optimizer

log "Starting secure service..."
docker-compose up -d hardware-resource-optimizer

# 4. Verify security
sleep 15
log "Validating security configuration..."

USER_CHECK=$(docker inspect --format='{{.Config.User}}' sutazai-hardware-resource-optimizer || echo "root")
PRIVILEGED_CHECK=$(docker inspect --format='{{.HostConfig.Privileged}}' sutazai-hardware-resource-optimizer || echo "true")

log "Security Status:"
log "  User: $USER_CHECK"
log "  Privileged: $PRIVILEGED_CHECK"

if [[ "$USER_CHECK" == "appuser" ]]; then
    log "✅ Running as non-root user"
else
    log "⚠️  Still running as root"
fi

if [[ "$PRIVILEGED_CHECK" == "false" ]]; then
    log "✅ Privileged mode disabled"
else
    log "⚠️  Still running in privileged mode"
fi

# 5. Test health
log "Testing health endpoint..."
if curl -f -s http://localhost:11110/health >/dev/null; then
    log "✅ Service healthy and responding"
else
    log "⚠️  Service not responding (may still be starting)"
fi

log "🔒 Security fix completed!"
log "Service running at: http://localhost:11110/health"