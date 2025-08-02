#!/bin/bash
# Deploy Self-Healing Components for SutazAI

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SELF_HEALING_DIR="$PROJECT_ROOT/self-healing"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== Deploying SutazAI Self-Healing System ===${NC}"

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo -e "${RED}Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

# Check if Redis is running
if ! docker ps | grep -q sutazai-redis; then
    echo -e "${RED}Redis container is not running. Please start the core services first.${NC}"
    exit 1
fi

# Create necessary directories
echo -e "${YELLOW}Creating directories...${NC}"
mkdir -p "$SELF_HEALING_DIR"/{logs,data,recovery/snapshots}

# Make scripts executable
echo -e "${YELLOW}Setting script permissions...${NC}"
chmod +x "$SELF_HEALING_DIR"/scripts/*.py
chmod +x "$SELF_HEALING_DIR"/scripts/*.sh

# Install Python dependencies
echo -e "${YELLOW}Installing Python dependencies...${NC}"
cat > "$SELF_HEALING_DIR/requirements.txt" << EOF
pyyaml==6.0
redis==4.5.4
docker==6.1.2
psutil==5.9.5
requests==2.31.0
psycopg2-binary==2.9.6
prometheus-client==0.17.0
EOF

# Create virtual environment if it doesn't exist
if [ ! -d "$SELF_HEALING_DIR/venv" ]; then
    python3 -m venv "$SELF_HEALING_DIR/venv"
fi

# Activate and install dependencies
source "$SELF_HEALING_DIR/venv/bin/activate"
pip install -q -r "$SELF_HEALING_DIR/requirements.txt"

# Create systemd service for automated recovery
echo -e "${YELLOW}Creating systemd service...${NC}"
sudo tee /etc/systemd/system/sutazai-self-healing.service > /dev/null << EOF
[Unit]
Description=SutazAI Self-Healing Service
After=docker.service
Requires=docker.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$SELF_HEALING_DIR
Environment="PATH=$SELF_HEALING_DIR/venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=$SELF_HEALING_DIR/venv/bin/python $SELF_HEALING_DIR/scripts/automated-recovery.py
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF

# Create Docker container for self-healing monitor
echo -e "${YELLOW}Creating self-healing monitor container...${NC}"
cat > "$SELF_HEALING_DIR/Dockerfile" << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY monitoring/ ./monitoring/

# Make scripts executable
RUN chmod +x scripts/*.py scripts/*.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8200/health')"

# Run the monitoring service
CMD ["python", "scripts/self-healing-api.py"]
EOF

# Create API service for self-healing
echo -e "${YELLOW}Creating self-healing API service...${NC}"
cat > "$SELF_HEALING_DIR/scripts/self-healing-api.py" << 'EOF'
#!/usr/bin/env python3
"""
Self-Healing API Service
Provides REST API for self-healing status and control
"""

from flask import Flask, jsonify, request
import asyncio
import threading
from automated_recovery import AutomatedRecoveryManager
from circuit_breaker import CircuitBreakerManager
from graceful_degradation import GracefulDegradationManager

app = Flask(__name__)

# Global managers
recovery_manager = None
circuit_manager = None
degradation_manager = None
monitor_thread = None

def run_async_monitor():
    """Run the async monitor in a separate thread"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(recovery_manager.monitor_and_recover())

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "self-healing"}), 200

@app.route('/status', methods=['GET'])
def get_status():
    """Get overall self-healing status"""
    return jsonify({
        "health_checks": recovery_manager.get_health_status() if recovery_manager else {},
        "recovery_actions": recovery_manager.get_recovery_status() if recovery_manager else {},
        "circuit_breakers": circuit_manager.get_all_status() if circuit_manager else {},
        "feature_flags": degradation_manager.get_all_features() if degradation_manager else {}
    })

@app.route('/circuit-breakers', methods=['GET'])
def get_circuit_breakers():
    """Get circuit breaker status"""
    if circuit_manager:
        return jsonify(circuit_manager.get_all_status())
    return jsonify({"error": "Circuit breaker manager not initialized"}), 500

@app.route('/circuit-breakers/<service>/reset', methods=['POST'])
def reset_circuit_breaker(service):
    """Reset a circuit breaker"""
    if circuit_manager:
        circuit_manager.reset_circuit(service)
        return jsonify({"message": f"Circuit breaker for {service} reset"})
    return jsonify({"error": "Circuit breaker manager not initialized"}), 500

@app.route('/feature-flags', methods=['GET'])
def get_feature_flags():
    """Get feature flag status"""
    if degradation_manager:
        return jsonify(degradation_manager.get_all_features())
    return jsonify({"error": "Degradation manager not initialized"}), 500

@app.route('/feature-flags/<feature>', methods=['POST'])
def toggle_feature_flag(feature):
    """Toggle a feature flag"""
    if degradation_manager:
        enabled = request.json.get('enabled', True)
        degradation_manager.toggle_feature(feature, enabled)
        return jsonify({"message": f"Feature {feature} set to {enabled}"})
    return jsonify({"error": "Degradation manager not initialized"}), 500

if __name__ == '__main__':
    # Initialize managers
    recovery_manager = AutomatedRecoveryManager()
    circuit_manager = CircuitBreakerManager()
    degradation_manager = GracefulDegradationManager()
    
    # Start monitoring in background thread
    monitor_thread = threading.Thread(target=run_async_monitor)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Start API server
    app.run(host='0.0.0.0', port=8200)
EOF

# Build Docker image
echo -e "${YELLOW}Building self-healing Docker image...${NC}"
cd "$SELF_HEALING_DIR"
docker build -t sutazai-self-healing:latest .

# Update docker-compose.yml to include self-healing service
echo -e "${YELLOW}Adding self-healing service to docker-compose...${NC}"
if ! grep -q "self-healing:" "$PROJECT_ROOT/docker-compose.yml"; then
    cat >> "$PROJECT_ROOT/docker-compose.yml" << 'EOF'

  self-healing:
    image: sutazai-self-healing:latest
    container_name: sutazai-self-healing
    restart: unless-stopped
    environment:
      - POSTGRES_HOST=postgres
      - POSTGRES_PORT=5432
      - POSTGRES_USER=${POSTGRES_USER:-sutazai}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - POSTGRES_DB=${POSTGRES_DB:-sutazai}
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - REDIS_PASSWORD=${REDIS_PASSWORD}
      - TZ=${TZ:-UTC}
      - SUTAZAI_ENV=${SUTAZAI_ENV:-production}
    volumes:
      - ./self-healing/config:/app/config:ro
      - ./self-healing/logs:/app/logs
      - ./self-healing/data:/app/data
      - /var/run/docker.sock:/var/run/docker.sock:ro
    ports:
      - "8200:8200"
    depends_on:
      - redis
      - postgres
    networks:
      - sutazai-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8200/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s
EOF
fi

# Start the self-healing container
echo -e "${YELLOW}Starting self-healing container...${NC}"
docker-compose up -d self-healing

# Wait for service to be healthy
echo -e "${YELLOW}Waiting for self-healing service to be healthy...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:8200/health >/dev/null 2>&1; then
        echo -e "${GREEN}Self-healing service is healthy!${NC}"
        break
    fi
    echo -n "."
    sleep 2
done

# Enable and start systemd service (optional)
echo -e "${YELLOW}Enabling systemd service (optional)...${NC}"
sudo systemctl daemon-reload
# sudo systemctl enable sutazai-self-healing.service
# sudo systemctl start sutazai-self-healing.service

# Create monitoring dashboard
echo -e "${YELLOW}Creating Prometheus metrics for self-healing...${NC}"
cat > "$SELF_HEALING_DIR/monitoring/self-healing-metrics.yml" << 'EOF'
# Prometheus metrics for self-healing
groups:
  - name: self_healing
    interval: 30s
    rules:
      - record: self_healing:circuit_breaker:open_total
        expr: sum(circuit_breaker_open_total) by (service)
      
      - record: self_healing:recovery:attempts_total
        expr: sum(recovery_action_attempts_total) by (action)
      
      - record: self_healing:health_check:failures_total
        expr: sum(health_check_failures_total) by (check)
      
      - record: self_healing:feature_flag:disabled_total
        expr: count(feature_flag_enabled == 0)
EOF

# Display status
echo -e "${GREEN}=== Self-Healing Deployment Complete ===${NC}"
echo -e "Self-healing API: http://localhost:8200"
echo -e "Status endpoint: http://localhost:8200/status"
echo -e "Circuit breakers: http://localhost:8200/circuit-breakers"
echo -e "Feature flags: http://localhost:8200/feature-flags"
echo -e ""
echo -e "To view logs: docker logs -f sutazai-self-healing"
echo -e "To check status: curl http://localhost:8200/status | jq"

# Show initial status
echo -e "\n${YELLOW}Current self-healing status:${NC}"
curl -s http://localhost:8200/status | jq '.' 2>/dev/null || echo "Service starting up..."