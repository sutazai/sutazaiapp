#!/bin/bash
# Start complete monitoring stack for SutazAI Ollama agents
# This script starts all monitoring components needed to track 131 agents

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
    warning "Running as root. Consider using a non-root user for security."
fi

# Check Docker
if ! command -v docker &> /dev/null; then
    error "Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check Python dependencies
log "Checking Python dependencies..."
if ! python3 -c "import prometheus_client, aiohttp, psutil, sqlite3" 2>/dev/null; then
    log "Installing Python dependencies..."
    pip3 install prometheus_client aiohttp aiohttp-cors psutil
fi

# Create necessary directories
log "Creating monitoring directories..."
mkdir -p "$PROJECT_ROOT/monitoring"/{data,logs,config,grafana/dashboards,prometheus/rules,alertmanager}
mkdir -p "$PROJECT_ROOT/logs"

# Set permissions
chmod -R 755 "$PROJECT_ROOT/monitoring"

# Create Prometheus configuration
log "Creating Prometheus configuration..."
cat > "$PROJECT_ROOT/monitoring/prometheus.yml" << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'ollama-agent-monitor'
    static_configs:
      - targets: ['ollama-agent-monitor:8091']
    scrape_interval: 5s
    metrics_path: /metrics

  - job_name: 'ollama'
    static_configs:
      - targets: ['ollama:11434']
    scrape_interval: 10s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']

  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis-exporter'
    static_configs:
      - targets: ['redis-exporter:9121']

  # Dynamic agent discovery (if agents expose metrics)
  - job_name: 'sutazai-agents'
    static_configs:
      - targets: []
    scrape_interval: 30s
    honor_labels: true
EOF

# Create AlertManager configuration
log "Creating AlertManager configuration..."
cat > "$PROJECT_ROOT/monitoring/alertmanager/alertmanager.yml" << 'EOF'
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@sutazai.local'

route:
  group_by: ['alertname', 'severity', 'component']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'
  routes:
  - match:
      severity: critical
    receiver: 'critical-alerts'
    group_wait: 5s
    repeat_interval: 30m
  - match:
      component: freeze_prevention
    receiver: 'freeze-prevention'
    group_wait: 1s
    repeat_interval: 5m

receivers:
- name: 'web.hook'
  webhook_configs:
  - url: 'http://localhost:5001/webhook'
    send_resolved: true

- name: 'critical-alerts'
  webhook_configs:
  - url: 'http://localhost:5001/webhook/critical'
    send_resolved: true
  # Add email, Slack, etc. configurations here

- name: 'freeze-prevention'
  webhook_configs:
  - url: 'http://localhost:5001/webhook/freeze-prevention'
    send_resolved: true

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'component', 'agent_name']
EOF

# Create Grafana provisioning
log "Creating Grafana provisioning configuration..."
mkdir -p "$PROJECT_ROOT/monitoring/grafana/provisioning"/{dashboards,datasources}

cat > "$PROJECT_ROOT/monitoring/grafana/provisioning/datasources/prometheus.yml" << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF

cat > "$PROJECT_ROOT/monitoring/grafana/provisioning/dashboards/dashboards.yml" << 'EOF'
apiVersion: 1

providers:
  - name: 'SutazAI Dashboards'
    orgId: 1
    folder: 'SutazAI'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
EOF

# Create monitoring Docker Compose file
log "Creating monitoring Docker Compose configuration..."
cat > "$PROJECT_ROOT/docker-compose.monitoring-stack.yml" << 'EOF'
version: '3.8'

services:
  # Ollama Agent Monitor - Core monitoring service
  ollama-agent-monitor:
    build:
      context: .
      dockerfile: Dockerfile.monitoring
    container_name: sutazai-ollama-monitor
    restart: unless-stopped
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
      - BACKEND_URL=http://backend:8000
      - METRICS_PORT=8091
      - LOG_LEVEL=INFO
    volumes:
      - ./monitoring:/app/monitoring
      - ./logs:/app/logs
    ports:
      - "8091:8091"
    networks:
      - sutazai-network
    depends_on:
      - prometheus
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8091/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Real-time Dashboard
  realtime-dashboard:
    build:
      context: .
      dockerfile: Dockerfile.dashboard
    container_name: sutazai-dashboard
    restart: unless-stopped
    environment:
      - MONITOR_URL=http://ollama-agent-monitor:8091
    ports:
      - "8092:8092"
    volumes:
      - ./monitoring:/app/monitoring
    networks:
      - sutazai-network
    depends_on:
      - ollama-agent-monitor
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8092/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Prometheus
  prometheus:
    image: prom/prometheus:latest
    container_name: sutazai-prometheus
    restart: unless-stopped
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=30d'
      - '--web.enable-lifecycle'
      - '--storage.tsdb.retention.size=10GB'
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./monitoring/prometheus/rules:/etc/prometheus/rules
      - prometheus_data:/prometheus
    networks:
      - sutazai-network
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost:9090/-/healthy"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Grafana
  grafana:
    image: grafana/grafana:latest
    container_name: sutazai-grafana
    restart: unless-stopped
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=sutazai123
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_INSTALL_PLUGINS=grafana-clock-panel,grafana-simple-json-datasource
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
    networks:
      - sutazai-network
    depends_on:
      - prometheus
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # AlertManager
  alertmanager:
    image: prom/alertmanager:latest
    container_name: sutazai-alertmanager
    restart: unless-stopped
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'
      - '--storage.path=/alertmanager'
      - '--web.external-url=http://localhost:9093'
    ports:
      - "9093:9093"
    volumes:
      - ./monitoring/alertmanager:/etc/alertmanager
      - alertmanager_data:/alertmanager
    networks:
      - sutazai-network
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost:9093/-/healthy"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  prometheus_data:
  grafana_data:
  alertmanager_data:

networks:
  sutazai-network:
    external: true
EOF

# Create Dockerfiles for monitoring services
log "Creating monitoring Dockerfiles..."

cat > "$PROJECT_ROOT/Dockerfile.monitoring" << 'EOF'
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional monitoring dependencies
RUN pip install prometheus_client aiohttp aiohttp-cors psutil

# Copy application code
COPY monitoring/ ./monitoring/
COPY agents/core/ ./agents/core/

# Create non-root user
RUN useradd -m -u 1000 monitor && chown -R monitor:monitor /app
USER monitor

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8091/health || exit 1

CMD ["python", "-m", "monitoring.ollama_agent_monitor"]
EOF

cat > "$PROJECT_ROOT/Dockerfile.dashboard" << 'EOF'
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install aiohttp aiohttp-cors

# Copy dashboard code
COPY monitoring/realtime_dashboard.py ./
COPY monitoring/ollama_agent_monitor.py ./monitoring/

# Create non-root user
RUN useradd -m -u 1000 dashboard && chown -R dashboard:dashboard /app
USER dashboard

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8092/health || exit 1

CMD ["python", "realtime_dashboard.py"]
EOF

# Create monitoring requirements file
cat > "$PROJECT_ROOT/requirements.monitoring.txt" << 'EOF'
prometheus_client>=0.15.0
aiohttp>=3.8.0
aiohttp-cors>=0.7.0
psutil>=5.9.0
httpx>=0.24.0
EOF

# Function to check if service is healthy
check_service_health() {
    local service_name=$1
    local health_url=$2
    local max_attempts=30
    local attempt=1
    
    log "Checking health of $service_name..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$health_url" > /dev/null 2>&1; then
            success "$service_name is healthy"
            return 0
        fi
        
        if [ $attempt -eq $max_attempts ]; then
            error "$service_name failed to become healthy after $max_attempts attempts"
            return 1
        fi
        
        echo -n "."
        sleep 5
        ((attempt++))
    done
}

# Main startup function
start_monitoring_stack() {
    log "Starting SutazAI Monitoring Stack..."
    
    # Check if network exists
    if ! docker network ls | grep -q sutazai-network; then
        log "Creating sutazai-network..."
        docker network create sutazai-network
    fi
    
    # Stop any existing monitoring services
    log "Stopping existing monitoring services..."
    docker-compose -f "$PROJECT_ROOT/docker-compose.monitoring-stack.yml" down --remove-orphans || true
    
    # Build and start services
    log "Building and starting monitoring services..."
    docker-compose -f "$PROJECT_ROOT/docker-compose.monitoring-stack.yml" up -d --build
    
    # Wait for services to be healthy
    log "Waiting for services to become healthy..."
    sleep 10
    
    check_service_health "Prometheus" "http://localhost:9090/-/healthy"
    check_service_health "Grafana" "http://localhost:3000/api/health"
    check_service_health "AlertManager" "http://localhost:9093/-/healthy"
    check_service_health "Ollama Agent Monitor" "http://localhost:8091/health"
    check_service_health "Real-time Dashboard" "http://localhost:8092/health"
    
    success "Monitoring stack started successfully!"
    
    log "Services available at:"
    echo "  📊 Grafana Dashboard:     http://localhost:3000 (admin/sutazai123)"
    echo "  🔍 Prometheus:           http://localhost:9090"
    echo "  🚨 AlertManager:         http://localhost:9093"
    echo "  📈 Real-time Dashboard:  http://localhost:8092"
    echo "  📊 Agent Monitor API:    http://localhost:8091"
    echo ""
    log "To view logs: docker-compose -f docker-compose.monitoring-stack.yml logs -f"
    log "To stop: docker-compose -f docker-compose.monitoring-stack.yml down"
}

# Function to show monitoring status
show_status() {
    log "Monitoring Stack Status:"
    docker-compose -f "$PROJECT_ROOT/docker-compose.monitoring-stack.yml" ps
    
    echo ""
    log "Service Health Checks:"
    
    services=(
        "Prometheus:http://localhost:9090/-/healthy"
        "Grafana:http://localhost:3000/api/health"
        "AlertManager:http://localhost:9093/-/healthy"
        "Agent Monitor:http://localhost:8091/health"
        "Dashboard:http://localhost:8092/health"
    )
    
    for service_info in "${services[@]}"; do
        IFS=':' read -r name url <<< "$service_info"
        if curl -s "$url" > /dev/null 2>&1; then
            echo "  ✅ $name: Healthy"
        else
            echo "  ❌ $name: Unhealthy"
        fi
    done
}

# Function to stop monitoring stack
stop_monitoring_stack() {
    log "Stopping monitoring stack..."
    docker-compose -f "$PROJECT_ROOT/docker-compose.monitoring-stack.yml" down
    success "Monitoring stack stopped"
}

# Function to show logs
show_logs() {
    docker-compose -f "$PROJECT_ROOT/docker-compose.monitoring-stack.yml" logs -f
}

# Parse command line arguments
case "${1:-start}" in
    start)
        start_monitoring_stack
        ;;
    stop)
        stop_monitoring_stack
        ;;
    restart)
        stop_monitoring_stack
        sleep 5
        start_monitoring_stack
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs}"
        echo ""
        echo "  start   - Start the monitoring stack (default)"
        echo "  stop    - Stop the monitoring stack"
        echo "  restart - Restart the monitoring stack"
        echo "  status  - Show status of monitoring services"
        echo "  logs    - Show logs from all monitoring services"
        exit 1
        ;;
esac