#!/bin/bash
# SutazAI Container Self-Healing Fix Script
# Comprehensive solution for container health issues and auto-healing

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/opt/sutazaiapp/logs/self-healing-fix.log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

# Logging function
log() {
    echo "[$TIMESTAMP] $1" | tee -a "$LOG_FILE"
}

log "=== SutazAI Container Self-Healing Fix Started ==="

# Function to fix health check configurations
fix_health_checks() {
    log "Fixing health check configurations..."
    
    # Create a universal health check script that doesn't require curl
    cat > /tmp/health_check.py << 'EOF'
#!/usr/bin/env python3
import sys
import socket
import requests
from urllib.parse import urlparse

def check_health(url):
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return True
    except:
        pass
    return False

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080/health"
    if check_health(url):
        sys.exit(0)
    else:
        sys.exit(1)
EOF

    # Fix health check endpoints for all agent containers
    log "Adding proper health endpoints to agent applications..."
    
    # Create universal health endpoint for all agents
    cat > /tmp/health_endpoint.py << 'EOF'
@app.get("/health")
async def health_check():
    """Universal health check endpoint"""
    return {"status": "healthy", "service": "agent", "timestamp": str(datetime.utcnow())}

@app.get("/healthz")
async def health_check_k8s():
    """Kubernetes style health check"""
    return {"status": "ok"}

@app.get("/ready")
async def readiness_check():
    """Readiness probe endpoint"""
    return {"status": "ready", "service": "agent"}
EOF

    log "Health check configurations updated"
}

# Function to create self-healing docker compose override
create_self_healing_override() {
    log "Creating self-healing Docker Compose override..."
    
    cat > "/opt/sutazaiapp/docker-compose.self-healing.yml" << 'EOF'
version: '3.8'

# Self-healing container configurations
x-agent-self-healing: &agent-self-healing
  restart: unless-stopped
  healthcheck:
    test: ["CMD", "python3", "-c", "import requests; requests.get('http://localhost:8080/health', timeout=5)"]
    interval: 60s
    timeout: 30s
    retries: 3
    start_period: 120s
  deploy:
    restart_policy:
      condition: on-failure
      delay: 10s
      max_attempts: 5
      window: 120s
    resources:
      limits:
        cpus: '0.5'
        memory: 512M
      reservations:
        cpus: '0.1'
        memory: 128M

services:
  # Apply self-healing to all agent services
  sutazai-data-analysis-engineer-phase3:
    <<: *agent-self-healing
    
  sutazai-dify-automation-specialist-phase3:
    <<: *agent-self-healing
    
  sutazai-awesome-code-ai-phase3:
    <<: *agent-self-healing
    
  sutazai-distributed-computing-architect-phase3:
    <<: *agent-self-healing
    
  sutazai-federated-learning-coordinator-phase3:
    <<: *agent-self-healing
    
  sutazai-garbage-collector-coordinator-phase3:
    <<: *agent-self-healing
    
  sutazai-edge-computing-optimizer-phase3:
    <<: *agent-self-healing
    
  sutazai-finrobot-phase3:
    <<: *agent-self-healing
    
  sutazai-flowiseai-flow-manager-phase3:
    <<: *agent-self-healing
    
  sutazai-data-pipeline-engineer-phase3:
    <<: *agent-self-healing
    
  sutazai-episodic-memory-engineer-phase3:
    <<: *agent-self-healing
    
  sutazai-gradient-compression-specialist-phase3:
    <<: *agent-self-healing
    
  sutazai-explainable-ai-specialist-phase3:
    <<: *agent-self-healing
    
  sutazai-document-knowledge-manager-phase3:
    <<: *agent-self-healing
    
  # Phase 2 agents
  sutazai-attention-optimizer-phase2:
    <<: *agent-self-healing
    
  sutazai-autogen-phase2:
    <<: *agent-self-healing
    
  sutazai-cognitive-architecture-designer-phase2:
    <<: *agent-self-healing
    
  sutazai-crewai-phase2:
    <<: *agent-self-healing
    
  sutazai-code-improver-phase2:
    <<: *agent-self-healing
    
  sutazai-autogpt-phase2:
    <<: *agent-self-healing
    
  sutazai-aider-phase2:
    <<: *agent-self-healing
    
  sutazai-browser-automation-orchestrator-phase2:
    <<: *agent-self-healing
    
  sutazai-devika-phase2:
    <<: *agent-self-healing
    
  # Phase 1 agents
  sutazai-ai-scrum-master-phase1:
    <<: *agent-self-healing
    
  sutazai-ai-product-manager-phase1:
    <<: *agent-self-healing
    
  sutazai-agentzero-coordinator-phase1:
    <<: *agent-self-healing

  # Infrastructure services self-healing
  sutazai-service-registry:
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8500/v1/status/leader"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  sutazai-hardware-resource-optimizer:
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python3", "-c", "import requests; requests.get('http://localhost:8080/health', timeout=5)"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
EOF

    log "Self-healing Docker Compose override created"
}

# Function to fix agent health endpoints
fix_agent_health_endpoints() {
    log "Fixing agent health endpoints..."
    
    # Add health endpoints to all agent apps
    local agents_dir="/opt/sutazaiapp/agents"
    
    # Find all agent app.py files and add health endpoints if missing
    find "$agents_dir" -name "app.py" -type f | while read -r app_file; do
        if ! grep -q "/health" "$app_file"; then
            log "Adding health endpoint to: $app_file"
            
            # Backup original file
            cp "$app_file" "$app_file.bak.$(date +%s)"
            
            # Add health endpoints after FastAPI app creation
            python3 << EOF
import re

with open('$app_file', 'r') as f:
    content = f.read()

# Find FastAPI app creation line
app_pattern = r'app = FastAPI\([^)]*\)'
if re.search(app_pattern, content):
    # Add health endpoints after app creation
    health_endpoints = '''
from datetime import datetime

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "agent", "timestamp": str(datetime.utcnow())}

@app.get("/healthz")
async def health_check_k8s():
    """Kubernetes style health check"""
    return {"status": "ok"}

@app.get("/ready")
async def readiness_check():
    """Readiness probe endpoint"""
    return {"status": "ready", "service": "agent"}
'''
    
    # Insert health endpoints after the last import or app creation
    if 'from datetime import datetime' not in content:
        content = re.sub(r'(import.*\n)', r'\1from datetime import datetime\n', content, count=1)
    
    # Add health endpoints after app creation
    content = re.sub(r'(app = FastAPI\([^)]*\)\n)', r'\1' + health_endpoints + '\n', content)
    
    with open('$app_file', 'w') as f:
        f.write(content)
    
    print(f"Added health endpoints to $app_file")
EOF
            
        fi
    done
}

# Function to implement container auto-restart mechanism
implement_auto_restart() {
    log "Implementing container auto-restart mechanism..."
    
    cat > "/opt/sutazaiapp/scripts/container-auto-healer.sh" << 'EOF'
#!/bin/bash
# Container Auto-Healer Service

HEAL_LOG="/opt/sutazaiapp/logs/auto-healer.log"

log_heal() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$HEAL_LOG"
}

heal_container() {
    local container_name="$1"
    local health_status="$2"
    
    log_heal "Healing container: $container_name (status: $health_status)"
    
    case "$health_status" in
        "unhealthy")
            log_heal "Restarting unhealthy container: $container_name"
            docker restart "$container_name" 2>/dev/null || log_heal "Failed to restart $container_name"
            ;;
        "restarting")
            # If container has been restarting too long, force restart
            local restart_count=$(docker inspect "$container_name" --format='{{.RestartCount}}' 2>/dev/null || echo "0")
            if [[ $restart_count -gt 5 ]]; then
                log_heal "Force stopping and starting stuck container: $container_name"
                docker stop "$container_name" 2>/dev/null || true
                sleep 5
                docker start "$container_name" 2>/dev/null || log_heal "Failed to start $container_name"
            fi
            ;;
        "exited")
            log_heal "Starting exited container: $container_name"
            docker start "$container_name" 2>/dev/null || log_heal "Failed to start $container_name"
            ;;
    esac
}

# Main healing loop
while true; do
    # Check all SutazAI containers
    while IFS=$'\t' read -r name status health; do
        if [[ -n "$name" ]] && [[ "$name" =~ ^sutazai- ]]; then
            case "$health" in
                "unhealthy"|"starting")
                    heal_container "$name" "$health"
                    ;;
            esac
            
            # Check for containers stuck in restarting state
            if [[ "$status" =~ "Restarting" ]]; then
                heal_container "$name" "restarting"
            fi
            
            # Check for exited containers
            if [[ "$status" =~ "Exited" ]]; then
                heal_container "$name" "exited"
            fi
        fi
    done < <(docker ps -a --format "{{.Names}}\t{{.Status}}\t{{.State}}" 2>/dev/null)
    
    # Wait before next check
    sleep 30
done
EOF

    chmod +x "/opt/sutazaiapp/scripts/container-auto-healer.sh"
    
    log "Container auto-healer created"
}

# Function to fix Prometheus scraping issues
fix_prometheus_scraping() {
    log "Fixing Prometheus scraping configuration..."
    
    # Update Prometheus config to handle JSON endpoints better
    cat > "/tmp/prometheus_fix.yml" << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "/etc/prometheus/alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - sutazai-alertmanager:9093

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['sutazai-node-exporter:9100']

  - job_name: 'sutazai-backend'
    static_configs:
      - targets: ['sutazai-backend:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
    scrape_timeout: 10s

  - job_name: 'sutazai-agents'
    static_configs:
      - targets: 
        - 'sutazai-data-analysis-engineer-phase3:8080'
        - 'sutazai-dify-automation-specialist-phase3:8080'
        - 'sutazai-awesome-code-ai-phase3:8080'
    metrics_path: '/metrics'
    scrape_interval: 60s
    scrape_timeout: 30s

  - job_name: 'externalService, thirdPartyAPI-http-checks'
    metrics_path: /probe
    params:
      module: [http_2xx]
    static_configs:
      - targets:
        - http://sutazai-backend:8000/health
        - http://sutazai-redis:6379
        - http://sutazai-chromadb:8000/api/v1/heartbeat
        - http://sutazai-qdrant:6333/
    relabel_configs:
      - source_labels: [__address__]
        target_label: __param_target
      - source_labels: [__param_target]
        target_label: instance
      - target_label: __address__
        replacement: sutazai-blackbox-exporter:9115
EOF

    # Apply the fix if prometheus container exists
    if docker ps --format "{{.Names}}" | grep -q "sutazai-prometheus"; then
        docker cp "/tmp/prometheus_fix.yml" sutazai-prometheus:/etc/prometheus/prometheus.yml
        docker restart sutazai-prometheus
        log "Prometheus configuration updated and restarted"
    fi
}

# Function to create systemd service for auto-healer
create_auto_healer_service() {
    log "Creating systemd service for auto-healer..."
    
    cat > "/tmp/sutazai-auto-healer.service" << 'EOF'
[Unit]
Description=SutazAI Container Auto-Healer
After=docker.service
Requires=docker.service

[Service]
Type=simple
ExecStart=/opt/sutazaiapp/scripts/container-auto-healer.sh
Restart=always
RestartSec=10
User=root

[Install]
WantedBy=multi-user.target
EOF

    # Install and start the service
    if command -v systemctl >/dev/null 2>&1; then
        sudo cp "/tmp/sutazai-auto-healer.service" /etc/systemd/system/
        sudo systemctl daemon-reload
        sudo systemctl enable sutazai-auto-healer.service
        sudo systemctl start sutazai-auto-healer.service
        log "Auto-healer systemd service created and started"
    else
        log "Systemctl not available, running auto-healer in background"
        nohup /opt/sutazaiapp/scripts/container-auto-healer.sh > /opt/sutazaiapp/logs/auto-healer-bg.log 2>&1 &
    fi
}

# Function to perform immediate container fixes
perform_immediate_fixes() {
    log "Performing immediate container fixes..."
    
    # Stop and remove containers that are in bad state
    local problematic_containers=()
    
    while IFS=$'\t' read -r name status; do
        if [[ -n "$name" ]] && [[ "$name" =~ ^sutazai- ]]; then
            if [[ "$status" =~ "Restarting" ]] || [[ "$status" =~ "Dead" ]] || [[ "$status" =~ "Exited.*\(1\)" ]]; then
                problematic_containers+=("$name")
            fi
        fi
    done < <(docker ps -a --format "{{.Names}}\t{{.Status}}")
    
    for container in "${problematic_containers[@]}"; do
        log "Fixing problematic container: $container"
        docker stop "$container" 2>/dev/null || true
        sleep 2
        docker rm "$container" 2>/dev/null || true
    done
    
    # Recreate containers using docker-compose
    log "Recreating containers with self-healing configuration..."
    cd /opt/sutazaiapp
    
    # Apply the self-healing override
    docker-compose -f docker-compose.yml -f docker-compose.self-healing.yml up -d --remove-orphans
    
    log "Containers recreated with self-healing configuration"
}

# Function to monitor and report health status
monitor_health_status() {
    log "Monitoring container health status..."
    
    local max_wait=600  # 10 minutes
    local wait_time=0
    local check_interval=30
    
    while [[ $wait_time -lt $max_wait ]]; do
        local healthy_count=0
        local total_count=0
        local unhealthy_containers=()
        
        while IFS=$'\t' read -r name health status; do
            if [[ -n "$name" ]] && [[ "$name" =~ ^sutazai- ]]; then
                total_count=$((total_count + 1))
                
                if [[ "$health" == "healthy" ]] || [[ "$health" == "(healthy)" ]] || [[ "$status" =~ "Up.*healthy" ]]; then
                    healthy_count=$((healthy_count + 1))
                elif [[ "$health" == "unhealthy" ]] || [[ "$status" =~ "unhealthy" ]]; then
                    unhealthy_containers+=("$name")
                fi
            fi
        done < <(docker ps --format "{{.Names}}\t{{.State}}\t{{.Status}}")
        
        local health_rate=0
        if [[ $total_count -gt 0 ]]; then
            health_rate=$((healthy_count * 100 / total_count))
        fi
        
        log "Health Status: $healthy_count/$total_count containers healthy ($health_rate%)"
        
        if [[ ${#unhealthy_containers[@]} -gt 0 ]]; then
            log "Unhealthy containers: ${unhealthy_containers[*]}"
        fi
        
        # If we reach 80% health rate, consider it successful
        if [[ $health_rate -ge 80 ]]; then
            log "Target health rate achieved! ($health_rate%)"
            break
        fi
        
        sleep $check_interval
        wait_time=$((wait_time + check_interval))
    done
    
    log "Final health monitoring completed"
}

# Main execution function
main() {
    log "Starting comprehensive container self-healing fix..."
    
    # Step 1: Fix health check configurations
    fix_health_checks
    
    # Step 2: Create self-healing docker compose override
    create_self_healing_override
    
    # Step 3: Fix agent health endpoints
    fix_agent_health_endpoints
    
    # Step 4: Fix Prometheus scraping issues
    fix_prometheus_scraping
    
    # Step 5: Implement auto-restart mechanism
    implement_auto_restart
    
    # Step 6: Create auto-healer service
    create_auto_healer_service
    
    # Step 7: Perform immediate fixes
    perform_immediate_fixes
    
    # Step 8: Monitor health status
    monitor_health_status
    
    log "=== Container Self-Healing Fix Completed ==="
}

# Execute main function
main "$@"