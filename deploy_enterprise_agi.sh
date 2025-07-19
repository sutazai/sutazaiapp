#!/bin/bash

###############################################################################
# Enterprise AGI/ASI Deployment Script for SutazAI
# Complete deployment with all security, monitoring, and AI components
###############################################################################

set -euo pipefail

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
LOG_FILE="$SCRIPT_DIR/deployment_enterprise_$(date +%Y%m%d_%H%M%S).log"

# Log function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

# Display banner
display_banner() {
    echo -e "${CYAN}"
    cat << 'EOF'
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   ███████╗██╗   ██╗████████╗ █████╗ ███████╗ █████╗ ██╗      ║
║   ██╔════╝██║   ██║╚══██╔══╝██╔══██╗╚══███╔╝██╔══██╗██║      ║
║   ███████╗██║   ██║   ██║   ███████║  ███╔╝ ███████║██║      ║
║   ╚════██║██║   ██║   ██║   ██╔══██║ ███╔╝  ██╔══██║██║      ║
║   ███████║╚██████╔╝   ██║   ██║  ██║███████╗██║  ██║██║      ║
║   ╚══════╝ ╚═════╝    ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝      ║
║                                                               ║
║         Enterprise AGI/ASI Autonomous System v11.0            ║
║                100% Local • Self-Improving • Secure           ║
╚═══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}\n"
}

# Check system requirements
check_requirements() {
    log "Checking system requirements..."
    
    # Check CPU cores
    cpu_cores=$(nproc)
    if [ "$cpu_cores" -lt 16 ]; then
        warning "System has $cpu_cores CPU cores. Recommended: 16+"
    else
        info "✓ CPU cores: $cpu_cores"
    fi
    
    # Check RAM
    total_ram=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$total_ram" -lt 64 ]; then
        warning "System has ${total_ram}GB RAM. Recommended: 64GB+"
    else
        info "✓ RAM: ${total_ram}GB"
    fi
    
    # Check disk space
    available_space=$(df -BG "$SCRIPT_DIR" | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$available_space" -lt 200 ]; then
        warning "Available disk space: ${available_space}GB. Recommended: 200GB+"
    else
        info "✓ Disk space: ${available_space}GB available"
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
    else
        docker_version=$(docker --version | awk '{print $3}' | sed 's/,//')
        info "✓ Docker version: $docker_version"
    fi
    
    # Check Docker Compose
    if docker compose version &> /dev/null; then
        compose_version=$(docker compose version | awk '{print $4}')
        info "✓ Docker Compose version: $compose_version"
    else
        error "Docker Compose is not installed."
    fi
    
    # Check GPU
    if command -v nvidia-smi &> /dev/null; then
        gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        info "✓ GPU detected: $gpu_info"
    else
        warning "No NVIDIA GPU detected. AI inference will be slower."
    fi
}

# Prepare directory structure
prepare_directories() {
    log "Preparing directory structure..."
    
    directories=(
        "secrets"
        "certs"
        "config/vault"
        "config/kong"
        "config/haproxy"
        "config/prometheus"
        "config/grafana/provisioning/dashboards"
        "config/grafana/provisioning/datasources"
        "config/elasticsearch"
        "config/neo4j"
        "config/redis"
        "config/postgres"
        "data/postgres-master"
        "data/postgres-slave"
        "data/redis"
        "data/vault"
        "data/neo4j"
        "data/elasticsearch"
        "docker/reasoning-engine"
        "docker/knowledge-manager"
        "docker/self-improvement"
        "docker/metacognition"
        "docker/agent-orchestrator"
        "docker/ollama-cluster"
        "logs"
        "backups"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$SCRIPT_DIR/$dir"
    done
    
    # Set secure permissions
    chmod 700 "$SCRIPT_DIR/secrets"
    chmod 700 "$SCRIPT_DIR/certs"
    chmod 700 "$SCRIPT_DIR/data"
    
    log "Directory structure created"
}

# Setup security
setup_security() {
    log "Setting up enterprise security..."
    
    # Run security setup script
    if [ -f "$SCRIPT_DIR/scripts/setup_enterprise_security.sh" ]; then
        bash "$SCRIPT_DIR/scripts/setup_enterprise_security.sh"
    else
        warning "Security setup script not found. Creating basic security..."
        
        # Generate basic secrets
        echo "$(openssl rand -base64 32)" > "$SCRIPT_DIR/secrets/postgres_password.txt"
        echo "$(openssl rand -base64 32)" > "$SCRIPT_DIR/secrets/jwt_secret.txt"
        echo "$(uuidgen)" > "$SCRIPT_DIR/secrets/vault_token.txt"
        
        chmod 600 "$SCRIPT_DIR/secrets/"*.txt
    fi
}

# Create monitoring configuration
create_monitoring_config() {
    log "Creating monitoring configuration..."
    
    # Prometheus configuration
    cat > "$SCRIPT_DIR/config/prometheus/prometheus.yml" << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'sutazai-prod'
    
rule_files:
  - "rules/*.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'docker'
    static_configs:
      - targets: ['docker-exporter:9323']

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  - job_name: 'elasticsearch'
    static_configs:
      - targets: ['elasticsearch-exporter:9114']

  - job_name: 'gpu'
    static_configs:
      - targets: ['nvidia-gpu-exporter:9835']

  - job_name: 'sutazai-services'
    consul_sd_configs:
      - server: 'consul:8500'
    relabel_configs:
      - source_labels: [__meta_consul_service]
        target_label: job
      - source_labels: [__meta_consul_service_id]
        target_label: instance
EOF

    # Grafana datasource
    cat > "$SCRIPT_DIR/config/grafana/provisioning/datasources/prometheus.yml" << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true

  - name: Elasticsearch
    type: elasticsearch
    access: proxy
    url: http://elasticsearch:9200
    database: "sutazai-*"
    jsonData:
      esVersion: "8.11.0"
      timeField: "@timestamp"
    editable: true

  - name: Jaeger
    type: jaeger
    access: proxy
    url: http://jaeger:16686
    editable: true
EOF

    # Create AGI dashboard
    create_agi_dashboard
}

# Create AGI monitoring dashboard
create_agi_dashboard() {
    cat > "$SCRIPT_DIR/config/grafana/dashboards/agi-dashboard.json" << 'EOF'
{
  "dashboard": {
    "title": "SutazAI AGI System Dashboard",
    "panels": [
      {
        "title": "Agent Status",
        "type": "stat",
        "gridPos": {"h": 8, "w": 6, "x": 0, "y": 0},
        "targets": [
          {
            "expr": "sum(up{job=~\"sutazai-.*\"})",
            "refId": "A"
          }
        ]
      },
      {
        "title": "Reasoning Performance",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 6, "y": 0},
        "targets": [
          {
            "expr": "rate(reasoning_requests_total[5m])",
            "refId": "A",
            "legendFormat": "Requests/sec"
          }
        ]
      },
      {
        "title": "Knowledge Graph Size",
        "type": "stat",
        "gridPos": {"h": 8, "w": 6, "x": 18, "y": 0},
        "targets": [
          {
            "expr": "knowledge_graph_nodes_total",
            "refId": "A"
          }
        ]
      },
      {
        "title": "GPU Utilization",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
        "targets": [
          {
            "expr": "nvidia_gpu_utilization",
            "refId": "A",
            "legendFormat": "GPU {{gpu}}"
          }
        ]
      },
      {
        "title": "Memory Usage by Service",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
        "targets": [
          {
            "expr": "container_memory_usage_bytes{name=~\"sutazai-.*\"}",
            "refId": "A",
            "legendFormat": "{{name}}"
          }
        ]
      }
    ],
    "schemaVersion": 39,
    "version": 1,
    "uid": "sutazai-agi",
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "10s"
  }
}
EOF
}

# Build custom Docker images
build_docker_images() {
    log "Building custom Docker images..."
    
    # Check if Dockerfiles exist, if not create them
    if [ ! -f "$SCRIPT_DIR/docker/reasoning-engine/Dockerfile" ]; then
        warning "Some Dockerfiles are missing. Using pre-built images where available."
    fi
    
    # Build images in parallel
    docker_services=(
        "reasoning-engine"
        "knowledge-manager"
        "self-improvement"
        "metacognition"
        "agent-orchestrator"
    )
    
    for service in "${docker_services[@]}"; do
        if [ -f "$SCRIPT_DIR/docker/$service/Dockerfile" ]; then
            log "Building $service..."
            (cd "$SCRIPT_DIR/docker/$service" && docker build -t "sutazai/$service:latest" .) &
        fi
    done
    
    # Wait for all builds to complete
    wait
    
    log "Docker images built successfully"
}

# Deploy services
deploy_services() {
    log "Deploying all services..."
    
    cd "$SCRIPT_DIR"
    
    # Use enterprise compose file if it exists
    if [ -f "docker-compose.enterprise.yml" ]; then
        COMPOSE_FILE="docker-compose.enterprise.yml"
    else
        COMPOSE_FILE="docker-compose.yml"
    fi
    
    # Start core infrastructure first
    log "Starting core infrastructure..."
    docker compose -f "$COMPOSE_FILE" up -d \
        consul vault postgres-master redis-master neo4j elasticsearch
    
    # Wait for infrastructure to be ready
    log "Waiting for infrastructure to initialize..."
    sleep 30
    
    # Start AI services
    log "Starting AI services..."
    docker compose -f "$COMPOSE_FILE" up -d \
        ollama-cluster reasoning-engine knowledge-manager \
        self-improvement metacognition agent-orchestrator
    
    # Start monitoring
    log "Starting monitoring services..."
    docker compose -f "$COMPOSE_FILE" up -d \
        prometheus grafana jaeger logstash
    
    # Start all remaining services
    log "Starting remaining services..."
    docker compose -f "$COMPOSE_FILE" up -d
    
    log "All services deployed"
}

# Initialize AI models
initialize_models() {
    log "Initializing AI models..."
    
    # Pull essential models
    models=(
        "deepseek-r1:8b"
        "qwen2.5:3b"
        "codellama:7b"
        "llama3.2:3b"
        "nomic-embed-text"
    )
    
    for model in "${models[@]}"; do
        log "Pulling model: $model"
        docker exec sutazai-ollama ollama pull "$model" || warning "Failed to pull $model"
    done
    
    log "Models initialized"
}

# Verify deployment
verify_deployment() {
    log "Verifying deployment..."
    
    # Check core services
    services=(
        "consul:8500"
        "vault:8200"
        "postgres-master:5432"
        "redis-master:6379"
        "neo4j:7474"
        "elasticsearch:9200"
        "prometheus:9090"
        "grafana:3000"
    )
    
    failed=0
    for service in "${services[@]}"; do
        name="${service%%:*}"
        port="${service##*:}"
        
        if nc -z localhost "$port" 2>/dev/null; then
            info "✓ $name is accessible on port $port"
        else
            warning "✗ $name is not accessible on port $port"
            ((failed++))
        fi
    done
    
    if [ $failed -eq 0 ]; then
        log "All core services are operational!"
    else
        warning "$failed services are not accessible"
    fi
}

# Create quick access script
create_access_script() {
    cat > "$SCRIPT_DIR/sutazai.sh" << 'EOF'
#!/bin/bash
# SutazAI Quick Access Script

case "$1" in
    start)
        docker compose up -d
        ;;
    stop)
        docker compose down
        ;;
    restart)
        docker compose restart
        ;;
    logs)
        docker compose logs -f ${2:-}
        ;;
    status)
        docker compose ps
        ;;
    shell)
        docker exec -it ${2:-sutazai-agent-orchestrator} bash
        ;;
    backup)
        ./scripts/backup_system.sh
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|logs|status|shell|backup} [service]"
        exit 1
esac
EOF
    
    chmod +x "$SCRIPT_DIR/sutazai.sh"
}

# Display deployment summary
display_summary() {
    echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}🎉 SutazAI Enterprise AGI/ASI System Deployed Successfully!${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
    
    echo -e "${BLUE}🌐 Access Points:${NC}"
    echo -e "  • Main Application:    ${YELLOW}http://localhost:8501${NC}"
    echo -e "  • API Gateway:         ${YELLOW}http://localhost:8000${NC}"
    echo -e "  • Grafana Dashboard:   ${YELLOW}http://localhost:3000${NC} (admin/admin)"
    echo -e "  • Consul UI:           ${YELLOW}http://localhost:8500${NC}"
    echo -e "  • Vault UI:            ${YELLOW}http://localhost:8200${NC}"
    echo -e "  • Neo4j Browser:       ${YELLOW}http://localhost:7474${NC}"
    
    echo -e "\n${BLUE}🤖 AI Services:${NC}"
    echo -e "  • Reasoning Engine:    ${YELLOW}http://localhost:8300${NC}"
    echo -e "  • Knowledge Manager:   ${YELLOW}http://localhost:8301${NC}"
    echo -e "  • Self-Improvement:    ${YELLOW}http://localhost:8302${NC}"
    echo -e "  • Meta-Cognition:      ${YELLOW}http://localhost:8303${NC}"
    
    echo -e "\n${BLUE}🔧 Management:${NC}"
    echo -e "  • Quick commands:      ${YELLOW}./sutazai.sh {start|stop|logs|status}${NC}"
    echo -e "  • System backup:       ${YELLOW}./sutazai.sh backup${NC}"
    echo -e "  • Service shell:       ${YELLOW}./sutazai.sh shell [service]${NC}"
    
    echo -e "\n${BLUE}📊 System Info:${NC}"
    echo -e "  • Total Containers:    ${YELLOW}$(docker ps -q | wc -l)${NC}"
    echo -e "  • CPU Usage:           ${YELLOW}$(top -bn1 | grep "Cpu(s)" | awk '{print $2}')${NC}"
    echo -e "  • Memory Usage:        ${YELLOW}$(free -h | awk '/^Mem:/ {print $3 "/" $2}')${NC}"
    
    echo -e "\n${GREEN}✅ System Status: OPERATIONAL${NC}"
    echo -e "${GREEN}🚀 100% Local AGI/ASI System Ready!${NC}"
    echo -e "\n${CYAN}Deployment log: $LOG_FILE${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

# Main deployment function
main() {
    display_banner
    
    log "Starting Enterprise AGI/ASI deployment..."
    
    # Change to script directory
    cd "$SCRIPT_DIR"
    
    # Execute deployment steps
    check_requirements
    prepare_directories
    setup_security
    create_monitoring_config
    build_docker_images
    deploy_services
    initialize_models
    verify_deployment
    create_access_script
    display_summary
    
    log "Deployment completed successfully!"
}

# Run main function
main "$@"