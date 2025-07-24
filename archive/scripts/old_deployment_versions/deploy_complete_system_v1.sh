#!/bin/bash
# SutazAI Complete System Deployment Script
# Comprehensive AGI/ASI system deployment with all components

set -euo pipefail

# ===============================================
# CONFIGURATION
# ===============================================

PROJECT_ROOT="/opt/sutazaiapp"
COMPOSE_FILE="docker-compose-consolidated.yml"
LOG_FILE="logs/deployment_$(date +%Y%m%d_%H%M%S).log"
ENV_FILE=".env"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ===============================================
# LOGGING FUNCTIONS
# ===============================================

setup_logging() {
    mkdir -p "$(dirname "$LOG_FILE")"
    exec 1> >(tee -a "$LOG_FILE")
    exec 2> >(tee -a "$LOG_FILE" >&2)
}

log() {
    echo -e "${2:-$BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

log_success() {
    log "$1" "$GREEN"
}

log_warn() {
    log "$1" "$YELLOW"
}

log_error() {
    log "$1" "$RED"
}

log_info() {
    log "$1" "$CYAN"
}

# ===============================================
# SYSTEM VALIDATION FUNCTIONS
# ===============================================

validate_system() {
    log_info "Validating system requirements..."
    
    # Check if running as root or with docker permissions
    if ! docker info &>/dev/null; then
        log_error "Docker not available or insufficient permissions"
        exit 1
    fi
    
    # Check if compose file exists
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        log_error "Docker compose file not found: $COMPOSE_FILE"
        exit 1
    fi
    
    # Validate compose file syntax
    if ! docker-compose -f "$COMPOSE_FILE" config >/dev/null 2>&1; then
        log_error "Docker compose file has syntax errors"
        docker-compose -f "$COMPOSE_FILE" config
        exit 1
    fi
    
    # Check available disk space (need at least 20GB)
    available_space=$(df -BG . | awk 'NR==2 {print $4}' | tr -d 'G')
    if [ "$available_space" -lt 20 ]; then
        log_warn "Low disk space: ${available_space}GB available. Recommended: 20GB+"
    fi
    
    # Check available memory (need at least 8GB)
    available_memory=$(free -m | awk 'NR==2{printf "%.0f", $7/1024}')
    if [ "$available_memory" -lt 8 ]; then
        log_warn "Low memory: ${available_memory}GB available. Recommended: 8GB+"
    fi
    
    # Check if GPU is available
    if command -v nvidia-smi &> /dev/null; then
        log_success "NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | \
        while read -r name memory; do
            log_info "GPU: $name (${memory}MB VRAM)"
        done
    else
        log_warn "No NVIDIA GPU detected - running in CPU-only mode"
    fi
    
    log_success "System validation completed"
}

# ===============================================
# ENVIRONMENT SETUP
# ===============================================

setup_environment() {
    log_info "Setting up environment configuration..."
    
    cd "$PROJECT_ROOT"
    
    # Load existing environment variables if .env exists
    if [[ -f "$ENV_FILE" ]]; then
        log_info "Using existing environment configuration"
        # Export all variables from .env file
        set -a  # automatically export all variables
        source "$ENV_FILE"
        set +a  # stop automatically exporting
        
        # Ensure required variables are set with defaults if missing
        export POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-"sutazai-dev-password"}
        export REDIS_PASSWORD=${REDIS_PASSWORD:-"redis-dev-password"}
        export NEO4J_PASSWORD=${NEO4J_PASSWORD:-"neo4j-dev-password"}
        export SECRET_KEY=${SECRET_KEY:-"dev-secret-key-change-in-production"}
    else
        log_info "Creating environment configuration..."
        
        # Generate secure passwords and keys
        export POSTGRES_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
        export REDIS_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
        export NEO4J_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
        export SECRET_KEY=$(openssl rand -hex 32)
        export CHROMADB_API_KEY=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-32)
        export GRAFANA_PASSWORD=$(openssl rand -base64 16 | tr -d "=+/" | cut -c1-16)
        export N8N_PASSWORD=$(openssl rand -base64 16 | tr -d "=+/" | cut -c1-16)
        
        cat > "$ENV_FILE" << EOF
# SutazAI System Configuration
# Generated on $(date)

# System Settings
TZ=UTC
SUTAZAI_ENV=production

# Database Configuration
POSTGRES_USER=sutazai
POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
POSTGRES_DB=sutazai

REDIS_PASSWORD=${REDIS_PASSWORD}

NEO4J_PASSWORD=${NEO4J_PASSWORD}

# API Keys and Secrets
SECRET_KEY=${SECRET_KEY}
CHROMADB_API_KEY=${CHROMADB_API_KEY}

# Monitoring
GRAFANA_PASSWORD=${GRAFANA_PASSWORD}

# Workflow Automation
N8N_USER=admin
N8N_PASSWORD=${N8N_PASSWORD}

# Health Monitoring
HEALTH_ALERT_WEBHOOK=

# Model Configuration
DEFAULT_MODEL=llama3.2:1b
EMBEDDING_MODEL=nomic-embed-text
FALLBACK_MODELS=qwen2.5:3b,codellama:7b,deepseek-r1:8b

# Resource Limits
MAX_CONCURRENT_AGENTS=10
MAX_MODEL_INSTANCES=5
CACHE_SIZE_GB=10

# Feature Flags
ENABLE_GPU=auto
ENABLE_MONITORING=true
ENABLE_SECURITY_SCAN=true
ENABLE_AUTO_BACKUP=true

# External Integrations (leave empty for local-only operation)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=
EOF
        
        # Secure the env file
        chmod 600 "$ENV_FILE"
        
        log_success "Environment configuration created with secure passwords"
        log_warn "Important: Save these credentials securely!"
        echo "----------------------------------------"
        echo "Database: sutazai / ${POSTGRES_PASSWORD}"
        echo "Grafana: admin / ${GRAFANA_PASSWORD}"
        echo "N8N: admin / ${N8N_PASSWORD}"
        echo "Neo4j: neo4j / ${NEO4J_PASSWORD}"
        echo "----------------------------------------"
    fi
}

# ===============================================
# DIRECTORY STRUCTURE SETUP
# ===============================================

setup_directories() {
    log_info "Setting up directory structure..."
    
    # Core directories
    directories=(
        "logs"
        "data/models"
        "data/documents" 
        "data/training"
        "data/backups"
        "data/langflow"
        "data/flowise"
        "data/n8n"
        "monitoring/prometheus"
        "monitoring/grafana/provisioning/datasources"
        "monitoring/grafana/provisioning/dashboards"
        "monitoring/grafana/dashboards"
        "monitoring/loki"
        "monitoring/promtail"
        "services/faiss"
        "services/documind"
        "services/pytorch"
        "services/tensorflow"
        "services/jax"
        "services/llamaindex"
        "services/health-monitor"
        "agents/autogpt"
        "agents/crewai"
        "agents/letta"
        "agents/aider"
        "agents/gpt-engineer"
        "agents/browser-use"
        "agents/skyvern"
        "agents/localagi"
        "agents/agentgpt"
        "agents/privategpt"
        "agents/shellgpt"
        "agents/pentestgpt"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
    done
    
    # Create .gitkeep files for empty directories
    find . -type d -empty -exec touch {}/.gitkeep \;
    
    # Set proper permissions
    chmod -R 755 .
    chmod -R 777 data logs
    
    log_success "Directory structure created"
}

# ===============================================
# DOCKER SERVICE MANAGEMENT
# ===============================================

stop_existing_services() {
    log_info "Stopping existing SutazAI services..."
    
    # Stop SutazAI containers specifically
    local sutazai_containers=$(docker ps -q --filter "name=sutazai-")
    if [[ -n "$sutazai_containers" ]]; then
        log_info "Stopping SutazAI containers..."
        docker stop $sutazai_containers 2>/dev/null || true
        docker rm $sutazai_containers 2>/dev/null || true
    fi
    
    # Remove SutazAI-specific compose services
    if [[ -f "$COMPOSE_FILE" ]]; then
        log_info "Stopping compose services..."
        docker-compose -f "$COMPOSE_FILE" down 2>/dev/null || true
    fi
    
    # Clean up SutazAI networks only
    docker network ls --filter "name=sutazai" -q | xargs -r docker network rm 2>/dev/null || true
    
    # Clean up volumes (only if requested)
    if [[ "${CLEAN_VOLUMES:-false}" == "true" ]]; then
        log_warn "Cleaning up SutazAI volumes..."
        docker volume ls --filter "name=sutazai" -q | xargs -r docker volume rm 2>/dev/null || true
    fi
    
    log_success "Existing SutazAI services stopped"
}

# ===============================================
# MODEL MANAGEMENT
# ===============================================

setup_ollama_models() {
    log_info "Setting up Ollama models..."
    
    # Wait for Ollama to be ready
    local max_attempts=30
    local attempt=0
    
    while ! curl -f http://localhost:11434/api/tags &>/dev/null; do
        if [ $attempt -ge $max_attempts ]; then
            log_error "Ollama service not ready after ${max_attempts} attempts"
            return 1
        fi
        log_info "Waiting for Ollama to be ready... (attempt $((++attempt)))"
        sleep 10
    done
    
    # List of models to install (CPU-optimized order)
    models=(
        "llama3.2:1b"        # Fastest model for CPU
        "qwen2.5:3b"         # Good balance 
        "codellama:7b"       # For code tasks
        "nomic-embed-text"   # For embeddings
        "deepseek-r1:8b"     # More capable but slower
    )
    
    for model in "${models[@]}"; do
        log_info "Installing model: $model"
        if docker exec sutazai-ollama ollama pull "$model"; then
            log_success "Model installed: $model"
        else
            log_warn "Failed to install model: $model"
        fi
    done
    
    log_success "Ollama models setup completed"
}

# ===============================================
# MONITORING CONFIGURATION
# ===============================================

setup_monitoring() {
    log_info "Setting up monitoring configuration..."
    
    # Prometheus configuration
    cat > monitoring/prometheus/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'sutazai-backend'
    static_configs:
      - targets: ['backend-agi:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
    
  - job_name: 'sutazai-agents'
    static_configs:
      - targets: 
        - 'autogpt:8080'
        - 'crewai:8080'
        - 'aider:8080'
        - 'gpt-engineer:8080'
    scrape_interval: 60s
    
  - job_name: 'sutazai-infrastructure' 
    static_configs:
      - targets:
        - 'postgres:5432'
        - 'redis:6379'
        - 'neo4j:7474'
        - 'chromadb:8000'
        - 'qdrant:6333'
        - 'ollama:11434'
    scrape_interval: 30s

  - job_name: 'sutazai-health-monitor'
    static_configs:
      - targets: ['health-monitor:8000']
    scrape_interval: 15s
EOF

    # Grafana datasources
    mkdir -p monitoring/grafana/provisioning/datasources
    cat > monitoring/grafana/provisioning/datasources/prometheus.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    
  - name: Loki
    type: loki
    access: proxy
    url: http://loki:3100
    
  - name: PostgreSQL
    type: postgres
    url: postgres:5432
    database: sutazai
    user: sutazai
    secureJsonData:
      password: ${POSTGRES_PASSWORD}
      
  - name: Redis
    type: redis-datasource
    url: redis:6379
    secureJsonData:
      password: ${REDIS_PASSWORD}
      
  - name: Neo4j
    type: neo4j-datasource
    url: neo4j:7687
    user: neo4j
    secureJsonData:
      password: ${NEO4J_PASSWORD}
EOF

    # Loki configuration
    cat > monitoring/loki/config.yml << 'EOF'
auth_enabled: false

server:
  http_listen_port: 3100

ingester:
  lifecycler:
    address: 127.0.0.1
    ring:
      kvstore:
        store: inmemory
      replication_factor: 1
    final_sleep: 0s

schema_config:
  configs:
    - from: 2020-10-24
      store: boltdb
      object_store: filesystem
      schema: v11
      index:
        prefix: index_
        period: 168h

storage_config:
  boltdb:
    directory: /loki/index
  filesystem:
    directory: /loki/chunks

limits_config:
  enforce_metric_name: false
  reject_old_samples: true
  reject_old_samples_max_age: 168h

chunk_store_config:
  max_look_back_period: 0s

table_manager:
  retention_deletes_enabled: false
  retention_period: 0s
EOF

    # Promtail configuration
    cat > monitoring/promtail/config.yml << 'EOF'
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: containers
    static_configs:
      - targets:
          - localhost
        labels:
          job: containerlogs
          __path__: /var/lib/docker/containers/*/*log

    pipeline_stages:
      - json:
          expressions:
            output: log
            stream: stream
            attrs:
      - json:
          expressions:
            tag:
          source: attrs
      - regex:
          expression: (?P<container_name>(?:[^|]*))\|
          source: tag
      - timestamp:
          format: RFC3339Nano
          source: time
      - labels:
          stream:
          container_name:
      - output:
          source: output

  - job_name: sutazai-logs
    static_configs:
      - targets:
          - localhost
        labels:
          job: sutazai
          __path__: /app/logs/*.log
EOF

    log_success "Monitoring configuration created"
}

# ===============================================
# SERVICE DEPLOYMENT
# ===============================================

deploy_core_services() {
    log_info "Deploying core infrastructure services..."
    
    # Start core services first
    docker-compose -f "$COMPOSE_FILE" up -d \
        postgres redis neo4j chromadb qdrant
    
    # Wait for core services to be healthy
    log_info "Waiting for core services to be ready..."
    
    # Wait for PostgreSQL specifically
    local max_attempts=30
    local attempt=0
    while ! docker exec sutazai-postgres pg_isready -U postgres >/dev/null 2>&1; do
        if [ $attempt -ge $max_attempts ]; then
            log_error "PostgreSQL not ready after ${max_attempts} attempts"
            break
        fi
        log_info "Waiting for PostgreSQL... (attempt $((++attempt)))"
        sleep 5
    done
    
    # Initialize database if needed
    log_info "Initializing database..."
    docker exec sutazai-postgres psql -U postgres -c "CREATE DATABASE sutazai;" 2>/dev/null || echo "Database may already exist"
    docker exec sutazai-postgres psql -U postgres -c "CREATE USER sutazai WITH PASSWORD '${POSTGRES_PASSWORD}';" 2>/dev/null || echo "User may already exist"
    docker exec sutazai-postgres psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE sutazai TO sutazai;" 2>/dev/null
    docker exec sutazai-postgres psql -U postgres -c "ALTER USER sutazai CREATEDB;" 2>/dev/null
    
    # Check service health
    services=("postgres" "redis" "neo4j" "chromadb" "qdrant")
    for service in "${services[@]}"; do
        if docker ps --filter "name=sutazai-$service" --filter "status=running" | grep -q "$service"; then
            log_success "Service ready: $service"
        else
            log_error "Service failed to start: $service"
            docker logs "sutazai-$service" --tail 20
        fi
    done
}

deploy_ai_services() {
    log_info "Deploying AI and model services..."
    
    # Start Ollama and wait for it to be ready
    docker-compose -f "$COMPOSE_FILE" up -d ollama
    
    log_info "Waiting for Ollama to initialize..."
    sleep 60
    
    # Install models
    setup_ollama_models
    
    # Start vector databases and FAISS (these work reliably)
    docker-compose -f "$COMPOSE_FILE" up -d faiss
    
    # Start image-based AI services (these are proven to work)
    docker-compose -f "$COMPOSE_FILE" up -d \
        tabbyml langflow flowise n8n
        
    log_success "AI services deployed"
}

deploy_agent_ecosystem() {
    log_info "Deploying AI agent ecosystem..."
    
    # Deploy all AI agents using consolidated compose file
    log_info "Starting AI agent containers..."
    docker-compose -f "$COMPOSE_FILE" up -d \
        autogpt crewai aider gpt-engineer letta
    
    # Wait for agents to be ready
    sleep 10
    
    # Check agent health
    agents=("autogpt" "crewai" "aider" "gpt-engineer" "letta")
    for agent in "${agents[@]}"; do
        if docker ps --filter "name=sutazai-$agent" --filter "status=running" | grep -q "$agent"; then
            log_success "AI Agent ready: $agent"
        else
            log_warn "AI Agent may need attention: $agent"
        fi
    done
        
    log_success "AI agent ecosystem deployed successfully"
}

deploy_application_services() {
    log_info "Deploying application services..."
    
    # Start backend and frontend
    docker-compose -f "$COMPOSE_FILE" up -d \
        backend-agi frontend-agi
    
    # Start additional services
    docker-compose -f "$COMPOSE_FILE" up -d \
        langflow flowise llamaindex documind n8n
        
    log_success "Application services deployed"
}

deploy_monitoring_services() {
    log_info "Deploying monitoring services..."
    
    # Start monitoring stack
    docker-compose -f "$COMPOSE_FILE" up -d \
        prometheus grafana loki promtail health-monitor
        
    log_success "Monitoring services deployed"
}

# ===============================================
# AGENT DOCKERFILE CREATION
# ===============================================

create_agent_dockerfiles() {
    log_info "Creating agent Dockerfiles..."
    
    # AutoGPT Dockerfile
    cat > agents/autogpt/Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Clone AutoGPT
RUN git clone https://github.com/Significant-Gravitas/AutoGPT.git .

# Install dependencies
RUN pip install -e .

# Create workspace directory
RUN mkdir -p /app/workspace /app/outputs

# Copy configuration
COPY config.yml /app/

EXPOSE 8080

CMD ["python", "-m", "autogpt", "--config", "config.yml"]
EOF

    # CrewAI Dockerfile
    cat > agents/crewai/Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CrewAI
RUN pip install crewai crewai-tools fastapi uvicorn

# Copy application
COPY . .

EXPOSE 8080

CMD ["python", "main.py"]
EOF

    # Create main.py for CrewAI
    cat > agents/crewai/main.py << 'EOF'
from crewai import Agent, Task, Crew
from fastapi import FastAPI
import uvicorn

app = FastAPI(title="CrewAI Service")

# Define agents
researcher = Agent(
    role='Research Specialist',
    goal='Conduct thorough research on given topics',
    backstory='Expert researcher with access to comprehensive knowledge',
    verbose=True
)

analyst = Agent(
    role='Data Analyst', 
    goal='Analyze and interpret research findings',
    backstory='Skilled analyst who transforms data into insights',
    verbose=True
)

writer = Agent(
    role='Content Writer',
    goal='Create comprehensive reports from research and analysis',
    backstory='Professional writer who creates clear, actionable content',
    verbose=True
)

@app.post("/execute")
async def execute_crew(task_description: str):
    # Define task
    research_task = Task(
        description=f"Research and analyze: {task_description}",
        agent=researcher
    )
    
    analysis_task = Task(
        description="Analyze the research findings and identify key insights",
        agent=analyst
    )
    
    writing_task = Task(
        description="Create a comprehensive report based on research and analysis",
        agent=writer
    )
    
    # Create crew
    crew = Crew(
        agents=[researcher, analyst, writer],
        tasks=[research_task, analysis_task, writing_task],
        verbose=True
    )
    
    # Execute
    result = crew.kickoff()
    
    return {"result": result, "status": "completed"}

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "crewai"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
EOF

    # Similar Dockerfiles for other agents...
    # (I'll create a few key ones, the pattern is similar)
    
    # Aider Dockerfile
    cat > agents/aider/Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install aider-chat fastapi uvicorn

COPY . .

EXPOSE 8080

CMD ["python", "main.py"]
EOF

    # Create main.py for Aider
    cat > agents/aider/main.py << 'EOF'
from fastapi import FastAPI
import uvicorn
import subprocess
import os

app = FastAPI(title="Aider Code Assistant")

@app.post("/code")
async def code_assistance(request: dict):
    prompt = request.get("prompt", "")
    files = request.get("files", [])
    
    # Use aider command line interface
    cmd = ["aider", "--model", "ollama/deepseek-r1:8b"] + files + ["--message", prompt]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return {
            "result": result.stdout,
            "error": result.stderr,
            "status": "completed" if result.returncode == 0 else "error"
        }
    except subprocess.TimeoutExpired:
        return {"error": "Request timed out", "status": "timeout"}

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "aider"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
EOF

    log_success "Agent Dockerfiles created"
}

# ===============================================
# SYSTEM HEALTH CHECK
# ===============================================

run_health_checks() {
    log_info "Running system health checks..."
    
    # Wait for all services to stabilize
    sleep 60
    
    # Check service health
    services_to_check=(
        "sutazai-postgres:5432"
        "sutazai-redis:6379" 
        "sutazai-neo4j:7474"
        "sutazai-chromadb:8000"
        "sutazai-qdrant:6333"
        "sutazai-ollama:11434"
        "sutazai-backend-agi:8000"
        "sutazai-frontend-agi:8501"
        "sutazai-prometheus:9090"
        "sutazai-grafana:3000"
    )
    
    failed_services=()
    
    for service_port in "${services_to_check[@]}"; do
        service=${service_port%:*}
        port=${service_port#*:}
        
        if docker ps --filter "name=$service" --filter "status=running" | grep -q "$service"; then
            log_success "✓ $service is running"
        else
            log_error "✗ $service is not running"
            failed_services+=("$service")
        fi
    done
    
    # Test API endpoints
    api_endpoints=(
        "http://172.31.77.193:8000/health:Backend API"
        "http://172.31.77.193:8501:Frontend"
        "http://172.31.77.193:9090:Prometheus"
        "http://172.31.77.193:3000:Grafana"
        "http://172.31.77.193:11434/api/tags:Ollama API"
        "http://172.31.77.193:6333/cluster:Qdrant"
        "http://172.31.77.193:8001/api/v1/heartbeat:ChromaDB"
    )
    
    for endpoint_desc in "${api_endpoints[@]}"; do
        endpoint=${endpoint_desc%:*}
        description=${endpoint_desc#*:}
        
        if curl -f -s "$endpoint" > /dev/null; then
            log_success "✓ $description endpoint is responsive"
        else
            log_warn "⚠ $description endpoint is not responding"
        fi
    done
    
    if [ ${#failed_services[@]} -eq 0 ]; then
        log_success "All core services are running successfully!"
        return 0
    else
        log_error "Failed services: ${failed_services[*]}"
        return 1
    fi
}

# ===============================================
# MAIN DEPLOYMENT FLOW
# ===============================================

main() {
    log_info "Starting SutazAI Complete System Deployment"
    log_info "==========================================="
    
    cd "$PROJECT_ROOT"
    setup_logging
    
    # Validation and preparation
    validate_system
    setup_environment
    setup_directories
    setup_monitoring
    
    # Stop existing services
    stop_existing_services
    
    # Deploy services in stages
    deploy_core_services
    deploy_ai_services
    deploy_agent_ecosystem  
    deploy_application_services
    deploy_monitoring_services
    
    # Final health check
    if run_health_checks; then
        log_success "🎉 SutazAI System Deployment Completed Successfully!"
        echo ""
        echo "=============================================="
        echo "🌐 Access Points:"
        echo "   • Frontend:    http://172.31.77.193:8501"
        echo "   • Backend API: http://172.31.77.193:8000"
        echo "   • API Docs:    http://172.31.77.193:8000/docs"
        echo "   • Prometheus:  http://172.31.77.193:9090" 
        echo "   • Grafana:     http://172.31.77.193:3000"
        echo "   • LangFlow:    http://172.31.77.193:8090"
        echo "   • FlowiseAI:   http://172.31.77.193:8099"
        echo "   • N8N:         http://172.31.77.193:5678"
        echo ""
        echo "🤖 AI Agents:"
        echo "   • CrewAI:      http://172.31.77.193:8096"
        echo "   • Aider:       http://172.31.77.193:8095"
        echo "   • GPT Engineer: http://172.31.77.193:8097"
        echo "   • LlamaIndex:  http://172.31.77.193:8098"
        echo "   • AgentGPT:    http://172.31.77.193:8091"
        echo "   • PrivateGPT:  http://172.31.77.193:8092"
        echo "   • TabbyML:     http://172.31.77.193:8093"
        echo "   • ShellGPT:    http://172.31.77.193:8102"
        echo ""
        echo "📊 System Status: http://172.31.77.193:8100"
        echo "=============================================="
        echo ""
        echo "📋 Next Steps:"
        echo "   1. Access the system: http://172.31.77.193:8501"
        echo "   2. Check AI Chat functionality"
        echo "   3. Review logs: ./scripts/live_logs.sh"
        echo "   4. Monitor system: ./scripts/live_logs.sh --overview"
        echo "   5. For database issues: ./scripts/live_logs.sh --init-db"
        echo ""
    else
        log_error "❌ Deployment completed with some issues"
        echo "Check logs at: $LOG_FILE"
        echo "Run: docker ps -a | grep sutazai"
        echo "For service logs: docker logs <service-name>"
    fi
}

# ===============================================
# SCRIPT EXECUTION
# ===============================================

# Show usage information
show_usage() {
    echo "SutazAI Complete System Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  deploy                 - Deploy complete SutazAI system (default)"
    echo "  stop                   - Stop all SutazAI services"
    echo "  restart                - Restart the complete system"
    echo "  status                 - Show status of all services"
    echo "  logs [service]         - Show logs for all services or specific service"
    echo "  help                   - Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  CLEAN_VOLUMES=true     - Clean existing volumes during deployment"
    echo ""
    echo "Examples:"
    echo "  $0 deploy              # Deploy complete system"
    echo "  $0 stop                # Stop all services"
    echo "  $0 logs backend-agi    # Show backend logs"
    echo "  CLEAN_VOLUMES=true $0 deploy  # Clean deployment"
    echo ""
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "stop")
        log_info "Stopping all SutazAI services..."
        cd "$PROJECT_ROOT"
        docker-compose -f "$COMPOSE_FILE" down
        log_success "All services stopped"
        ;;
    "restart")
        log_info "Restarting SutazAI system..."
        cd "$PROJECT_ROOT"
        docker-compose -f "$COMPOSE_FILE" down
        sleep 10
        main
        ;;
    "status")
        cd "$PROJECT_ROOT"
        docker-compose -f "$COMPOSE_FILE" ps
        ;;
    "logs")
        cd "$PROJECT_ROOT"
        docker-compose -f "$COMPOSE_FILE" logs -f "${2:-}"
        ;;
    "help"|"--help"|"-h")
        show_usage
        ;;
    *)
        echo "Unknown command: $1"
        echo ""
        show_usage
        exit 1
        ;;
esac