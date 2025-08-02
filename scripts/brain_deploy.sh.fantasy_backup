#!/bin/bash
# SutazAI Brain - One-Command Deployment Script
# Deploys the complete 100% local AGI/ASI system

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "$BRAIN_DIR")"
LOG_FILE="$BRAIN_DIR/logs/brain_deploy.log"

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" | tee -a "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARN:${NC} $1" | tee -a "$LOG_FILE"
}

log_info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO:${NC} $1" | tee -a "$LOG_FILE"
}

# Create necessary directories
create_directories() {
    log "🗂️  Creating Brain directories..."
    mkdir -p "$BRAIN_DIR"/{logs,config,data,models/adapters}
    mkdir -p "$BRAIN_DIR"/agents/{dockerfiles,configs}
    mkdir -p "$BRAIN_DIR"/monitoring/{dashboards,alerts}
}

# Check prerequisites
check_prerequisites() {
    log "🔍 Checking prerequisites..."
    
    # Check if main deployment is running
    if ! docker ps | grep -q "sutazai-ollama"; then
        log_error "Main SutazAI deployment not running. Please run deploy_complete_system.sh first."
        exit 1
    fi
    
    # Check required services
    local required_services=("ollama" "redis" "postgresql" "qdrant" "chromadb")
    for service in "${required_services[@]}"; do
        if ! docker ps | grep -q "sutazai-$service"; then
            log_error "Required service not running: sutazai-$service"
            exit 1
        fi
    done
    
    # Check available resources
    local available_memory=$(free -g | awk '/^Mem:/{print $7}')
    if [ "$available_memory" -lt 8 ]; then
        log_warn "Low available memory: ${available_memory}GB. Brain may experience performance issues."
    fi
    
    log "✅ Prerequisites check passed"
}

# Initialize Git repository for self-improvement
init_git_repo() {
    log "🔧 Initializing Git repository for self-improvement..."
    
    cd "$BRAIN_DIR"
    
    if [ ! -d .git ]; then
        git init
        git config user.name "SutazAI Brain"
        git config user.email "brain@sutazai.local"
        
        # Create .gitignore
        cat > .gitignore << EOF
__pycache__/
*.py[cod]
*$py.class
*.so
.env
logs/
data/
models/*.bin
models/*.gguf
*.backup
.DS_Store
EOF
        
        git add .
        git commit -m "Initial Brain commit"
    fi
    
    log "✅ Git repository initialized"
}

# Create Brain configuration
create_config() {
    log "⚙️  Creating Brain configuration..."
    
    cat > "$BRAIN_DIR/config/brain_config.yaml" << EOF
# SutazAI Brain Configuration
# Auto-generated on $(date)

# Hardware constraints (adjust based on your system)
max_memory_gb: 48.0
gpu_memory_gb: 4.0
cpu_cores: $(nproc)

# Model settings
default_embedding_model: nomic-embed-text
default_reasoning_model: tinyllama
default_coding_model: codellama:7b
evaluation_model: tinyllama
comparison_model: qwen2.5:7b

# Quality thresholds
min_quality_score: 0.85
improvement_threshold: 0.85
memory_retention_days: 30

# Parallelism
max_concurrent_agents: 5
max_model_instances: 3

# Self-improvement
auto_improve: true
pr_batch_size: 50
require_human_approval: true

# Service configuration
ollama_host: http://sutazai-ollama:11434
redis_host: sutazai-redis
qdrant_host: sutazai-qdrant
chroma_host: sutazai-chromadb
postgres_host: sutazai-postgresql
brain_repo_path: /workspace/brain
EOF
    
    log "✅ Configuration created"
}

# Create Brain Dockerfile
create_brain_dockerfile() {
    log "🐳 Creating Brain Dockerfile..."
    
    cat > "$BRAIN_DIR/Dockerfile" << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy Brain code
COPY . .

# Create necessary directories
RUN mkdir -p logs data models/adapters

# Set environment
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose API port
EXPOSE 8888

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8888/health || exit 1

# Start Brain
CMD ["python", "main.py"]
EOF
    
    log "✅ Dockerfile created"
}

# Create requirements.txt
create_requirements() {
    log "📦 Creating requirements.txt..."
    
    cat > "$BRAIN_DIR/requirements.txt" << EOF
# Core dependencies
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pyyaml==6.0.1

# LangChain and LangGraph
langchain==0.1.0
langchain-community==0.1.0
langgraph==0.0.20

# Vector databases
qdrant-client==1.7.0
chromadb==0.4.18
redis[hiredis]==5.0.1

# Database
asyncpg==0.29.0
sqlalchemy==2.0.23

# ML/Embeddings
sentence-transformers==2.2.2
torch==2.1.1
numpy==1.24.3

# Utilities
httpx==0.25.2
aiofiles==23.2.1
psutil==5.9.6
GitPython==3.1.40
docker==7.0.0

# Monitoring
prometheus-client==0.19.0
EOF
    
    log "✅ Requirements created"
}

# Create agent Dockerfiles
create_agent_dockerfiles() {
    log "🤖 Creating agent Dockerfiles..."
    
    # Base agent Dockerfile template
    cat > "$BRAIN_DIR/agents/dockerfiles/base_agent.dockerfile" << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install agent-specific dependencies
ARG AGENT_NAME
COPY requirements_${AGENT_NAME}.txt .
RUN pip install --no-cache-dir -r requirements_${AGENT_NAME}.txt

# Copy agent code
COPY ${AGENT_NAME}_agent.py .

# Set environment
ENV PYTHONUNBUFFERED=1
ENV AGENT_TYPE=${AGENT_NAME}

# Expose agent port
EXPOSE 8080

# Start agent
CMD ["python", "-m", "uvicorn", "${AGENT_NAME}_agent:app", "--host", "0.0.0.0", "--port", "8080"]
EOF
    
    # Create a sample agent implementation
    cat > "$BRAIN_DIR/agents/sample_agent.py" << 'EOF'
#!/usr/bin/env python3
"""
Sample Agent Implementation
Replace this with actual agent implementations
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any

app = FastAPI()

class AgentRequest(BaseModel):
    input: str
    task_plan: list
    context: Dict[str, Any]

class AgentResponse(BaseModel):
    output: Any
    quality_score: float = 0.8

@app.post("/execute")
async def execute(request: AgentRequest) -> AgentResponse:
    # Implement agent logic here
    return AgentResponse(
        output=f"Processed: {request.input}",
        quality_score=0.85
    )

@app.get("/health")
async def health():
    return {"status": "healthy"}
EOF
    
    log "✅ Agent Dockerfiles created"
}

# Create docker-compose for Brain
create_docker_compose() {
    log "🐳 Creating Brain docker-compose.yml..."
    
    cat > "$BRAIN_DIR/docker-compose.yml" << EOF
version: '3.8'

services:
  brain:
    build: .
    container_name: sutazai-brain
    restart: unless-stopped
    ports:
      - "8888:8888"
    environment:
      - POSTGRES_PASSWORD=\${POSTGRES_PASSWORD:-sutazai_password}
    volumes:
      - ./:/app
      - ./logs:/app/logs
      - ./data:/app/data
      - ./models:/app/models
    networks:
      - sutazai-network
    depends_on:
      - brain-db-init
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G

  brain-db-init:
    image: postgres:16-alpine
    container_name: sutazai-brain-db-init
    environment:
      - PGHOST=sutazai-postgresql
      - PGUSER=sutazai
      - PGPASSWORD=\${POSTGRES_PASSWORD:-sutazai_password}
    volumes:
      - ./sql/init_brain_db.sql:/init.sql
    networks:
      - sutazai-network
    command: >
      sh -c "
        until pg_isready -h sutazai-postgresql; do
          echo 'Waiting for PostgreSQL...'
          sleep 2
        done
        psql -h sutazai-postgresql -U sutazai -d postgres -c 'CREATE DATABASE IF NOT EXISTS sutazai_brain;'
        psql -h sutazai-postgresql -U sutazai -d sutazai_brain -f /init.sql
      "
    restart: "no"

networks:
  sutazai-network:
    external: true
EOF
    
    log "✅ Docker-compose created"
}

# Create database initialization script
create_db_init() {
    log "🗄️  Creating database initialization script..."
    
    mkdir -p "$BRAIN_DIR/sql"
    
    cat > "$BRAIN_DIR/sql/init_brain_db.sql" << 'EOF'
-- SutazAI Brain Database Schema

-- Memory audit table
CREATE TABLE IF NOT EXISTS memory_audit (
    id SERIAL PRIMARY KEY,
    memory_id VARCHAR(255) NOT NULL,
    action VARCHAR(50) NOT NULL,
    metadata JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Memory searches table
CREATE TABLE IF NOT EXISTS memory_searches (
    id SERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    results_count INTEGER,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Memory statistics table
CREATE TABLE IF NOT EXISTS memory_stats (
    id SERIAL PRIMARY KEY,
    action VARCHAR(50) NOT NULL,
    count INTEGER,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Agent performance table
CREATE TABLE IF NOT EXISTS agent_performance (
    id SERIAL PRIMARY KEY,
    agent_name VARCHAR(100) NOT NULL,
    task_id VARCHAR(255),
    execution_time FLOAT,
    success BOOLEAN,
    quality_score FLOAT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Improvement patches table
CREATE TABLE IF NOT EXISTS improvement_patches (
    id SERIAL PRIMARY KEY,
    patch_id VARCHAR(255) UNIQUE NOT NULL,
    description TEXT,
    files_changed TEXT[],
    diff TEXT,
    test_results JSONB,
    pr_url VARCHAR(500),
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    applied_at TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_memory_audit_memory_id ON memory_audit(memory_id);
CREATE INDEX idx_memory_searches_timestamp ON memory_searches(timestamp);
CREATE INDEX idx_agent_performance_agent ON agent_performance(agent_name);
CREATE INDEX idx_improvement_patches_status ON improvement_patches(status);
EOF
    
    log "✅ Database initialization script created"
}

# Create monitoring setup
create_monitoring() {
    log "📊 Creating monitoring configuration..."
    
    # Prometheus configuration
    cat > "$BRAIN_DIR/monitoring/prometheus.yml" << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'brain'
    static_configs:
      - targets: ['sutazai-brain:8888']
    metrics_path: '/metrics'
EOF
    
    # Create a basic Grafana dashboard JSON
    cat > "$BRAIN_DIR/monitoring/dashboards/brain_dashboard.json" << 'EOF'
{
  "dashboard": {
    "title": "SutazAI Brain Dashboard",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [{"expr": "rate(brain_requests_total[5m])"}]
      },
      {
        "title": "Agent Performance",
        "targets": [{"expr": "brain_agent_quality_score"}]
      },
      {
        "title": "Memory Usage",
        "targets": [{"expr": "brain_memory_entries_total"}]
      },
      {
        "title": "Improvement Patches",
        "targets": [{"expr": "brain_patches_created_total"}]
      }
    ]
  }
}
EOF
    
    log "✅ Monitoring configuration created"
}

# Create initialization files
create_init_files() {
    log "📝 Creating initialization files..."
    
    # Create __init__.py files
    touch "$BRAIN_DIR"/__init__.py
    touch "$BRAIN_DIR"/core/__init__.py
    touch "$BRAIN_DIR"/agents/__init__.py
    touch "$BRAIN_DIR"/memory/__init__.py
    touch "$BRAIN_DIR"/evaluator/__init__.py
    touch "$BRAIN_DIR"/improver/__init__.py
    
    # Create README
    cat > "$BRAIN_DIR/README.md" << EOF
# SutazAI Brain - 100% Local AGI/ASI System

## Overview
The Brain is a self-improving AGI/ASI system that orchestrates 25+ LLMs and 30+ specialized agents to solve complex tasks.

## Architecture
- **Perception Layer**: Processes inputs and retrieves memories
- **Working Memory**: Multi-tier memory system (Redis + Qdrant + ChromaDB)
- **Reasoning Core**: LangGraph-based orchestration
- **Execution Engine**: 30+ containerized agents
- **Meta-Learning**: Self-improvement through patch generation
- **Self-Repair**: Automated PR creation for improvements

## Quick Start
\`\`\`bash
./deploy.sh
\`\`\`

## API Endpoints
- \`POST /process\`: Process a request through the Brain
- \`GET /health\`: Health check
- \`GET /status\`: Detailed system status
- \`GET /agents\`: List available agents
- \`GET /memory/stats\`: Memory system statistics

## Configuration
Edit \`config/brain_config.yaml\` to adjust:
- Hardware limits
- Model selection
- Quality thresholds
- Self-improvement settings

## Monitoring
Access Grafana dashboard at http://localhost:3000/d/brain
EOF
    
    log "✅ Initialization files created"
}

# Pull required Ollama models
pull_ollama_models() {
    log "🤖 Pulling required Ollama models..."
    
    local models=(
        "tinyllama"
        "codellama:7b"
        "qwen2.5:7b"
        "nomic-embed-text"
    )
    
    for model in "${models[@]}"; do
        log_info "Pulling $model..."
        docker exec sutazai-ollama ollama pull "$model" || log_warn "Failed to pull $model"
    done
    
    log "✅ Model pulling complete"
}

# Build and deploy Brain
deploy_brain() {
    log "🚀 Deploying Brain system..."
    
    cd "$BRAIN_DIR"
    
    # Build Brain image
    log_info "Building Brain Docker image..."
    docker-compose build
    
    # Start Brain services
    log_info "Starting Brain services..."
    docker-compose up -d
    
    # Wait for Brain to initialize
    log_info "Waiting for Brain to initialize..."
    local max_attempts=30
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -f http://localhost:8888/health >/dev/null 2>&1; then
            log "✅ Brain is healthy and ready!"
            break
        fi
        attempt=$((attempt + 1))
        log_info "Waiting for Brain... (attempt $attempt/$max_attempts)"
        sleep 5
    done
    
    if [ $attempt -eq $max_attempts ]; then
        log_error "Brain failed to start properly"
        docker-compose logs brain
        exit 1
    fi
}

# Show deployment summary
show_summary() {
    log "🎉 Brain deployment complete!"
    echo
    echo "========================================="
    echo "🧠 SutazAI Brain is now running!"
    echo "========================================="
    echo
    echo "📡 API Endpoint: http://localhost:8888"
    echo "📊 Status: http://localhost:8888/status"
    echo "🤖 Agents: http://localhost:8888/agents"
    echo
    echo "🔧 Configuration: $BRAIN_DIR/config/brain_config.yaml"
    echo "📝 Logs: $BRAIN_DIR/logs/"
    echo
    echo "🚀 Quick test:"
    echo "   curl -X POST http://localhost:8888/process \\"
    echo "     -H 'Content-Type: application/json' \\"
    echo "     -d '{\"input\": \"Write a hello world function in Python\"}'"
    echo
    echo "💡 The Brain will now continuously learn and improve itself!"
    echo "========================================="
}

# Main deployment flow
main() {
    log "🧠 Starting SutazAI Brain Deployment"
    
    # Create log directory first
    mkdir -p "$BRAIN_DIR/logs"
    
    # Execute deployment steps
    create_directories
    check_prerequisites
    init_git_repo
    create_config
    create_brain_dockerfile
    create_requirements
    create_agent_dockerfiles
    create_docker_compose
    create_db_init
    create_monitoring
    create_init_files
    pull_ollama_models
    deploy_brain
    show_summary
}

# Execute main function
main "$@"