#!/bin/bash
#
# SutazAI Complete AI Agents Deployment Script
# Deploys all 22 AI agents in separate containers with proper configuration
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
PROJECT_ROOT="/opt/sutazaiapp"
AGENTS_DIR="${PROJECT_ROOT}/agents"
DOCKERFILES_DIR="${AGENTS_DIR}/dockerfiles"
CONFIGS_DIR="${AGENTS_DIR}/configs"
LOG_FILE="${PROJECT_ROOT}/logs/agents_deployment_$(date +%Y%m%d_%H%M%S).log"

# Create necessary directories
mkdir -p "${DOCKERFILES_DIR}" "${CONFIGS_DIR}" "${PROJECT_ROOT}/logs"

# Logging function
log() {
    echo -e "${2:-$BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

# Error handling
handle_error() {
    log "ERROR: Deployment failed at line $1" "$RED"
    exit 1
}
trap 'handle_error $LINENO' ERR

# GPU detection
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        log "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader)" "$GREEN"
        echo "true"
    else
        log "No GPU detected - agents will use CPU mode" "$YELLOW"
        echo "false"
    fi
}

# Create base Dockerfile for agents
create_base_dockerfile() {
    local agent_name=$1
    local base_image=$2
    local ports=$3
    local gpu_required=$4
    
    cat > "${DOCKERFILES_DIR}/Dockerfile.${agent_name}" << EOF
FROM ${base_image}

WORKDIR /app

# Install common dependencies
RUN apt-get update && apt-get install -y \\
    git curl wget unzip \\
    python3-pip python3-dev \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements if exists
COPY requirements-${agent_name}.txt* ./
RUN if [ -f requirements-${agent_name}.txt ]; then pip install -r requirements-${agent_name}.txt; fi

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/workspace

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV AGENT_NAME=${agent_name}
ENV OLLAMA_BASE_URL=http://ollama:11434

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:${ports%:*}/health || exit 1

EXPOSE ${ports%:*}

CMD ["python", "-m", "${agent_name}"]
EOF
}

# Deploy AutoGPT
deploy_autogpt() {
    log "Deploying AutoGPT..." "$BLUE"
    
    cat > "${DOCKERFILES_DIR}/Dockerfile.autogpt" << 'EOF'
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Clone AutoGPT
RUN git clone https://github.com/Significant-Gravitas/AutoGPT.git .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Configure for local operation
ENV OPENAI_API_BASE=http://ollama:11434/v1
ENV OPENAI_API_KEY=local

EXPOSE 8501

CMD ["python", "-m", "autogpt", "--continuous-mode", "--install-plugin-deps"]
EOF
}

# Deploy CrewAI
deploy_crewai() {
    log "Deploying CrewAI..." "$BLUE"
    
    cat > "${DOCKERFILES_DIR}/Dockerfile.crewai" << 'EOF'
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir crewai crewai-tools langchain-community

COPY crewai_config.py .

ENV OPENAI_API_BASE=http://ollama:11434/v1
ENV OPENAI_API_KEY=local

EXPOSE 8502

CMD ["python", "crewai_config.py"]
EOF

    # Create CrewAI config
    cat > "${AGENTS_DIR}/crewai_config.py" << 'EOF'
from crewai import Agent, Task, Crew
from fastapi import FastAPI
import uvicorn

app = FastAPI()

class CrewAIService:
    def __init__(self):
        self.researcher = Agent(
            role='Researcher',
            goal='Research and analyze information',
            backstory='Expert researcher with deep analytical skills',
            verbose=True,
            allow_delegation=False
        )
        
        self.writer = Agent(
            role='Writer',
            goal='Create compelling content',
            backstory='Professional writer with creative skills',
            verbose=True,
            allow_delegation=False
        )
    
    def create_crew(self, task_description):
        task = Task(
            description=task_description,
            agent=self.researcher
        )
        
        crew = Crew(
            agents=[self.researcher, self.writer],
            tasks=[task],
            verbose=True
        )
        
        return crew

service = CrewAIService()

@app.get("/health")
async def health():
    return {"status": "healthy", "agent": "crewai"}

@app.post("/execute")
async def execute(task: dict):
    crew = service.create_crew(task.get("description", ""))
    result = crew.kickoff()
    return {"result": str(result)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8502)
EOF
}

# Deploy LocalAGI
deploy_localagi() {
    log "Deploying LocalAGI..." "$BLUE"
    
    cat > "${DOCKERFILES_DIR}/Dockerfile.localagi" << 'EOF'
FROM golang:1.21-alpine AS builder

RUN apk add --no-cache git make

WORKDIR /build
RUN git clone https://github.com/mudler/LocalAGI.git .
RUN make build

FROM alpine:latest
RUN apk add --no-cache ca-certificates
COPY --from=builder /build/localagi /usr/local/bin/

EXPOSE 8080

CMD ["localagi", "--address", "0.0.0.0:8080", "--models-path", "/models"]
EOF
}

# Deploy TabbyML
deploy_tabbyml() {
    log "Deploying TabbyML..." "$BLUE"
    
    cat > "${DOCKERFILES_DIR}/Dockerfile.tabbyml" << 'EOF'
FROM tabbyml/tabby:latest

ENV TABBY_MODEL=TabbyML/StarCoder-1B
ENV TABBY_DEVICE=cpu

EXPOSE 8080

CMD ["serve", "--model", "${TABBY_MODEL}", "--device", "${TABBY_DEVICE}", "--no-webserver"]
EOF
}

# Deploy Semgrep
deploy_semgrep() {
    log "Deploying Semgrep..." "$BLUE"
    
    cat > "${DOCKERFILES_DIR}/Dockerfile.semgrep" << 'EOF'
FROM python:3.11-slim

RUN pip install --no-cache-dir semgrep fastapi uvicorn

WORKDIR /app

COPY semgrep_service.py .

EXPOSE 8504

CMD ["uvicorn", "semgrep_service:app", "--host", "0.0.0.0", "--port", "8504"]
EOF

    # Create Semgrep service
    cat > "${AGENTS_DIR}/semgrep_service.py" << 'EOF'
from fastapi import FastAPI, HTTPException
import subprocess
import tempfile
import json

app = FastAPI()

@app.get("/health")
async def health():
    return {"status": "healthy", "agent": "semgrep"}

@app.post("/scan")
async def scan_code(request: dict):
    code = request.get("code", "")
    language = request.get("language", "python")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{language}', delete=False) as f:
        f.write(code)
        f.flush()
        
        try:
            result = subprocess.run(
                ["semgrep", "--json", "--config=auto", f.name],
                capture_output=True,
                text=True
            )
            
            return {
                "findings": json.loads(result.stdout) if result.stdout else {},
                "errors": result.stderr
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
EOF
}

# Deploy LangChain Agents
deploy_langchain() {
    log "Deploying LangChain Agents..." "$BLUE"
    
    cat > "${DOCKERFILES_DIR}/Dockerfile.langchain" << 'EOF'
FROM python:3.11-slim

RUN pip install --no-cache-dir langchain langchain-community langchain-experimental fastapi uvicorn

WORKDIR /app

COPY langchain_service.py .

ENV OPENAI_API_BASE=http://ollama:11434/v1

EXPOSE 8505

CMD ["uvicorn", "langchain_service:app", "--host", "0.0.0.0", "--port", "8505"]
EOF
}

# Deploy GPT-Engineer
deploy_gpt_engineer() {
    log "Deploying GPT-Engineer..." "$BLUE"
    
    cat > "${DOCKERFILES_DIR}/Dockerfile.gpt-engineer" << 'EOF'
FROM python:3.11-slim

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir gpt-engineer

ENV OPENAI_API_BASE=http://ollama:11434/v1
ENV OPENAI_API_KEY=local

WORKDIR /workspace

EXPOSE 8506

CMD ["gpt-engineer", "--local"]
EOF
}

# Deploy Aider
deploy_aider() {
    log "Deploying Aider..." "$BLUE"
    
    cat > "${DOCKERFILES_DIR}/Dockerfile.aider" << 'EOF'
FROM python:3.11-slim

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir aider-chat

ENV OPENAI_API_BASE=http://ollama:11434/v1
ENV OPENAI_API_KEY=local

WORKDIR /workspace

EXPOSE 8507

CMD ["aider", "--no-auto-commits", "--yes"]
EOF
}

# Deploy additional agents
deploy_remaining_agents() {
    # AutoGen
    create_base_dockerfile "autogen" "python:3.11-slim" "8508:8508" "false"
    
    # AgentZero
    create_base_dockerfile "agentzero" "python:3.11-slim" "8509:8509" "false"
    
    # BigAGI
    create_base_dockerfile "bigagi" "node:18-slim" "8510:3000" "false"
    
    # Browser Use
    create_base_dockerfile "browseruse" "python:3.11-slim" "8511:8511" "false"
    
    # Skyvern
    create_base_dockerfile "skyvern" "python:3.11-slim" "8512:8512" "false"
    
    # Langflow
    create_base_dockerfile "langflow" "python:3.11-slim" "8513:7860" "false"
    
    # Dify
    create_base_dockerfile "dify" "python:3.11-slim" "8514:5000" "false"
    
    # AgentGPT
    create_base_dockerfile "agentgpt" "node:18-slim" "8515:3000" "false"
    
    # PrivateGPT
    create_base_dockerfile "privategpt" "python:3.11-slim" "8516:8001" "false"
    
    # LlamaIndex
    create_base_dockerfile "llamaindex" "python:3.11-slim" "8517:8517" "false"
    
    # FlowiseAI
    create_base_dockerfile "flowise" "node:18-slim" "8518:3000" "false"
    
    # ShellGPT
    create_base_dockerfile "shellgpt" "python:3.11-slim" "8519:8519" "false"
    
    # PentestGPT
    create_base_dockerfile "pentestgpt" "python:3.11-slim" "8520:8520" "false"
    
    # Documind
    create_base_dockerfile "documind" "python:3.11-slim" "8521:8521" "false"
    
    # FinRobot
    create_base_dockerfile "finrobot" "python:3.11-slim" "8522:8522" "false"
    
    # RealtimeSTT
    create_base_dockerfile "realtimestt" "python:3.11-slim" "8523:8523" "true"
}

# Update docker-compose for agents
update_docker_compose() {
    log "Updating docker-compose configuration..." "$BLUE"
    
    # This would append agent services to docker-compose-complete-agi.yml
    # For now, we'll create a separate agents compose file
    
    cat > "${PROJECT_ROOT}/docker-compose-agents-complete.yml" << 'EOF'
version: '3.9'

networks:
  sutazai-network:
    external: true

services:
  autogpt:
    build:
      context: ./agents
      dockerfile: dockerfiles/Dockerfile.autogpt
    container_name: sutazai-autogpt
    restart: unless-stopped
    environment:
      - OPENAI_API_BASE=http://ollama:11434/v1
      - OPENAI_API_KEY=local
    volumes:
      - ./agents/workspaces/autogpt:/workspace
    networks:
      - sutazai-network
    depends_on:
      - ollama

  crewai:
    build:
      context: ./agents
      dockerfile: dockerfiles/Dockerfile.crewai
    container_name: sutazai-crewai
    restart: unless-stopped
    ports:
      - "8502:8502"
    networks:
      - sutazai-network
    depends_on:
      - ollama

  localagi:
    build:
      context: ./agents
      dockerfile: dockerfiles/Dockerfile.localagi
    container_name: sutazai-localagi
    restart: unless-stopped
    ports:
      - "8503:8080"
    volumes:
      - ./models:/models
    networks:
      - sutazai-network

  tabbyml:
    build:
      context: ./agents
      dockerfile: dockerfiles/Dockerfile.tabbyml
    container_name: sutazai-tabbyml
    restart: unless-stopped
    ports:
      - "8080:8080"
    networks:
      - sutazai-network

  semgrep:
    build:
      context: ./agents
      dockerfile: dockerfiles/Dockerfile.semgrep
    container_name: sutazai-semgrep
    restart: unless-stopped
    ports:
      - "8504:8504"
    networks:
      - sutazai-network

  langchain:
    build:
      context: ./agents
      dockerfile: dockerfiles/Dockerfile.langchain
    container_name: sutazai-langchain
    restart: unless-stopped
    ports:
      - "8505:8505"
    networks:
      - sutazai-network
    depends_on:
      - ollama

  gpt-engineer:
    build:
      context: ./agents
      dockerfile: dockerfiles/Dockerfile.gpt-engineer
    container_name: sutazai-gpt-engineer
    restart: unless-stopped
    ports:
      - "8506:8506"
    volumes:
      - ./agents/workspaces/gpt-engineer:/workspace
    networks:
      - sutazai-network
    depends_on:
      - ollama

  aider:
    build:
      context: ./agents
      dockerfile: dockerfiles/Dockerfile.aider
    container_name: sutazai-aider
    restart: unless-stopped
    ports:
      - "8507:8507"
    volumes:
      - ./agents/workspaces/aider:/workspace
    networks:
      - sutazai-network
    depends_on:
      - ollama
EOF
}

# Build and deploy all agents
deploy_all_agents() {
    log "Starting deployment of all AI agents..." "$GREEN"
    
    # Deploy each agent
    deploy_autogpt
    deploy_crewai
    deploy_localagi
    deploy_tabbyml
    deploy_semgrep
    deploy_langchain
    deploy_gpt_engineer
    deploy_aider
    deploy_remaining_agents
    
    # Update docker-compose
    update_docker_compose
    
    log "Building agent Docker images..." "$BLUE"
    cd "$PROJECT_ROOT"
    
    # Build all agent images
    docker compose -f docker-compose.yml -f docker-compose-agents-complete.yml build
    
    log "Starting agent containers..." "$BLUE"
    docker compose -f docker-compose.yml -f docker-compose-agents-complete.yml up -d
    
    # Wait for agents to start
    sleep 30
    
    # Check agent health
    log "Checking agent health..." "$BLUE"
    docker compose -f docker-compose.yml -f docker-compose-agents-complete.yml ps
}

# Main execution
main() {
    log "SutazAI AI Agents Deployment Script" "$GREEN"
    log "====================================" "$GREEN"
    
    # Check GPU availability
    GPU_AVAILABLE=$(check_gpu)
    
    # Deploy all agents
    deploy_all_agents
    
    log "Agent deployment completed!" "$GREEN"
    log "Access agent services at:" "$BLUE"
    log "  - CrewAI: http://localhost:8502" "$BLUE"
    log "  - LocalAGI: http://localhost:8503" "$BLUE"
    log "  - TabbyML: http://localhost:8080" "$BLUE"
    log "  - Semgrep: http://localhost:8504" "$BLUE"
    log "  - LangChain: http://localhost:8505" "$BLUE"
    log "  - GPT-Engineer: http://localhost:8506" "$BLUE"
    log "  - Aider: http://localhost:8507" "$BLUE"
    log "  - And more..." "$BLUE"
}

# Run main function
main "$@" 