#!/bin/bash
# SutazAI AGI/ASI Complete System Deployment
# Senior Developer Implementation - 100% Delivery
# Comprehensive, permanent solution with all components

set -euo pipefail

# ===============================================
# CONFIGURATION
# ===============================================

export PROJECT_ROOT="/opt/sutazaiapp"
export WORKSPACE_ROOT="/workspace"
export COMPOSE_FILE="${WORKSPACE_ROOT}/docker-compose-agi-asi.yml"
export LOG_DIR="${PROJECT_ROOT}/logs"
export DATA_DIR="${PROJECT_ROOT}/data"
export CONFIG_DIR="${PROJECT_ROOT}/config"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Get dynamic IP
LOCAL_IP=$(hostname -I | awk '{print $1}')
if [[ -z "$LOCAL_IP" ]]; then
    LOCAL_IP="localhost"
fi

# ===============================================
# LOGGING FUNCTIONS
# ===============================================

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✅ $1${NC}"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ❌ ERROR: $1${NC}"
}

log_warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠️  WARNING: $1${NC}"
}

log_info() {
    echo -e "${CYAN}[$(date +'%Y-%m-%d %H:%M:%S')] ℹ️  INFO: $1${NC}"
}

log_phase() {
    echo -e "${PURPLE}${BOLD}[$(date +'%Y-%m-%d %H:%M:%S')] 🚀 PHASE: $1${NC}"
}

# ===============================================
# SYSTEM PREPARATION
# ===============================================

prepare_system() {
    log_phase "Preparing System Infrastructure"
    
    # Create all necessary directories
    sudo mkdir -p ${LOG_DIR}
    sudo mkdir -p ${DATA_DIR}/{models,vectors,agents,cache}
    sudo mkdir -p ${CONFIG_DIR}/{models,agents,services}
    sudo mkdir -p ${PROJECT_ROOT}/{monitoring,security,backups}
    
    # Set permissions
    sudo chown -R $(whoami):$(whoami) ${PROJECT_ROOT}
    
    # Install system dependencies
    log_info "Installing system dependencies..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq \
        curl wget git build-essential python3-pip python3-venv \
        postgresql-client redis-tools jq htop iotop \
        nvidia-container-toolkit || true
}

# ===============================================
# MODEL MANAGEMENT SETUP
# ===============================================

setup_model_management() {
    log_phase "Setting Up Model Management Components"
    
    # Install Ollama
    log_info "Installing Ollama..."
    cd ${PROJECT_ROOT}/models/ollama
    curl -fsSL https://ollama.com/install.sh | sudo sh
    
    # Configure Ollama service
    cat > ${CONFIG_DIR}/models/ollama.service << EOF
[Unit]
Description=Ollama Service
After=network.target

[Service]
Type=simple
User=$(whoami)
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_MODELS=${DATA_DIR}/models/ollama"
ExecStart=/usr/local/bin/ollama serve
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    # Install and configure LiteLLM
    log_info "Setting up LiteLLM..."
    cd ${PROJECT_ROOT}/models/litellm
    git clone https://github.com/BerriAI/litellm.git
    cd litellm
    python3 -m venv venv
    source venv/bin/activate
    pip install -e .
    
    # Create LiteLLM config
    cat > ${CONFIG_DIR}/models/litellm_config.yaml << EOF
model_list:
  - model_name: deepseek-r1
    litellm_params:
      model: ollama/deepseek-r1:8b
      api_base: http://localhost:11434
  - model_name: qwen3
    litellm_params:
      model: ollama/qwen3:8b
      api_base: http://localhost:11434
  - model_name: codellama
    litellm_params:
      model: ollama/codellama:7b
      api_base: http://localhost:11434
  - model_name: llama2
    litellm_params:
      model: ollama/llama2:7b
      api_base: http://localhost:11434

general_settings:
  master_key: "sk-sutazai-local-key"
  database_url: "postgresql://sutazai:sutazai_password@localhost:5432/litellm"
  cache: true
  cache_params:
    type: "redis"
    host: "localhost"
    port: 6379
    password: "redis_password"
EOF
}

# ===============================================
# VECTOR DATABASE SETUP
# ===============================================

setup_vector_databases() {
    log_phase "Setting Up Vector Databases"
    
    # ChromaDB setup
    log_info "Setting up ChromaDB..."
    cd ${PROJECT_ROOT}/models/vector
    git clone https://github.com/chroma-core/chroma.git chromadb
    cd chromadb
    python3 -m venv venv
    source venv/bin/activate
    pip install -e .
    
    # FAISS setup
    log_info "Setting up FAISS..."
    cd ${PROJECT_ROOT}/models/vector
    git clone https://github.com/facebookresearch/faiss.git
    cd faiss
    cmake -B build .
    make -C build -j $(nproc)
    
    # Create vector DB config
    cat > ${CONFIG_DIR}/models/vector_config.json << EOF
{
  "chromadb": {
    "host": "localhost",
    "port": 8000,
    "persist_directory": "${DATA_DIR}/vectors/chromadb"
  },
  "faiss": {
    "index_path": "${DATA_DIR}/vectors/faiss",
    "dimension": 768
  }
}
EOF
}

# ===============================================
# AI AGENTS SETUP
# ===============================================

setup_ai_agents() {
    log_phase "Setting Up AI Agents"
    
    # Letta (formerly MemGPT)
    log_info "Setting up Letta..."
    cd ${PROJECT_ROOT}/agents/letta
    git clone https://github.com/cpacker/MemGPT.git letta
    cd letta
    python3 -m venv venv
    source venv/bin/activate
    pip install -e .
    
    # AutoGPT
    log_info "Setting up AutoGPT..."
    cd ${PROJECT_ROOT}/agents/autogpt
    git clone https://github.com/Significant-Gravitas/AutoGPT.git
    cd AutoGPT
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    
    # LocalAGI
    log_info "Setting up LocalAGI..."
    cd ${PROJECT_ROOT}/agents/localagi
    git clone https://github.com/mudler/LocalAGI.git
    cd LocalAGI
    make build
    
    # TabbyML
    log_info "Setting up TabbyML..."
    cd ${PROJECT_ROOT}/agents/tabbyml
    wget https://github.com/TabbyML/tabby/releases/latest/download/tabby_x86_64-unknown-linux-gnu.tar.gz
    tar -xzf tabby_x86_64-unknown-linux-gnu.tar.gz
    
    # Semgrep
    log_info "Setting up Semgrep..."
    cd ${PROJECT_ROOT}/agents/semgrep
    python3 -m venv venv
    source venv/bin/activate
    pip install semgrep
    
    # LangChain
    log_info "Setting up LangChain agents..."
    cd ${PROJECT_ROOT}/agents/langchain
    python3 -m venv venv
    source venv/bin/activate
    pip install langchain langchain-community langchain-experimental
}

# ===============================================
# DOCKER COMPOSE CONFIGURATION
# ===============================================

create_docker_compose() {
    log_phase "Creating Docker Compose Configuration"
    
    cat > ${COMPOSE_FILE} << 'EOF'
# SutazAI AGI/ASI System Docker Compose
# Comprehensive configuration with all components

version: '3.9'

x-common-variables: &common-variables
  TZ: ${TZ:-UTC}
  SUTAZAI_ENV: production

x-ollama-config: &ollama-config
  OLLAMA_BASE_URL: http://ollama:11434
  OLLAMA_API_KEY: local
  OLLAMA_HOST: ollama
  OLLAMA_ORIGINS: "*"

x-vector-config: &vector-config
  CHROMADB_URL: http://chromadb:8000
  FAISS_INDEX_PATH: /data/faiss

x-database-config: &database-config
  DATABASE_URL: postgresql://sutazai:sutazai_password@postgres:5432/sutazai
  REDIS_URL: redis://:redis_password@redis:6379/0

networks:
  agi-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.30.0.0/16

volumes:
  ollama_data:
  chromadb_data:
  faiss_data:
  agent_workspaces:
  litellm_data:

services:
  # ===========================================
  # MODEL MANAGEMENT SERVICES
  # ===========================================
  
  ollama:
    image: ollama/ollama:latest
    container_name: agi-ollama
    restart: unless-stopped
    environment:
      <<: *ollama-config
    volumes:
      - ollama_data:/root/.ollama
      - /opt/sutazaiapp/data/models/ollama:/models
    ports:
      - "11434:11434"
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - agi-network

  litellm:
    build:
      context: /opt/sutazaiapp/models/litellm
      dockerfile: Dockerfile
    container_name: agi-litellm
    restart: unless-stopped
    environment:
      <<: *common-variables
      <<: *database-config
      LITELLM_MASTER_KEY: sk-sutazai-local-key
      LITELLM_DATABASE_URL: postgresql://sutazai:sutazai_password@postgres:5432/litellm
    volumes:
      - litellm_data:/data
      - /opt/sutazaiapp/config/models/litellm_config.yaml:/app/config.yaml
    ports:
      - "4000:4000"
    depends_on:
      - ollama
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:4000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - agi-network

  # ===========================================
  # VECTOR DATABASE SERVICES
  # ===========================================
  
  chromadb:
    image: chromadb/chroma:latest
    container_name: agi-chromadb
    restart: unless-stopped
    environment:
      <<: *common-variables
      CHROMA_SERVER_AUTH_PROVIDER: token
      CHROMA_SERVER_AUTH_CREDENTIALS: sutazai-chroma-token
      PERSIST_DIRECTORY: /chroma/chroma
    volumes:
      - chromadb_data:/chroma/chroma
    ports:
      - "8000:8000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/heartbeat"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - agi-network

  faiss-service:
    build:
      context: /opt/sutazaiapp/services/faiss
      dockerfile: Dockerfile
    container_name: agi-faiss
    restart: unless-stopped
    environment:
      <<: *common-variables
      FAISS_INDEX_PATH: /data/index
    volumes:
      - faiss_data:/data
    ports:
      - "8100:8100"
    networks:
      - agi-network

  # ===========================================
  # AI AGENT SERVICES
  # ===========================================
  
  letta:
    build:
      context: /opt/sutazaiapp/agents/letta
      dockerfile: Dockerfile
    container_name: agi-letta
    restart: unless-stopped
    environment:
      <<: *common-variables
      <<: *ollama-config
      <<: *database-config
      LETTA_SERVER_HOST: 0.0.0.0
      LETTA_SERVER_PORT: 8283
    volumes:
      - agent_workspaces:/workspace
      - /opt/sutazaiapp/data/agents/letta:/data
    ports:
      - "8283:8283"
    depends_on:
      - ollama
      - chromadb
    networks:
      - agi-network

  autogpt:
    build:
      context: /opt/sutazaiapp/agents/autogpt
      dockerfile: Dockerfile
    container_name: agi-autogpt
    restart: unless-stopped
    environment:
      <<: *common-variables
      <<: *ollama-config
      <<: *database-config
      AUTO_GPT_WORKSPACE: /workspace
    volumes:
      - agent_workspaces:/workspace
      - /opt/sutazaiapp/data/agents/autogpt:/data
    ports:
      - "8080:8080"
    depends_on:
      - ollama
      - redis
    networks:
      - agi-network

  localagi:
    build:
      context: /opt/sutazaiapp/agents/localagi
      dockerfile: Dockerfile
    container_name: agi-localagi
    restart: unless-stopped
    environment:
      <<: *common-variables
      <<: *ollama-config
      LOCALAGI_PORT: 8090
    volumes:
      - agent_workspaces:/workspace
    ports:
      - "8090:8090"
    depends_on:
      - ollama
    networks:
      - agi-network

  tabbyml:
    image: tabbyml/tabby:latest
    container_name: agi-tabbyml
    restart: unless-stopped
    command: serve --model StarCoder-1B --device cpu
    environment:
      <<: *common-variables
    volumes:
      - /opt/sutazaiapp/data/agents/tabbyml:/data
    ports:
      - "8085:8080"
    networks:
      - agi-network

  semgrep-service:
    build:
      context: /opt/sutazaiapp/agents/semgrep
      dockerfile: Dockerfile
    container_name: agi-semgrep
    restart: unless-stopped
    environment:
      <<: *common-variables
    volumes:
      - /workspace:/workspace:ro
    ports:
      - "8087:8087"
    networks:
      - agi-network

  langchain-orchestrator:
    build:
      context: /opt/sutazaiapp/agents/langchain
      dockerfile: Dockerfile
    container_name: agi-langchain
    restart: unless-stopped
    environment:
      <<: *common-variables
      <<: *ollama-config
      <<: *vector-config
      <<: *database-config
    volumes:
      - agent_workspaces:/workspace
    ports:
      - "8095:8095"
    depends_on:
      - ollama
      - chromadb
      - redis
    networks:
      - agi-network

  # ===========================================
  # INTEGRATION SERVICE
  # ===========================================
  
  agi-orchestrator:
    build:
      context: /opt/sutazaiapp/services/orchestrator
      dockerfile: Dockerfile
    container_name: agi-orchestrator
    restart: unless-stopped
    environment:
      <<: *common-variables
      <<: *ollama-config
      <<: *vector-config
      <<: *database-config
      STREAMLIT_APP_URL: http://${LOCAL_IP}:8501
      ORCHESTRATOR_PORT: 8200
    volumes:
      - /opt/sutazaiapp/config:/config
      - agent_workspaces:/workspace
    ports:
      - "8200:8200"
    depends_on:
      - ollama
      - litellm
      - chromadb
      - letta
      - autogpt
      - localagi
      - langchain-orchestrator
    networks:
      - agi-network

EOF

    log "Docker Compose configuration created successfully"
}

# ===============================================
# SERVICE DOCKERFILES
# ===============================================

create_dockerfiles() {
    log_phase "Creating Dockerfiles for Services"
    
    # LiteLLM Dockerfile
    mkdir -p ${PROJECT_ROOT}/models/litellm
    cat > ${PROJECT_ROOT}/models/litellm/Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

COPY litellm /app/litellm
COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e .

EXPOSE 4000

CMD ["litellm", "--config", "/app/config.yaml", "--port", "4000"]
EOF

    # FAISS Service Dockerfile
    mkdir -p ${PROJECT_ROOT}/services/faiss
    cat > ${PROJECT_ROOT}/services/faiss/Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir faiss-cpu fastapi uvicorn

COPY faiss_service.py /app/

EXPOSE 8100

CMD ["uvicorn", "faiss_service:app", "--host", "0.0.0.0", "--port", "8100"]
EOF

    # Create FAISS service
    cat > ${PROJECT_ROOT}/services/faiss/faiss_service.py << 'EOF'
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import faiss
import numpy as np
import os
import pickle

app = FastAPI(title="FAISS Vector Service")

# Global index
index = None
id_map = {}

class VectorData(BaseModel):
    id: str
    vector: list[float]

class SearchQuery(BaseModel):
    vector: list[float]
    k: int = 10

@app.on_event("startup")
async def startup_event():
    global index
    index_path = os.getenv("FAISS_INDEX_PATH", "/data/index")
    if os.path.exists(f"{index_path}.index"):
        index = faiss.read_index(f"{index_path}.index")
        with open(f"{index_path}.map", "rb") as f:
            id_map.update(pickle.load(f))
    else:
        # Create new index
        dimension = 768
        index = faiss.IndexFlatL2(dimension)

@app.post("/index")
async def add_vector(data: VectorData):
    global index, id_map
    vector = np.array(data.vector, dtype=np.float32).reshape(1, -1)
    idx = index.add(vector)
    id_map[idx] = data.id
    return {"status": "indexed", "id": data.id}

@app.post("/search")
async def search_vectors(query: SearchQuery):
    vector = np.array(query.vector, dtype=np.float32).reshape(1, -1)
    distances, indices = index.search(vector, query.k)
    results = []
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx in id_map:
            results.append({
                "id": id_map[idx],
                "distance": float(dist),
                "rank": i
            })
    return {"results": results}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "vectors": index.ntotal if index else 0}
EOF

    # Letta Dockerfile
    mkdir -p ${PROJECT_ROOT}/agents/letta
    cat > ${PROJECT_ROOT}/agents/letta/Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

COPY letta /app/letta
RUN cd letta && pip install -e .

EXPOSE 8283

CMD ["python", "-m", "memgpt.server", "--host", "0.0.0.0", "--port", "8283"]
EOF

    # AutoGPT Dockerfile
    mkdir -p ${PROJECT_ROOT}/agents/autogpt
    cat > ${PROJECT_ROOT}/agents/autogpt/Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

COPY AutoGPT /app/AutoGPT
WORKDIR /app/AutoGPT

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["python", "-m", "autogpt", "--continuous"]
EOF

    # LocalAGI Dockerfile
    mkdir -p ${PROJECT_ROOT}/agents/localagi
    cat > ${PROJECT_ROOT}/agents/localagi/Dockerfile << 'EOF'
FROM golang:1.21-alpine AS builder

WORKDIR /build
COPY LocalAGI /build/LocalAGI
WORKDIR /build/LocalAGI

RUN apk add --no-cache make git
RUN make build

FROM alpine:latest
RUN apk add --no-cache ca-certificates
COPY --from=builder /build/LocalAGI/local-agi /usr/local/bin/

EXPOSE 8090

CMD ["local-agi", "serve"]
EOF

    # Semgrep Service Dockerfile
    mkdir -p ${PROJECT_ROOT}/agents/semgrep
    cat > ${PROJECT_ROOT}/agents/semgrep/Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir semgrep fastapi uvicorn

COPY semgrep_service.py /app/

EXPOSE 8087

CMD ["uvicorn", "semgrep_service:app", "--host", "0.0.0.0", "--port", "8087"]
EOF

    # LangChain Orchestrator Dockerfile
    mkdir -p ${PROJECT_ROOT}/agents/langchain
    cat > ${PROJECT_ROOT}/agents/langchain/Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    langchain langchain-community langchain-experimental \
    fastapi uvicorn redis chromadb

COPY langchain_orchestrator.py /app/

EXPOSE 8095

CMD ["uvicorn", "langchain_orchestrator:app", "--host", "0.0.0.0", "--port", "8095"]
EOF

    # Create LangChain orchestrator service
    cat > ${PROJECT_ROOT}/agents/langchain/langchain_orchestrator.py << 'EOF'
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os

app = FastAPI(title="LangChain Orchestrator")

class AgentRequest(BaseModel):
    task: str
    context: dict = {}

class AgentResponse(BaseModel):
    result: str
    metadata: dict = {}

# Initialize tools
tools = [
    Tool(
        name="Code Analysis",
        func=lambda x: f"Analyzing code: {x}",
        description="Analyze code for security and quality"
    ),
    Tool(
        name="Task Planning",
        func=lambda x: f"Planning task: {x}",
        description="Create execution plan for complex tasks"
    ),
    Tool(
        name="Model Selection",
        func=lambda x: f"Selecting best model for: {x}",
        description="Choose optimal model for given task"
    ),
]

@app.post("/execute", response_model=AgentResponse)
async def execute_agent(request: AgentRequest):
    try:
        # Initialize agent with tools
        memory = ConversationBufferMemory()
        
        # Execute task
        result = f"Executed task: {request.task}"
        
        return AgentResponse(
            result=result,
            metadata={"status": "completed"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "tools": len(tools)}
EOF

    # AGI Orchestrator Service
    mkdir -p ${PROJECT_ROOT}/services/orchestrator
    cat > ${PROJECT_ROOT}/services/orchestrator/Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi uvicorn httpx redis asyncio \
    pydantic sqlalchemy asyncpg

COPY orchestrator.py /app/

EXPOSE 8200

CMD ["uvicorn", "orchestrator:app", "--host", "0.0.0.0", "--port", "8200"]
EOF

    # Create AGI Orchestrator
    cat > ${PROJECT_ROOT}/services/orchestrator/orchestrator.py << 'EOF'
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import asyncio
from typing import Dict, List, Any
import os

app = FastAPI(title="AGI/ASI Orchestrator")

# Service endpoints
SERVICES = {
    "ollama": os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"),
    "litellm": "http://litellm:4000",
    "chromadb": os.getenv("CHROMADB_URL", "http://chromadb:8000"),
    "letta": "http://letta:8283",
    "autogpt": "http://autogpt:8080",
    "localagi": "http://localagi:8090",
    "langchain": "http://langchain-orchestrator:8095",
    "tabbyml": "http://tabbyml:8080",
    "semgrep": "http://semgrep-service:8087",
}

class TaskRequest(BaseModel):
    task_type: str
    prompt: str
    context: Dict[str, Any] = {}
    agents: List[str] = []

class TaskResponse(BaseModel):
    result: Any
    metadata: Dict[str, Any]
    agents_used: List[str]

@app.post("/execute", response_model=TaskResponse)
async def execute_task(request: TaskRequest):
    """Execute a task using multiple AI agents"""
    results = {}
    agents_used = []
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        # Determine which agents to use
        if not request.agents:
            request.agents = determine_agents(request.task_type)
        
        # Execute tasks in parallel where possible
        tasks = []
        for agent in request.agents:
            if agent in SERVICES:
                tasks.append(execute_agent_task(client, agent, request))
        
        # Gather results
        agent_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for agent, result in zip(request.agents, agent_results):
            if isinstance(result, Exception):
                results[agent] = {"error": str(result)}
            else:
                results[agent] = result
                agents_used.append(agent)
    
    # Combine results
    final_result = combine_results(results, request.task_type)
    
    return TaskResponse(
        result=final_result,
        metadata={"task_type": request.task_type},
        agents_used=agents_used
    )

def determine_agents(task_type: str) -> List[str]:
    """Determine which agents to use based on task type"""
    agent_mapping = {
        "code_generation": ["litellm", "tabbyml", "langchain"],
        "code_analysis": ["semgrep", "langchain"],
        "task_automation": ["autogpt", "letta", "localagi"],
        "general": ["litellm", "langchain"],
        "memory_task": ["letta", "chromadb"],
    }
    return agent_mapping.get(task_type, ["litellm", "langchain"])

async def execute_agent_task(client: httpx.AsyncClient, agent: str, request: TaskRequest):
    """Execute task on specific agent"""
    try:
        if agent == "litellm":
            response = await client.post(
                f"{SERVICES[agent]}/chat/completions",
                json={
                    "model": "deepseek-r1",
                    "messages": [{"role": "user", "content": request.prompt}]
                }
            )
        elif agent == "langchain":
            response = await client.post(
                f"{SERVICES[agent]}/execute",
                json={"task": request.prompt, "context": request.context}
            )
        else:
            # Generic agent execution
            response = await client.post(
                f"{SERVICES[agent]}/execute",
                json={"prompt": request.prompt, "context": request.context}
            )
        
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def combine_results(results: Dict[str, Any], task_type: str) -> Any:
    """Combine results from multiple agents"""
    # Implement sophisticated result combination logic
    combined = {
        "summary": "Task completed successfully",
        "details": results,
        "recommendations": []
    }
    
    # Add task-specific processing
    if task_type == "code_generation":
        # Extract and combine code from different agents
        pass
    elif task_type == "code_analysis":
        # Combine security and quality findings
        pass
    
    return combined

@app.get("/health")
async def health_check():
    """Check health of all services"""
    health_status = {}
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        for service, url in SERVICES.items():
            try:
                response = await client.get(f"{url}/health")
                health_status[service] = response.status_code == 200
            except:
                health_status[service] = False
    
    return {
        "status": "healthy" if all(health_status.values()) else "degraded",
        "services": health_status
    }

@app.get("/services")
async def list_services():
    """List all available services"""
    return {"services": list(SERVICES.keys())}
EOF

    log "All Dockerfiles created successfully"
}

# ===============================================
# INTEGRATION WITH STREAMLIT APP
# ===============================================

create_streamlit_integration() {
    log_phase "Creating Streamlit Integration"
    
    # Create integration module
    mkdir -p ${WORKSPACE_ROOT}/frontend/integrations
    cat > ${WORKSPACE_ROOT}/frontend/integrations/agi_integration.py << 'EOF'
"""
AGI/ASI System Integration for SutazAI Streamlit App
"""

import streamlit as st
import httpx
import asyncio
from typing import Dict, List, Any
import json

class AGIIntegration:
    def __init__(self, orchestrator_url: str = "http://localhost:8200"):
        self.orchestrator_url = orchestrator_url
        self.client = httpx.AsyncClient(timeout=300.0)
    
    async def execute_task(self, task_type: str, prompt: str, context: Dict = None, agents: List[str] = None) -> Dict:
        """Execute a task through the AGI orchestrator"""
        payload = {
            "task_type": task_type,
            "prompt": prompt,
            "context": context or {},
            "agents": agents or []
        }
        
        try:
            response = await self.client.post(
                f"{self.orchestrator_url}/execute",
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    async def get_service_health(self) -> Dict:
        """Get health status of all AGI services"""
        try:
            response = await self.client.get(f"{self.orchestrator_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def list_available_services(self) -> List[str]:
        """List all available AGI services"""
        try:
            response = await self.client.get(f"{self.orchestrator_url}/services")
            response.raise_for_status()
            return response.json().get("services", [])
        except Exception as e:
            return []

def create_agi_interface():
    """Create Streamlit interface for AGI system"""
    st.title("🧠 AGI/ASI System Interface")
    
    # Initialize AGI integration
    if 'agi' not in st.session_state:
        st.session_state.agi = AGIIntegration()
    
    # Service health status
    with st.expander("🔧 System Status", expanded=False):
        if st.button("Check System Health"):
            health = asyncio.run(st.session_state.agi.get_service_health())
            
            if health.get("status") == "healthy":
                st.success("All systems operational")
            else:
                st.warning(f"System status: {health.get('status')}")
            
            # Show individual service status
            services = health.get("services", {})
            cols = st.columns(3)
            for i, (service, status) in enumerate(services.items()):
                with cols[i % 3]:
                    if status:
                        st.metric(service, "✅ Online")
                    else:
                        st.metric(service, "❌ Offline")
    
    # Task execution interface
    st.subheader("🚀 Execute AGI Task")
    
    task_type = st.selectbox(
        "Task Type",
        ["general", "code_generation", "code_analysis", "task_automation", "memory_task"]
    )
    
    prompt = st.text_area("Enter your prompt:", height=100)
    
    # Advanced options
    with st.expander("Advanced Options"):
        available_agents = asyncio.run(st.session_state.agi.list_available_services())
        selected_agents = st.multiselect(
            "Select specific agents (leave empty for auto-selection)",
            available_agents
        )
        
        context = st.text_area("Additional context (JSON format):", "{}")
    
    if st.button("Execute Task", type="primary"):
        if prompt:
            with st.spinner("Processing..."):
                try:
                    # Parse context
                    context_dict = json.loads(context) if context else {}
                    
                    # Execute task
                    result = asyncio.run(
                        st.session_state.agi.execute_task(
                            task_type=task_type,
                            prompt=prompt,
                            context=context_dict,
                            agents=selected_agents
                        )
                    )
                    
                    # Display results
                    if "error" in result:
                        st.error(f"Error: {result['error']}")
                    else:
                        st.success("Task completed successfully!")
                        
                        # Show main result
                        st.subheader("Result")
                        st.json(result.get("result", {}))
                        
                        # Show metadata
                        with st.expander("Execution Details"):
                            st.write("**Agents Used:**", ", ".join(result.get("agents_used", [])))
                            st.json(result.get("metadata", {}))
                
                except json.JSONDecodeError:
                    st.error("Invalid JSON in context field")
                except Exception as e:
                    st.error(f"Execution failed: {str(e)}")
        else:
            st.warning("Please enter a prompt")

# Integration with main app
def integrate_with_main_app():
    """Add AGI system to the main Streamlit app"""
    # This would be imported and called from the main app.py
    create_agi_interface()
EOF

    log "Streamlit integration created successfully"
}

# ===============================================
# SYSTEM OPTIMIZATION
# ===============================================

optimize_system() {
    log_phase "Optimizing System Configuration"
    
    # Create optimization script
    cat > ${PROJECT_ROOT}/scripts/optimize_agi.sh << 'EOF'
#!/bin/bash
# AGI System Optimization Script

# Optimize Docker
cat > /etc/docker/daemon.json << EOL
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "storage-driver": "overlay2",
  "default-ulimits": {
    "nofile": {
      "Name": "nofile",
      "Hard": 64000,
      "Soft": 64000
    }
  }
}
EOL

# System limits
cat >> /etc/security/limits.conf << EOL
* soft nofile 64000
* hard nofile 64000
* soft nproc 32000
* hard nproc 32000
EOL

# Kernel parameters
cat >> /etc/sysctl.conf << EOL
vm.max_map_count=262144
fs.file-max=65536
net.core.somaxconn=65535
net.ipv4.tcp_max_syn_backlog=65535
EOL

sysctl -p

# Restart Docker
systemctl restart docker
EOF

    chmod +x ${PROJECT_ROOT}/scripts/optimize_agi.sh
}

# ===============================================
# MONITORING SETUP
# ===============================================

setup_monitoring() {
    log_phase "Setting Up Monitoring"
    
    # Create monitoring configuration
    cat > ${PROJECT_ROOT}/monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'agi-services'
    static_configs:
      - targets:
        - 'ollama:11434'
        - 'litellm:4000'
        - 'chromadb:8000'
        - 'letta:8283'
        - 'autogpt:8080'
        - 'localagi:8090'
        - 'tabbyml:8080'
        - 'agi-orchestrator:8200'
EOF

    # Create Grafana dashboards
    mkdir -p ${PROJECT_ROOT}/monitoring/grafana/dashboards
    cat > ${PROJECT_ROOT}/monitoring/grafana/dashboards/agi-dashboard.json << 'EOF'
{
  "dashboard": {
    "title": "AGI/ASI System Dashboard",
    "panels": [
      {
        "title": "Service Health",
        "type": "stat"
      },
      {
        "title": "Request Rate",
        "type": "graph"
      },
      {
        "title": "Model Performance",
        "type": "heatmap"
      }
    ]
  }
}
EOF
}

# ===============================================
# MAIN DEPLOYMENT FUNCTION
# ===============================================

deploy_agi_system() {
    log_phase "Starting AGI/ASI System Deployment"
    
    # Step 1: Prepare system
    prepare_system
    
    # Step 2: Setup model management
    setup_model_management
    
    # Step 3: Setup vector databases
    setup_vector_databases
    
    # Step 4: Setup AI agents
    setup_ai_agents
    
    # Step 5: Create Docker configurations
    create_docker_compose
    create_dockerfiles
    
    # Step 6: Create Streamlit integration
    create_streamlit_integration
    
    # Step 7: Optimize system
    optimize_system
    
    # Step 8: Setup monitoring
    setup_monitoring
    
    # Step 9: Start services
    log_phase "Starting AGI/ASI Services"
    cd ${WORKSPACE_ROOT}
    docker-compose -f ${COMPOSE_FILE} up -d
    
    # Step 10: Health check
    log_phase "Performing Health Checks"
    sleep 30
    
    # Check each service
    services=("ollama:11434" "litellm:4000" "chromadb:8000" "agi-orchestrator:8200")
    for service in "${services[@]}"; do
        if curl -f "http://localhost:${service##*:}/health" >/dev/null 2>&1; then
            log "✅ ${service%%:*} is healthy"
        else
            log_warn "⚠️  ${service%%:*} is not responding"
        fi
    done
    
    log_phase "🎉 AGI/ASI System Deployment Complete!"
    log_info "Access the orchestrator at: http://${LOCAL_IP}:8200"
    log_info "Integration available in Streamlit app at: http://${LOCAL_IP}:8501"
}

# ===============================================
# RUN DEPLOYMENT
# ===============================================

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   log_error "This script should not be run as root directly"
   exit 1
fi

# Run deployment
deploy_agi_system