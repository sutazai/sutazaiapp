#!/bin/bash

# Fix Docker build paths for services with mismatched Dockerfile locations

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}🔧 Fixing Docker build paths in docker-compose.yml${NC}"

# Create a backup
cp docker-compose.yml docker-compose.yml.backup.$(date +%Y%m%d_%H%M%S)

# Fix backend service (Dockerfile is missing, create it)
if [ ! -f "backend/Dockerfile" ]; then
    echo -e "${YELLOW}Creating backend/Dockerfile${NC}"
    cat > backend/Dockerfile << 'EOF'
FROM python:3.12.8-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

ENV PYTHONUNBUFFERED=1
ENV DEBUG=false

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health').raise_for_status()"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF
fi

# Fix frontend service (Dockerfile is missing, create it)
if [ ! -f "frontend/Dockerfile" ]; then
    echo -e "${YELLOW}Creating frontend/Dockerfile${NC}"
    cat > frontend/Dockerfile << 'EOF'
FROM python:3.12.8-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

ENV PYTHONUNBUFFERED=1

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/healthz || exit 1

CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
EOF
fi

# Fix services with wrong build contexts
echo -e "${CYAN}Fixing docker-compose.yml build paths...${NC}"

# Create a Python script to fix the paths
cat > /tmp/fix_compose.py << 'EOF'
import yaml
import sys

with open('docker-compose.yml', 'r') as f:
    data = yaml.safe_load(f)

services_to_fix = {
    'llamaindex': './docker/llamaindex',
    'letta': './docker/letta',
    'autogen': './docker/autogen',
    'privategpt': './docker/privategpt',
    'aider': './docker/aider',
    'agentgpt': './docker/agentgpt',
    'agentzero': './docker/agentzero',
    'autogpt': './docker/autogpt',
    'browser-use': './docker/browser-use',
    'crewai': './docker/crewai',
    'documind': './docker/documind',
    'finrobot': './docker/finrobot',
    'gpt-engineer': './docker/gpt-engineer',
    'opendevin': './docker/opendevin',
    'pentestgpt': './docker/pentestgpt',
    'shellgpt': './docker/shellgpt',
    'skyvern': './docker/skyvern',
    'awesome-code-ai': './docker/awesome-code-ai',
    'code-improver': './docker/code-improver',
    'context-framework': './docker/context-framework',
    'faiss': './docker/faiss',
    'fsdp': './docker/fsdp',
    'health-monitor': './docker/health-monitor',
    'jax': './docker/jax',
    'pytorch': './docker/pytorch',
    'service-hub': './docker/service-hub',
    'tensorflow': './docker/tensorflow',
    'mcp-server': './mcp_server',
    'ai-metrics-exporter': './docker/ai-metrics-exporter'
}

for service, correct_context in services_to_fix.items():
    if service in data['services']:
        if 'build' in data['services'][service]:
            if isinstance(data['services'][service]['build'], dict):
                data['services'][service]['build']['context'] = correct_context
            else:
                # Convert string build to dict format
                data['services'][service]['build'] = {
                    'context': correct_context,
                    'dockerfile': 'Dockerfile'
                }

with open('docker-compose.yml', 'w') as f:
    yaml.dump(data, f, default_flow_style=False, sort_keys=False)

print("Fixed docker-compose.yml build paths")
EOF

# Run the Python script to fix paths
python3 /tmp/fix_compose.py

# Create missing ai-metrics-exporter Dockerfile
if [ ! -d "docker/ai-metrics-exporter" ]; then
    echo -e "${YELLOW}Creating docker/ai-metrics-exporter directory and Dockerfile${NC}"
    mkdir -p docker/ai-metrics-exporter
    cat > docker/ai-metrics-exporter/Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir prometheus-client psutil

# Create metrics exporter script
RUN cat > ai_metrics_exporter.py << 'SCRIPT'
from prometheus_client import start_http_server, Gauge
import psutil
import time

# Create metrics
cpu_usage = Gauge('ai_system_cpu_usage', 'CPU usage percentage')
memory_usage = Gauge('ai_system_memory_usage', 'Memory usage percentage')
disk_usage = Gauge('ai_system_disk_usage', 'Disk usage percentage')

def collect_metrics():
    while True:
        cpu_usage.set(psutil.cpu_percent())
        memory_usage.set(psutil.virtual_memory().percent)
        disk_usage.set(psutil.disk_usage('/').percent)
        time.sleep(15)

if __name__ == '__main__':
    start_http_server(9100)
    collect_metrics()
SCRIPT

EXPOSE 9100

CMD ["python", "ai_metrics_exporter.py"]
EOF
fi

echo -e "${GREEN}✅ Docker paths fixed!${NC}"
echo -e "${CYAN}You can now deploy services properly.${NC}"