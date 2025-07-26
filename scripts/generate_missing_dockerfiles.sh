#!/bin/bash
# Generate missing Dockerfiles for all AI agents

DOCKER_DIR="/opt/sutazaiapp/docker"
mkdir -p "$DOCKER_DIR"

echo "=== Generating Missing Dockerfiles for AI Agents ==="

# Function to create agent Dockerfile
create_agent_dockerfile() {
    local agent_name=$1
    local base_image=$2
    local pip_packages=$3
    local npm_packages=$4
    local expose_port=${5:-8080}
    
    local dir="$DOCKER_DIR/$agent_name"
    mkdir -p "$dir"
    
    if [ -f "$dir/Dockerfile" ]; then
        echo "✓ Dockerfile already exists for $agent_name"
        return
    fi
    
    echo "Creating Dockerfile for $agent_name..."
    
    cat > "$dir/Dockerfile" << DOCKERFILE
FROM $base_image

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    git \\
    build-essential \\
    python3-dev \\
    wget \\
    && rm -rf /var/lib/apt/lists/*

DOCKERFILE

    # Add Python dependencies if specified
    if [ -n "$pip_packages" ]; then
        cat >> "$dir/Dockerfile" << DOCKERFILE

# Install Python dependencies
RUN pip install --no-cache-dir $pip_packages

DOCKERFILE
    fi
    
    # Add Node dependencies if specified
    if [ -n "$npm_packages" ]; then
        cat >> "$dir/Dockerfile" << DOCKERFILE

# Install Node dependencies
RUN npm install -g $npm_packages

DOCKERFILE
    fi
    
    # Add common agent setup
    cat >> "$dir/Dockerfile" << DOCKERFILE

# Create startup script
COPY main.py requirements.txt* ./
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Add health check endpoint
RUN echo '#!/usr/bin/env python3
import http.server
import socketserver
import threading
import json
import os

class HealthHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "healthy", "agent": "$agent_name"}).encode())
        else:
            self.send_response(404)
            self.end_headers()

def start_health_server():
    with socketserver.TCPServer(("", 8000), HealthHandler) as httpd:
        httpd.serve_forever()

if __name__ == "__main__":
    # Start health check server in background
    health_thread = threading.Thread(target=start_health_server)
    health_thread.daemon = True
    health_thread.start()
    
    # Import and run main application
    import main
    main.run()
' > /app/web_interface.py

EXPOSE $expose_port
EXPOSE 8000

CMD ["python3", "web_interface.py"]
DOCKERFILE
    
    # Create basic main.py if it doesn't exist
    if [ ! -f "$dir/main.py" ]; then
        cat > "$dir/main.py" << MAINPY
#!/usr/bin/env python3
"""
$agent_name Agent Implementation
"""
import os
import json
import asyncio
import aiohttp
from typing import Dict, Any, Optional

class ${agent_name^}Agent:
    def __init__(self):
        self.name = "$agent_name"
        self.ollama_base = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        self.model = os.getenv("DEFAULT_MODEL", "deepseek-r1:8b")
        
    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming request"""
        # Implement agent-specific logic here
        prompt = request.get("prompt", "")
        
        # Call Ollama for LLM processing
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.ollama_base}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                }
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "agent": self.name,
                        "response": result.get("response", ""),
                        "status": "success"
                    }
                else:
                    return {
                        "agent": self.name,
                        "error": f"Ollama request failed: {response.status}",
                        "status": "error"
                    }

def run():
    """Run the agent"""
    print(f"Starting $agent_name agent...")
    agent = ${agent_name^}Agent()
    
    # In production, this would start a web server
    # For now, just indicate the agent is ready
    print(f"$agent_name agent ready on port $expose_port")
    
    # Keep the agent running
    try:
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        print(f"Shutting down $agent_name agent...")

if __name__ == "__main__":
    run()
MAINPY
    fi
    
    # Create requirements.txt if it doesn't exist
    if [ ! -f "$dir/requirements.txt" ]; then
        cat > "$dir/requirements.txt" << REQUIREMENTS
aiohttp==3.9.1
asyncio==3.4.3
python-dotenv==1.0.0
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
REQUIREMENTS
    fi
    
    echo "✓ Created Dockerfile for $agent_name"
}

# Create Dockerfiles for each missing agent
# Format: "agent_name:base_image:pip_packages:npm_packages:port"
agents=(
    "localagi:python:3.11-slim:langchain langchain-community ollama::8090"
    "tabbyml:python:3.11-slim:tabby-client fastapi::8093"
    "semgrep:python:3.11-slim:semgrep::8080"
    "autogen:python:3.11-slim:pyautogen::8104"
    "agentzero:python:3.11-slim:openai requests::8105"
    "bigagi:node:20-slim::express cors dotenv:3000"
    "browser-use:python:3.11-slim:playwright browser-use::8094"
    "skyvern:python:3.11-slim:selenium webdriver-manager::8080"
    "awesome-code-ai:python:3.11-slim:openai fastapi::8112"
    "agentgpt:node:20-slim::next react prisma:3000"
    "pentestgpt:python:3.11-slim:requests beautifulsoup4::8080"
    "finrobot:python:3.11-slim:pandas numpy yfinance::8109"
    "realtimestt:python:3.11-slim:speechrecognition pyaudio::8110"
    "opendevin:python:3.11-slim:fastapi websockets::3000"
    "documind:python:3.11-slim:pypdf2 python-docx pillow::8103"
    "code-improver:python:3.11-slim:gitpython ast-comments::8113"
    "service-hub:python:3.11-slim:fastapi redis::8114"
    "context-framework:python:3.11-slim:transformers torch::8111"
    "litellm:python:3.11-slim:litellm::4000"
    "health-monitor:python:3.11-slim:docker psutil::8100"
)

# Process each agent
for agent_spec in "${agents[@]}"; do
    IFS=':' read -r agent base pip npm port <<< "$agent_spec"
    create_agent_dockerfile "$agent" "$base" "$pip" "$npm" "$port"
done

# Create PyTorch Dockerfile
echo "Creating PyTorch Dockerfile..."
mkdir -p "$DOCKER_DIR/pytorch"
cat > "$DOCKER_DIR/pytorch/Dockerfile" << 'DOCKERFILE'
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    matplotlib \
    seaborn \
    pandas \
    scikit-learn

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
DOCKERFILE

# Create TensorFlow Dockerfile
echo "Creating TensorFlow Dockerfile..."
mkdir -p "$DOCKER_DIR/tensorflow"
cat > "$DOCKER_DIR/tensorflow/Dockerfile" << 'DOCKERFILE'
FROM tensorflow/tensorflow:2.14.0-gpu

WORKDIR /workspace

RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    matplotlib \
    seaborn \
    pandas \
    scikit-learn

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
DOCKERFILE

# Create JAX Dockerfile
echo "Creating JAX Dockerfile..."
mkdir -p "$DOCKER_DIR/jax"
cat > "$DOCKER_DIR/jax/Dockerfile" << 'DOCKERFILE'
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    jax[cpu] \
    flax \
    optax \
    fastapi \
    uvicorn

COPY web_interface.py .

EXPOSE 8080

CMD ["python", "web_interface.py"]
DOCKERFILE

echo ""
echo "=== Dockerfile Generation Complete ==="
echo "Generated Dockerfiles in: $DOCKER_DIR"
echo ""
echo "Next steps:"
echo "1. Review generated Dockerfiles"
echo "2. Add agent-specific implementations"
echo "3. Build images with: docker-compose build"