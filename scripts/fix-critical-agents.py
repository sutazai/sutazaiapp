#!/usr/bin/env python3
"""
Purpose: Fix missing files for critical agents
Usage: python fix-critical-agents.py
Requirements: Python 3.8+
"""

import os
import shutil
from pathlib import Path

AGENTS_DIR = Path("/opt/sutazaiapp/agents")

# Generic Dockerfile template
DOCKERFILE_TEMPLATE = """FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Install base Python packages
RUN pip install --no-cache-dir \\
    fastapi==0.104.1 \\
    uvicorn==0.24.0 \\
    pydantic==2.5.0 \\
    httpx==0.25.2 \\
    python-dotenv==1.0.0 \\
    redis==5.0.1 \\
    prometheus-client==0.19.0 \\
    psutil==5.9.6 \\
    structlog==23.2.0

# Copy agent-specific requirements if they exist
COPY requirements.txt* ./
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Copy application code
COPY . .

# Create non-root user
RUN groupadd -r agent && useradd -r -g agent agent
RUN chown -R agent:agent /app
USER agent

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \\
    CMD curl -f http://localhost:8080/health || exit 1

# Set agent-specific environment
ENV PYTHONUNBUFFERED=1
ENV AGENT_NAME={agent_name}

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
"""

# Generic main.py template
MAIN_PY_TEMPLATE = '''"""
{agent_name} Agent
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import logging
import httpx
from datetime import datetime

# Configure logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = FastAPI(title="{agent_title}")

# Agent configuration
AGENT_NAME = os.getenv("AGENT_NAME", "{agent_name}")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

class TaskRequest(BaseModel):
    task: str
    context: dict = {{}}

class TaskResponse(BaseModel):
    result: str
    agent: str
    timestamp: str
    status: str = "success"

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {{
        "status": "healthy",
        "agent": AGENT_NAME,
        "timestamp": datetime.utcnow().isoformat()
    }}

@app.post("/task", response_model=TaskResponse)
async def process_task(request: TaskRequest):
    """Process a task"""
    try:
        logger.info(f"Processing task: {{request.task}}")
        
        # TODO: Implement agent-specific logic here
        result = f"{{AGENT_NAME}} processed: {{request.task}}"
        
        return TaskResponse(
            result=result,
            agent=AGENT_NAME,
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        logger.error(f"Error processing task: {{str(e)}}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/info")
async def info():
    """Get agent information"""
    return {{
        "name": AGENT_NAME,
        "type": "{agent_type}",
        "version": "1.0.0",
        "capabilities": ["{capability}"]
    }}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
'''

# Agent-specific configurations
AGENT_CONFIGS = {
    "ai-agent-orchestrator": {
        "type": "orchestration",
        "capability": "Agent coordination and task distribution"
    },
    "infrastructure-devops-manager": {
        "type": "infrastructure",
        "capability": "Infrastructure management and DevOps automation"
    },
    "deployment-automation-master": {
        "type": "deployment",
        "capability": "Automated deployment and rollout management"
    },
    "ollama-integration-specialist": {
        "type": "integration",
        "capability": "Ollama LLM integration and optimization"
    },
    "senior-ai-engineer": {
        "type": "engineering",
        "capability": "AI system design and implementation"
    },
    "testing-qa-validator": {
        "type": "testing",
        "capability": "Automated testing and quality validation"
    }
}

def fix_agent(agent_name, agent_path):
    """Fix missing files for an agent"""
    fixes_applied = []
    
    # Create requirements.txt if missing
    req_path = agent_path / "requirements.txt"
    if not req_path.exists():
        req_path.write_text("# Agent-specific requirements\n# Base requirements are included from base image\n")
        fixes_applied.append("Created requirements.txt")
    
    # Create Dockerfile if missing
    dockerfile_path = agent_path / "Dockerfile"
    if not dockerfile_path.exists():
        dockerfile_content = DOCKERFILE_TEMPLATE.format(agent_name=agent_name)
        dockerfile_path.write_text(dockerfile_content)
        fixes_applied.append("Created Dockerfile")
    
    # Create main.py if missing and no other entry point exists
    main_path = agent_path / "main.py"
    app_path = agent_path / "app.py"
    universal_path = agent_path / "universal_startup.py"
    
    if not main_path.exists() and not app_path.exists() and not universal_path.exists():
        config = AGENT_CONFIGS.get(agent_name, {
            "type": "general",
            "capability": "General agent capabilities"
        })
        
        main_content = MAIN_PY_TEMPLATE.format(
            agent_name=agent_name,
            agent_title=agent_name.replace("-", " ").title(),
            agent_type=config["type"],
            capability=config["capability"]
        )
        main_path.write_text(main_content)
        fixes_applied.append("Created main.py")
    
    return fixes_applied

def main():
    """Fix critical agents"""
    # Critical agents that need fixes
    critical_agents = [
        "ai-agent-orchestrator",
        "infrastructure-devops-manager", 
        "deployment-automation-master",
        "ollama-integration-specialist",
        "senior-ai-engineer",
        "testing-qa-validator",
        "agent-creator",
        "agent-debugger",
        "agent-orchestrator",
        "ai-senior-backend-developer",
        "ai-senior-engineer",
        "ai-senior-frontend-developer",
        "ai-senior-full-stack-developer",
        "ai-system-validator",
        "ai-testing-qa-validator",
        "container-orchestrator-k3s",
        "kali-hacker",
        "mega-code-auditor"
    ]
    
    print("=== Fixing Critical Agents ===\n")
    
    fixed_count = 0
    for agent_name in critical_agents:
        agent_path = AGENTS_DIR / agent_name
        if agent_path.exists():
            fixes = fix_agent(agent_name, agent_path)
            if fixes:
                print(f"✓ {agent_name}: {', '.join(fixes)}")
                fixed_count += 1
            else:
                print(f"• {agent_name}: No fixes needed")
        else:
            print(f"✗ {agent_name}: Directory not found")
    
    print(f"\nFixed {fixed_count} agents")
    
    # Also fix code-generation-improver since it's missing files
    improver_path = AGENTS_DIR / "code-generation-improver"
    if improver_path.exists():
        fixes = fix_agent("code-generation-improver", improver_path)
        if fixes:
            print(f"✓ code-generation-improver: {', '.join(fixes)}")

if __name__ == "__main__":
    main()