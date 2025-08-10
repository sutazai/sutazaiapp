#!/usr/bin/env python3
"""
Fix agent Dockerfiles to ensure correct startup
"""
from pathlib import Path

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

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \\
    CMD curl -f http://localhost:8080/health || exit 1

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV AGENT_NAME={agent_name}

# Create non-root user
RUN groupadd -r agent && useradd -r -g agent agent
RUN chown -R agent:agent /app
USER agent

EXPOSE 8080

# Use main:app for startup
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
"""

def fix_dockerfile(agent_dir: Path, agent_name: str):
    """Fix Dockerfile for an agent"""
    dockerfile_path = agent_dir / 'Dockerfile'
    
    with open(dockerfile_path, 'w') as f:
        f.write(DOCKERFILE_TEMPLATE.format(agent_name=agent_name))
    
    print(f"✓ Fixed Dockerfile for {agent_name}")

def main():
    """Fix all Phase 1 agent Dockerfiles"""
    agents_dir = Path('/opt/sutazaiapp/agents')
    
    # Phase 1 agents
    phase1_agents = [
        'agent-orchestrator',
        'agentzero-coordinator', 
        'ai-agent-orchestrator',
        'ai-system-architect',
        'deployment-automation-master',
        'ai-senior-backend-developer',
        'ai-senior-frontend-developer',
        'ai-senior-full-stack-developer',
        'ai-senior-engineer',
        'ai-product-manager',
        'ai-scrum-master',
        'ai-system-validator',
        'ai-testing-qa-validator',
        'agent-creator',
        'agent-debugger',
        'ai-qa-team-lead',
        'adversarial-attack-detector',
        'emergency-shutdown-coordinator',
        'cicd-pipeline-orchestrator'
    ]
    
    print("Fixing Dockerfiles for Phase 1 agents...\n")
    
    fixed = 0
    for agent_name in phase1_agents:
        agent_dir = agents_dir / agent_name
        if agent_dir.exists():
            fix_dockerfile(agent_dir, agent_name)
            fixed += 1
        else:
            print(f"⚠ Agent directory not found: {agent_name}")
    
    print(f"\n✅ Fixed {fixed} Dockerfiles")
    print("\nNext: Rebuild containers with fixed Dockerfiles")

if __name__ == "__main__":
    main()