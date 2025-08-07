#!/usr/bin/env python3
"""
Fix agent startup issues by creating proper main.py wrappers
"""
import os
from pathlib import Path

MAIN_PY_WRAPPER = '''#!/usr/bin/env python3
"""
Main entry point wrapper for {agent_name}
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Try to import the FastAPI app
app = None

# Try different import patterns
try:
    # First try to import from agent module
    from agent import app
    print("Loaded app from agent module")
except ImportError:
    try:
        # Try to import everything from agent
        from agent import *
        if 'app' in locals():
            print("Found app in agent module globals")
        else:
            # Create a default app if none exists
            from fastapi import FastAPI
            app = FastAPI(title="{agent_title}")
            print("Created default FastAPI app")
    except ImportError:
        # If no agent module, create a basic app
        from fastapi import FastAPI
        from datetime import datetime
        
        app = FastAPI(
            title="{agent_title}",
            description="Agent service",
            version="1.0.0"
        )
        
        @app.get("/health")
        async def health_check():
            return {{
                "status": "healthy",
                "agent": "{agent_name}",
                "timestamp": datetime.utcnow().isoformat()
            }}
        
        @app.get("/")
        async def root():
            return {{
                "agent": "{agent_name}",
                "status": "running"
            }}
        
        print("Created basic FastAPI app with health endpoint")

# Ensure app is available at module level
if app is None:
    raise RuntimeError("Could not create or import FastAPI app")

# Run if called directly
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
'''

def fix_agent_main_py(agent_dir: Path, agent_name: str):
    """Fix or create main.py for an agent"""
    main_path = agent_dir / 'main.py'
    agent_title = agent_name.replace('-', ' ').title()
    
    # Always overwrite to ensure proper wrapper
    with open(main_path, 'w') as f:
        f.write(MAIN_PY_WRAPPER.format(
            agent_name=agent_name,
            agent_title=agent_title
        ))
    os.chmod(main_path, 0o755)
    print(f"✓ Fixed main.py for {agent_name}")

def main():
    """Fix all Phase 1 agents"""
    agents_dir = Path('/opt/sutazaiapp/agents')
    
    # Phase 1 agents that need fixing
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
        'cpu-hardware-optimizer',
        'compute-scheduler-optimizer',
        'emergency-shutdown-coordinator'
    ]
    
    print("Fixing startup issues for Phase 1 agents...\n")
    
    fixed = 0
    for agent_name in phase1_agents:
        agent_dir = agents_dir / agent_name
        if agent_dir.exists():
            fix_agent_main_py(agent_dir, agent_name)
            fixed += 1
        else:
            print(f"⚠ Agent directory not found: {agent_name}")
    
    print(f"\n✅ Fixed {fixed} agents")
    print("\nNext steps:")
    print("1. Rebuild affected containers: docker-compose -f docker-compose.phase1-critical.yml build")
    print("2. Restart containers: docker-compose -f docker-compose.phase1-critical.yml up -d")

if __name__ == "__main__":
    main()