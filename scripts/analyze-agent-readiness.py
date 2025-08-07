#!/usr/bin/env python3
"""
Purpose: Analyze agent directories to check readiness for deployment
Usage: python analyze-agent-readiness.py
Requirements: Python 3.8+
"""

import os
import json
import sys
from pathlib import Path

AGENTS_DIR = Path("/opt/sutazaiapp/agents")
REQUIRED_FILES = {
    "main.py": ["uvicorn", "fastapi"],
    "app.py": ["fastapi"],
    "requirements.txt": None,
    "Dockerfile": None
}

CRITICAL_AGENTS = [
    "ai-system-architect",
    "ai-agent-orchestrator", 
    "infrastructure-devops-manager",
    "deployment-automation-master",
    "ai-system-validator",
    "ai-testing-qa-validator",
    "ai-senior-backend-developer",
    "ai-senior-frontend-developer",
    "ai-senior-full-stack-developer",
    "ai-senior-engineer",
    "agent-orchestrator",
    "agent-debugger",
    "agent-creator",
    "container-orchestrator-k3s",
    "hardware-resource-optimizer",
    "ollama-integration-specialist",
    "senior-ai-engineer",
    "testing-qa-validator",
    "kali-hacker",
    "mega-code-auditor"
]

def check_agent_directory(agent_path):
    """Check if an agent directory has all required files"""
    issues = []
    has_main = False
    has_app = False
    
    for file, required_imports in REQUIRED_FILES.items():
        file_path = agent_path / file
        if not file_path.exists():
            if file == "main.py":
                has_main = False
            elif file == "app.py":
                has_app = False
            
            # Only flag as issue if neither main.py nor app.py exists
            if file in ["main.py", "app.py"]:
                continue
            else:
                issues.append(f"Missing {file}")
        else:
            if file == "main.py":
                has_main = True
            elif file == "app.py":
                has_app = True
    
    # Check if at least one Python entry point exists
    if not has_main and not has_app:
        issues.append("Missing Python entry point (main.py or app.py)")
    
    # Check for universal_startup.py as alternative
    if (agent_path / "universal_startup.py").exists():
        # Remove entry point issue if universal_startup exists
        issues = [i for i in issues if "entry point" not in i]
    
    return issues

def analyze_agents():
    """Analyze all agent directories"""
    results = {
        "ready": [],
        "needs_fix": {},
        "critical_missing": []
    }
    
    # Get all agent directories
    agent_dirs = [d for d in AGENTS_DIR.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    for agent_dir in sorted(agent_dirs):
        agent_name = agent_dir.name
        
        # Skip non-agent directories
        if agent_name in ['configs', 'core', 'agi', 'dockerfiles']:
            continue
            
        issues = check_agent_directory(agent_dir)
        
        if not issues:
            results["ready"].append(agent_name)
        else:
            results["needs_fix"][agent_name] = issues
            if agent_name in CRITICAL_AGENTS:
                results["critical_missing"].append(agent_name)
    
    return results

def generate_report(results):
    """Generate a report of agent readiness"""
    print("=== SutazAI Agent Deployment Readiness Report ===\n")
    
    print(f"Total agents analyzed: {len(results['ready']) + len(results['needs_fix'])}")
    print(f"Ready for deployment: {len(results['ready'])}")
    print(f"Need fixes: {len(results['needs_fix'])}")
    print(f"Critical agents needing fixes: {len(results['critical_missing'])}\n")
    
    print("=== Critical Agents Needing Fixes ===")
    for agent in results['critical_missing']:
        print(f"- {agent}: {', '.join(results['needs_fix'][agent])}")
    
    print("\n=== All Agents Needing Fixes ===")
    for agent, issues in sorted(results['needs_fix'].items()):
        if agent not in results['critical_missing']:
            print(f"- {agent}: {', '.join(issues)}")
    
    print("\n=== Ready Agents ===")
    ready_critical = [a for a in results['ready'] if a in CRITICAL_AGENTS]
    print(f"Critical agents ready: {len(ready_critical)}")
    for agent in ready_critical:
        print(f"- {agent}")
    
    # Save detailed results
    with open("/opt/sutazaiapp/agent-readiness-report.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed report saved to: /opt/sutazaiapp/agent-readiness-report.json")

if __name__ == "__main__":
    results = analyze_agents()
    generate_report(results)