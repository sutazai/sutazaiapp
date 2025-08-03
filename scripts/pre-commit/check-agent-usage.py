#!/usr/bin/env python3
"""
Purpose: Verify correct AI agent selection and usage (Rule 14 enforcement)
Usage: python check-agent-usage.py <file1> <file2> ...
Requirements: Python 3.8+
"""

import sys
import re
from pathlib import Path
from typing import List, Dict, Set

# Agent specializations mapping
AGENT_SPECIALIZATIONS = {
    'frontend': ['senior-frontend-developer', 'ui-ux-designer'],
    'backend': ['senior-backend-developer', 'api-architect'],
    'infrastructure': ['infrastructure-devops-manager', 'container-orchestrator-k3s'],
    'testing': ['testing-qa-validator', 'security-pentesting-specialist'],
    'documentation': ['document-knowledge-manager', 'technical-writer'],
    'deployment': ['deployment-automation-master', 'ci-cd-specialist'],
    'security': ['security-pentesting-specialist', 'kali-security-specialist'],
    'data': ['private-data-analyst', 'data-pipeline-engineer'],
    'ai': ['senior-ai-engineer', 'model-training-specialist']
}

def check_agent_usage(filepath: Path) -> List[str]:
    """Check if correct agents are being used for tasks."""
    violations = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for agent references
        agent_refs = re.findall(r'agent[_-]?name["\']?\s*[:=]\s*["\']([^"\']+)["\']', content, re.IGNORECASE)
        
        # Check for task descriptions
        task_keywords = {
            'frontend': ['react', 'vue', 'angular', 'css', 'ui', 'component'],
            'backend': ['api', 'database', 'endpoint', 'server', 'model'],
            'testing': ['test', 'qa', 'validation', 'coverage'],
            'security': ['security', 'vulnerability', 'pentest', 'audit']
        }
        
        # Determine task type from content
        detected_tasks = set()
        for task_type, keywords in task_keywords.items():
            if any(keyword in content.lower() for keyword in keywords):
                detected_tasks.add(task_type)
        
        # Check if appropriate agents are used
        for task_type in detected_tasks:
            appropriate_agents = AGENT_SPECIALIZATIONS.get(task_type, [])
            if agent_refs and not any(agent in appropriate_agents for agent in agent_refs):
                violations.append(
                    f"{filepath}: {task_type} task should use specialized agents: {', '.join(appropriate_agents)}"
                )
                
    except:
        pass
    
    return violations

def main():
    if len(sys.argv) < 2:
        print("No files to check")
        return 0
    
    all_violations = []
    
    for filepath_str in sys.argv[1:]:
        filepath = Path(filepath_str)
        if filepath.suffix == '.py':
            violations = check_agent_usage(filepath)
            all_violations.extend(violations)
    
    if all_violations:
        print("❌ Rule 14 Violation: Incorrect agent usage detected")
        for violation in all_violations:
            print(f"  - {violation}")
        return 1
    
    print("✅ Rule 14: Agent usage check passed")
    return 0

if __name__ == "__main__":
    sys.exit(main())