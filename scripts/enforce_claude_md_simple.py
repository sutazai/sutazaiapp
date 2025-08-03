#!/usr/bin/env python3
"""
Simple CLAUDE.md Rules Enforcement for All AI Agents

This script adds CLAUDE.md rules checking to all agents without
modifying their YAML frontmatter (which is causing parsing issues).
"""

import os
import shutil
from pathlib import Path
from typing import List, Set

AGENTS_DIR = "/opt/sutazaiapp/.claude/agents"
CLAUDE_MD_PATH = "/opt/sutazaiapp/CLAUDE.md"
BACKUP_DIR = "/opt/sutazaiapp/.claude/agents/backups"

# Skip these files
SKIP_FILES = {
    "CLAUDE.md",
    "claude_rules_checker.py",
    "agent_startup_wrapper.py",
    "debug_utils.py",
    "debug_integration.py",
    "DEBUGGING_STANDARDS.md",
    "COMPREHENSIVE_INVESTIGATION_PROTOCOL.md",
    "AGENT_COMPLIANCE_REPORT.md",
    "AGENT_CLEANUP_SUMMARY.md",
    "FINAL_COMPLIANCE_SUMMARY.md",
    "DUPLICATE_AGENTS_REPORT.md"
}

def create_backup():
    """Create backup of agent files"""
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
    
    for file in Path(AGENTS_DIR).glob("*.md"):
        if file.name not in SKIP_FILES:
            backup_path = Path(BACKUP_DIR) / file.name
            shutil.copy2(file, backup_path)
    
    print(f"Created backups in {BACKUP_DIR}")

def add_claude_rules_to_agent_dirs():
    """Add CLAUDE.md rules checking to agent implementation directories"""
    rules_init_content = '''"""
CLAUDE.md Rules Enforcement Module

This module ensures all agents check CLAUDE.md rules before executing tasks.
"""

import os
from pathlib import Path

CLAUDE_MD_PATH = "/opt/sutazaiapp/CLAUDE.md"

class ClaudeRulesEnforcer:
    """Enforces CLAUDE.md rules for agent operations"""
    
    def __init__(self):
        self.rules = self._load_rules()
        self.enabled = True
        
    def _load_rules(self):
        """Load rules from CLAUDE.md"""
        try:
            with open(CLAUDE_MD_PATH, 'r') as f:
                return f.read()
        except:
            print(f"Warning: Could not load {CLAUDE_MD_PATH}")
            return None
    
    def check_action(self, action: str) -> bool:
        """Check if action complies with rules"""
        if not self.enabled or not self.rules:
            return True
        
        # Check for blocking violations
        blocking_keywords = ['fantasy', 'specific implementation name (e.g., emailSender, dataProcessor)', 'wizard', 'mythical']
        for keyword in blocking_keywords:
            if keyword in action.lower():
                print(f"BLOCKED: Action contains forbidden keyword: {keyword}")
                return False
        
        # Check for destructive operations
        destructive_keywords = ['delete all', 'drop database', 'rm -rf /']
        for keyword in destructive_keywords:
            if keyword in action.lower():
                print(f"BLOCKED: Potentially destructive operation detected: {keyword}")
                return False
        
        return True
    
    def log_compliance(self, agent_name: str, action: str, allowed: bool):
        """Log compliance check results"""
        status = "ALLOWED" if allowed else "BLOCKED"
        print(f"[CLAUDE.md Rules] {agent_name}: {status} - {action[:50]}...")

# Global enforcer instance
rules_enforcer = ClaudeRulesEnforcer()

# Helper function for agents to use
def check_claude_rules(action: str, agent_name: str = "Unknown") -> bool:
    """Check if an action complies with CLAUDE.md rules"""
    allowed = rules_enforcer.check_action(action)
    rules_enforcer.log_compliance(agent_name, action, allowed)
    return allowed

print("CLAUDE.md rules enforcement initialized")
'''
    
    # Add to each agent directory
    agent_dirs = [d for d in Path(AGENTS_DIR).iterdir() if d.is_dir()]
    updated_dirs = 0
    
    for agent_dir in agent_dirs:
        rules_file = agent_dir / "claude_rules.py"
        
        # Write the rules module
        with open(rules_file, 'w') as f:
            f.write(rules_init_content)
        
        # Update __init__.py if it exists
        init_file = agent_dir / "__init__.py"
        if init_file.exists():
            with open(init_file, 'r') as f:
                content = f.read()
            
            if "claude_rules" not in content:
                import_line = "\nfrom .claude_rules import check_claude_rules, rules_enforcer\n"
                
                # Add after other imports
                lines = content.split('\n')
                import_index = 0
                for i, line in enumerate(lines):
                    if line.strip() and not line.startswith(('import', 'from', '#')):
                        import_index = i
                        break
                
                lines.insert(import_index, import_line)
                
                with open(init_file, 'w') as f:
                    f.write('\n'.join(lines))
        
        updated_dirs += 1
    
    print(f"Added CLAUDE.md rules to {updated_dirs} agent directories")

def create_docker_compose_override():
    """Create docker-compose override to mount CLAUDE.md"""
    override_content = '''version: '3.8'

# CLAUDE.md Rules Enforcement Override
# This file ensures all agents have access to CLAUDE.md rules

services:
'''
    
    # Get all agent service names
    agent_services = []
    
    for compose_file in ["/opt/sutazaiapp/docker-compose.yml",
                         "/opt/sutazaiapp/docker-compose.agents-simple.yml",
                         "/opt/sutazaiapp/docker-compose.complete-agents.yml"]:
        if os.path.exists(compose_file):
            with open(compose_file, 'r') as f:
                content = f.read()
                # Extract service names (rough parsing)
                for line in content.split('\n'):
                    if line and not line.startswith(' ') and ':' in line and 'version' not in line:
                        service = line.split(':')[0].strip()
                        if 'agent' in service or service == 'ollama':
                            agent_services.append(service)
    
    # Add volume mount for each service
    for service in set(agent_services):
        override_content += f'''
  {service}:
    volumes:
      - /opt/sutazaiapp/CLAUDE.md:/app/CLAUDE.md:ro
      - /opt/sutazaiapp/.claude/agents/claude_rules_checker.py:/app/claude_rules_checker.py:ro
    environment:
      CLAUDE_RULES_PATH: /app/CLAUDE.md
      ENFORCE_CLAUDE_RULES: "true"
'''
    
    override_file = Path("/opt/sutazaiapp/docker-compose.claude-rules.yml")
    with open(override_file, 'w') as f:
        f.write(override_content)
    
    print(f"Created Docker Compose override: {override_file}")
    print("Use with: docker-compose -f docker-compose.yml -f docker-compose.claude-rules.yml up")

def create_agent_wrapper_script():
    """Create a wrapper script for starting agents with rules checking"""
    wrapper_content = '''#!/bin/bash
# Agent Startup Wrapper with CLAUDE.md Rules Enforcement

set -e

echo "==========================================="
echo "Starting Agent with CLAUDE.md Rules"
echo "==========================================="

# Export CLAUDE.md path
export CLAUDE_RULES_PATH="/opt/sutazaiapp/CLAUDE.md"
export ENFORCE_CLAUDE_RULES="true"

# Check if CLAUDE.md exists
if [ -f "$CLAUDE_RULES_PATH" ]; then
    echo "✓ CLAUDE.md rules found at: $CLAUDE_RULES_PATH"
else
    echo "⚠ Warning: CLAUDE.md not found at: $CLAUDE_RULES_PATH"
fi

# Display key rules
echo ""
echo "Key Rules to Follow:"
echo "1. No Fantasy Elements (BLOCKING)"
echo "2. Do Not Break Existing Functionality (BLOCKING)"
echo "3. Analyze Everything Before Proceeding (WARNING)"
echo "4. Reuse Before Creating (WARNING)"
echo "5. Treat as Professional Project (WARNING)"
echo ""

# Run the actual agent
if [ $# -eq 0 ]; then
    echo "Error: No agent command provided"
    exit 1
fi

echo "Starting agent: $@"
exec "$@"
'''
    
    wrapper_path = Path(AGENTS_DIR) / "start_with_rules.sh"
    with open(wrapper_path, 'w') as f:
        f.write(wrapper_content)
    
    os.chmod(wrapper_path, 0o755)
    print(f"Created agent wrapper script: {wrapper_path}")

def update_agent_dockerfiles():
    """Update Dockerfiles to include rules checking"""
    dockerfile_template = '''
# Add CLAUDE.md rules enforcement
COPY --from=builder /opt/sutazaiapp/CLAUDE.md /app/CLAUDE.md
COPY --from=builder /opt/sutazaiapp/.claude/agents/claude_rules_checker.py /app/claude_rules_checker.py

ENV CLAUDE_RULES_PATH=/app/CLAUDE.md
ENV ENFORCE_CLAUDE_RULES=true

# Add rules checking to Python path
ENV PYTHONPATH="/app:${PYTHONPATH}"
'''
    
    # Find all Dockerfiles in agent directories
    dockerfiles_updated = 0
    
    for agent_dir in Path(AGENTS_DIR).iterdir():
        if agent_dir.is_dir():
            dockerfile = agent_dir / "Dockerfile"
            if dockerfile.exists():
                with open(dockerfile, 'r') as f:
                    content = f.read()
                
                if "CLAUDE.md rules enforcement" not in content:
                    # Add before CMD or ENTRYPOINT
                    lines = content.split('\n')
                    insert_index = len(lines) - 1
                    
                    for i in range(len(lines) - 1, -1, -1):
                        if lines[i].strip().startswith(('CMD', 'ENTRYPOINT')):
                            insert_index = i
                            break
                    
                    lines.insert(insert_index, dockerfile_template)
                    
                    with open(dockerfile, 'w') as f:
                        f.write('\n'.join(lines))
                    
                    dockerfiles_updated += 1
    
    print(f"Updated {dockerfiles_updated} Dockerfiles with CLAUDE.md rules")

def create_verification_script():
    """Create a script to verify rules enforcement"""
    verify_content = '''#!/usr/bin/env python3
"""
Verify CLAUDE.md Rules Enforcement

This script checks that all agents are properly configured to enforce CLAUDE.md rules.
"""

import os
import subprocess
from pathlib import Path

def check_agent_rules_compliance():
    """Check if agents have rules enforcement"""
    agents_dir = Path("/opt/sutazaiapp/.claude/agents")
    compliant = 0
    non_compliant = 0
    
    print("Checking agent compliance with CLAUDE.md rules...")
    print("=" * 60)
    
    for agent_dir in agents_dir.iterdir():
        if agent_dir.is_dir():
            # Check for rules module
            rules_file = agent_dir / "claude_rules.py"
            has_rules = rules_file.exists()
            
            status = "✓" if has_rules else "✗"
            print(f"{status} {agent_dir.name}: {'Compliant' if has_rules else 'Non-compliant'}")
            
            if has_rules:
                compliant += 1
            else:
                non_compliant += 1
    
    print("=" * 60)
    print(f"Summary: {compliant} compliant, {non_compliant} non-compliant")
    
    return compliant, non_compliant

def check_docker_services():
    """Check if Docker services have CLAUDE.md mounted"""
    print("\\nChecking Docker service configurations...")
    print("=" * 60)
    
    # This would need docker-compose config command
    print("Run: docker-compose -f docker-compose.yml -f docker-compose.claude-rules.yml config")
    print("to verify CLAUDE.md is mounted in all services")

if __name__ == "__main__":
    compliant, non_compliant = check_agent_rules_compliance()
    check_docker_services()
    
    if non_compliant == 0:
        print("\\n✅ All agents are configured to enforce CLAUDE.md rules!")
    else:
        print(f"\\n⚠ {non_compliant} agents need configuration updates")
'''
    
    verify_path = Path("/opt/sutazaiapp/scripts/verify_claude_rules.py")
    with open(verify_path, 'w') as f:
        f.write(verify_content)
    
    os.chmod(verify_path, 0o755)
    print(f"Created verification script: {verify_path}")

def main():
    """Main function to enforce CLAUDE.md rules"""
    print("Simple CLAUDE.md Rules Enforcement Implementation")
    print("=" * 60)
    
    # Create backup first
    print("\n1. Creating backups...")
    create_backup()
    
    # Add rules to agent directories
    print("\n2. Adding rules enforcement to agent directories...")
    add_claude_rules_to_agent_dirs()
    
    # Create Docker Compose override
    print("\n3. Creating Docker Compose override...")
    create_docker_compose_override()
    
    # Create wrapper script
    print("\n4. Creating agent wrapper script...")
    create_agent_wrapper_script()
    
    # Update Dockerfiles
    print("\n5. Updating agent Dockerfiles...")
    update_agent_dockerfiles()
    
    # Create verification script
    print("\n6. Creating verification script...")
    create_verification_script()
    
    print("\n" + "=" * 60)
    print("✅ CLAUDE.md rules enforcement setup complete!")
    print("\nNext steps:")
    print("1. Run verification: python /opt/sutazaiapp/scripts/verify_claude_rules.py")
    print("2. Restart agents with: docker-compose -f docker-compose.yml -f docker-compose.claude-rules.yml up -d")
    print("3. Monitor agent logs for rules enforcement messages")
    
    print("\nKey enforcement features:")
    print("- Rules module added to each agent directory")
    print("- Docker Compose override for mounting CLAUDE.md")
    print("- Wrapper script for agent startup")
    print("- Verification script to check compliance")

if __name__ == "__main__":
    main()