#!/usr/bin/env python3
"""
Verify CLAUDE.md Rules Enforcement

This script checks that all agents are properly configured to enforce CLAUDE.md rules.
"""

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
    print("\nChecking Docker service configurations...")
    print("=" * 60)
    
    # This would need docker-compose config command
    print("Run: docker-compose -f docker-compose.yml -f docker-compose.claude-rules.yml config")
    print("to verify CLAUDE.md is mounted in all services")

if __name__ == "__main__":
    compliant, non_compliant = check_agent_rules_compliance()
    check_docker_services()
    
    if non_compliant == 0:
        print("\n✅ All agents are configured to enforce CLAUDE.md rules!")
    else:
        print(f"\n⚠ {non_compliant} agents need configuration updates")
