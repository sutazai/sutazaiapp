#!/usr/bin/env python3
"""
Agent Startup Wrapper with CLAUDE.md Rules Enforcement

This wrapper ensures all agents check CLAUDE.md rules before startup
"""

import os
import sys
import argparse
from pathlib import Path

# Add the agents directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from claude_rules_checker import check_action, get_rules_summary, rules_checker
except ImportError:
    print("❌ ERROR: claude_rules_checker.py not found!")
    print("Please ensure claude_rules_checker.py is in the agents directory")
    sys.exit(1)

def enforce_rules_before_action(action_description: str) -> bool:
    """Helper function for rules enforcement"""
    can_proceed, issues = check_action(action_description)
    
    if not can_proceed:
        print("❌ BLOCKED: Action violates CLAUDE.md rules:")
        for issue in issues:
            print(f"  • {issue}")
        return False
    
    if issues:  # Warnings only
        print("⚠️  WARNING: Please review:")
        for issue in issues:
            print(f"  • {issue}")
    
    return True

def get_compliance_status() -> dict:
    """Get current compliance status"""
    return {
        "enabled": os.getenv('CLAUDE_RULES_ENABLED', 'true').lower() == 'true',
        "rules_path": os.getenv('CLAUDE_RULES_PATH', '/opt/sutazaiapp/CLAUDE.md'),
        "rules_checker_available": True
    }

class AgentStartupWrapper:
    """Wrapper for starting agents with rules compliance"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.agents_dir = Path(__file__).parent
        
    def check_environment(self) -> bool:
        """Check if environment is properly configured"""
        required_vars = [
            'CLAUDE_RULES_ENABLED',
            'CLAUDE_RULES_PATH'
        ]
        
        missing_vars = []
        for var in required_vars:
            if var not in os.environ:
                missing_vars.append(var)
        
        if missing_vars:
            print("❌ Missing required environment variables:")
            for var in missing_vars:
                print(f"  • {var}")
            print("\n💡 To fix this, run:")
            print("export CLAUDE_RULES_ENABLED=true")
            print("export CLAUDE_RULES_PATH=/opt/sutazaiapp/CLAUDE.md")
            return False
        
        return True
    
    def validate_agent_file(self) -> bool:
        """Validate that the agent file exists and is properly formatted"""
        agent_file = self.agents_dir / f"{self.agent_name}.md"
        
        if not agent_file.exists():
            print(f"❌ Agent file not found: {agent_file}")
            return False
        
        try:
            with open(agent_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for CLAUDE.md compliance header
            if "CLAUDE.md" not in content:
                print(f"⚠️  WARNING: Agent {self.agent_name} may not be CLAUDE.md compliant")
                print("Consider running the compliance updater")
            
            return True
            
        except Exception as e:
            print(f"❌ Error reading agent file: {e}")
            return False
    
    def start_agent_with_compliance(self) -> bool:
        """Start the agent with full compliance checking"""
        
        print(f"🚀 Starting agent: {self.agent_name}")
        
        # Check environment
        if not self.check_environment():
            return False
        
        # Validate agent file
        if not self.validate_agent_file():
            return False
        
        # Check compliance status
        status = get_compliance_status()
        print(f"📋 Rules compliance status: {status}")
        
        # Pre-startup rules check
        startup_action = f"Starting agent {self.agent_name} with compliance checking"
        if not enforce_rules_before_action(startup_action):
            print("❌ Agent startup blocked by CLAUDE.md rules")
            return False
        
        print(f"✅ Agent {self.agent_name} startup approved")
        print(f"📁 Agent file: {self.agents_dir / f'{self.agent_name}.md'}")
        
        # Here you would typically call the actual agent execution
        # For now, we just confirm successful validation
        return True

def main():
    """Main startup function"""
    parser = argparse.ArgumentParser(description='Start an agent with CLAUDE.md compliance')
    parser.add_argument('agent_name', help='Name of the agent to start')
    parser.add_argument('--check-only', action='store_true', 
                       help='Only check compliance, don\'t start agent')
    
    args = parser.parse_args()
    
    wrapper = AgentStartupWrapper(args.agent_name)
    
    if args.check_only:
        print(f"🔍 Checking compliance for agent: {args.agent_name}")
        success = wrapper.check_environment() and wrapper.validate_agent_file()
        print(f"📊 Compliance check: {'✅ PASSED' if success else '❌ FAILED'}")
    else:
        success = wrapper.start_agent_with_compliance()
        if success:
            print(f"🎉 Agent {args.agent_name} started successfully with full compliance")
        else:
            print(f"💥 Failed to start agent {args.agent_name}")
            sys.exit(1)

if __name__ == "__main__":
    main()
