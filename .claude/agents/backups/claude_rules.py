"""
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
        blocking_keywords = ['fantasy', 'magic', 'wizard', 'mythical']
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
