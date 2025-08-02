#!/usr/bin/env python3
"""
CLAUDE.md Rules Checker Module

Shared module for all agents to check and enforce CLAUDE.md rules
"""

import os
import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path

class ClaudeRulesChecker:
    """Checks and enforces CLAUDE.md rules"""
    
    BLOCKING_RULES = [
        "Rule 1: No Fantasy Elements",
        "Rule 2: Do Not Break Existing Functionality"
    ]
    
    WARNING_RULES = [
        "Rule 3: Analyze Everything",
        "Rule 4: Reuse Before Creating",
        "Rule 5: Professional Project"
    ]
    
    def __init__(self, rules_path: str = "/opt/sutazaiapp/CLAUDE.md"):
        self.rules_path = rules_path
        self.rules_content = self._load_rules()
        
    def _load_rules(self) -> Optional[str]:
        """Load rules from CLAUDE.md"""
        try:
            with open(self.rules_path, 'r') as f:
                return f.read()
        except:
            return None
    
    def check_blocking_violations(self, action: str) -> List[str]:
        """Check for BLOCKING rule violations"""
        violations = []
        
        # Rule 1: No Fantasy Elements
        fantasy_keywords = ['wizard', 'magic', 'spell', 'fantasy', 'mythical']
        if any(keyword in action.lower() for keyword in fantasy_keywords):
            violations.append("BLOCKING: Rule 1 - No fantasy elements allowed")
        
        # Rule 2: Don't break existing functionality
        breaking_keywords = ['delete', 'remove', 'drop', 'destroy']
        if any(keyword in action.lower() for keyword in breaking_keywords):
            violations.append("BLOCKING: Rule 2 - Verify this won't break existing functionality")
        
        return violations
    
    def check_warning_violations(self, action: str) -> List[str]:
        """Check for WARNING rule violations"""
        warnings = []
        
        # Rule 3: Analyze everything
        if 'analyze' not in action.lower() and 'check' not in action.lower():
            warnings.append("WARNING: Rule 3 - Ensure thorough analysis before proceeding")
        
        # Rule 4: Reuse before creating
        if 'create' in action.lower() or 'new' in action.lower():
            warnings.append("WARNING: Rule 4 - Check for existing solutions before creating new ones")
        
        return warnings
    
    def validate_action(self, action: str) -> Tuple[bool, List[str]]:
        """Validate an action against all rules"""
        blocking = self.check_blocking_violations(action)
        warnings = self.check_warning_violations(action)
        
        # If there are blocking violations, action should not proceed
        can_proceed = len(blocking) == 0
        
        all_issues = blocking + warnings
        
        return can_proceed, all_issues

# Global instance
rules_checker = ClaudeRulesChecker()

def check_action(action: str) -> Tuple[bool, List[str]]:
    """Check if an action is allowed according to CLAUDE.md rules"""
    return rules_checker.validate_action(action)

def get_rules_summary() -> Dict[str, List[str]]:
    """Get a summary of all rules"""
    return {
        "blocking": rules_checker.BLOCKING_RULES,
        "warning": rules_checker.WARNING_RULES
    }
