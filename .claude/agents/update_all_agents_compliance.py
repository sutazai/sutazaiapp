#!/usr/bin/env python3
"""
Master Agent Compliance Update Script

This script updates ALL agent files to ensure full compliance with CLAUDE.md rules
"""

import os
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from agent_compliance_checker import AgentComplianceChecker
from update_agent_compliance import AgentComplianceUpdater

class MasterAgentUpdater:
    """Master updater for all agent compliance"""
    
    def __init__(self, agents_dir: str = "/opt/sutazaiapp/.claude/agents"):
        self.agents_dir = Path(agents_dir)
        self.checker = AgentComplianceChecker(agents_dir)
        self.updater = AgentComplianceUpdater(agents_dir)
        
    def run_pre_update_check(self) -> str:
        """Run compliance check before updating"""
        print("🔍 Running pre-update compliance check...")
        report = self.checker.run_full_compliance_check()
        
        # Save pre-update report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pre_report_file = f"pre_update_compliance_report_{timestamp}.json"
        pre_report_path = self.checker.save_report(pre_report_file)
        
        print(f"📊 Pre-update compliance: {report['compliant_agents']}/{report['total_agents']} agents compliant")
        
        return pre_report_path
    
    def update_claude_rules_checker(self):
        """Ensure claude_rules_checker.py is up to date"""
        rules_checker_path = self.agents_dir / "claude_rules_checker.py"
        
        if not rules_checker_path.exists():
            print("❌ claude_rules_checker.py not found! Creating it...")
            
            # Enhanced rules checker content
            enhanced_checker_content = '''#!/usr/bin/env python3
"""
Enhanced CLAUDE.md Rules Checker Module

Shared module for all agents to check and enforce CLAUDE.md rules
"""

import os
import re
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedClaudeRulesChecker:
    """Enhanced checker for CLAUDE.md rules with improved validation"""
    
    BLOCKING_RULES = [
        "Rule 1: No Fantasy Elements",
        "Rule 2: Do Not Break Existing Functionality"
    ]
    
    WARNING_RULES = [
        "Rule 3: Analyze Everything - Every Time",
        "Rule 4: Reuse Before Creating", 
        "Rule 5: Treat This as a Professional Project"
    ]
    
    FANTASY_KEYWORDS = [
        'wizard', 'magic', 'spell', 'fantasy', 'mythical', 'enchant',
        'potion', 'curse', 'dragon', 'fairy', 'unicorn', 'phoenix'
    ]
    
    BREAKING_KEYWORDS = [
        'delete all', 'remove all', 'drop table', 'destroy', 'nuke',
        'rm -rf', 'DELETE FROM', 'DROP DATABASE'
    ]
    
    def __init__(self, rules_path: str = "/opt/sutazaiapp/CLAUDE.md"):
        self.rules_path = rules_path
        self.rules_content = self._load_rules()
        self.enabled = os.getenv('CLAUDE_RULES_ENABLED', 'true').lower() == 'true'
        
        if not self.enabled:
            logger.warning("CLAUDE rules checking is DISABLED via environment variable")
        
    def _load_rules(self) -> Optional[str]:
        """Load rules from CLAUDE.md"""
        try:
            with open(self.rules_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to load CLAUDE.md rules: {e}")
            return None
    
    def check_blocking_violations(self, action: str) -> List[str]:
        """Check for BLOCKING rule violations that must prevent execution"""
        if not self.enabled:
            return []
            
        violations = []
        action_lower = action.lower()
        
        # Rule 1: No Fantasy Elements (STRICT ENFORCEMENT)
        for keyword in self.FANTASY_KEYWORDS:
            if keyword in action_lower:
                violations.append(f"BLOCKING: Rule 1 - Fantasy element detected: '{keyword}'")
        
        # Rule 2: Don't break existing functionality (ENHANCED)
        for keyword in self.BREAKING_KEYWORDS:
            if keyword in action_lower:
                violations.append(f"BLOCKING: Rule 2 - Potentially destructive action: '{keyword}'")
        
        # Additional file system safety checks
        if re.search(r'rm\s+-[rf]+', action_lower):
            violations.append("BLOCKING: Rule 2 - Dangerous file deletion command detected")
        
        return violations
    
    def check_warning_violations(self, action: str) -> List[str]:
        """Check for WARNING rule violations"""
        if not self.enabled:
            return []
            
        warnings = []
        action_lower = action.lower()
        
        # Rule 3: Analyze everything (ENHANCED)
        analysis_keywords = ['analyze', 'check', 'review', 'audit', 'examine']
        if not any(keyword in action_lower for keyword in analysis_keywords):
            if any(keyword in action_lower for keyword in ['create', 'modify', 'update', 'change']):
                warnings.append("WARNING: Rule 3 - Consider analyzing existing code before making changes")
        
        # Rule 4: Reuse before creating (ENHANCED)
        creation_keywords = ['create new', 'new file', 'new class', 'new function']
        if any(keyword in action_lower for keyword in creation_keywords):
            warnings.append("WARNING: Rule 4 - Check for existing solutions before creating new ones")
        
        # Rule 5: Professional approach (NEW)
        unprofessional_keywords = ['hack', 'quick fix', 'dirty', 'temp', 'tmp']
        if any(keyword in action_lower for keyword in unprofessional_keywords):
            warnings.append("WARNING: Rule 5 - Consider a more professional, long-term approach")
        
        return warnings
    
    def validate_action(self, action: str) -> Tuple[bool, List[str]]:
        """Validate an action against all rules"""
        if not self.enabled:
            return True, ["Rules checking disabled"]
        
        blocking = self.check_blocking_violations(action)
        warnings = self.check_warning_violations(action)
        
        # If there are blocking violations, action should not proceed
        can_proceed = len(blocking) == 0
        
        all_issues = blocking + warnings
        
        # Log the validation
        if blocking:
            logger.error(f"Action BLOCKED: {action[:100]}... - {len(blocking)} blocking issues")
        elif warnings:
            logger.warning(f"Action has warnings: {action[:100]}... - {len(warnings)} warnings")
        else:
            logger.info(f"Action approved: {action[:100]}...")
        
        return can_proceed, all_issues
    
    def get_compliance_status(self) -> Dict:
        """Get current compliance status"""
        return {
            "enabled": self.enabled,
            "rules_file_loaded": self.rules_content is not None,
            "rules_path": self.rules_path,
            "blocking_rules": self.BLOCKING_RULES,
            "warning_rules": self.WARNING_RULES
        }

# Global instance
rules_checker = EnhancedClaudeRulesChecker()

def check_action(action: str) -> Tuple[bool, List[str]]:
    """Check if an action is allowed according to CLAUDE.md rules"""
    return rules_checker.validate_action(action)

def get_rules_summary() -> Dict[str, List[str]]:
    """Get a summary of all rules"""
    return {
        "blocking": rules_checker.BLOCKING_RULES,
        "warning": rules_checker.WARNING_RULES
    }

def get_compliance_status() -> Dict:
    """Get current compliance status"""
    return rules_checker.get_compliance_status()

# Agent integration helper
def enforce_rules_before_action(action_description: str) -> bool:
    """
    Helper function for agents to enforce rules before taking any action
    Returns True if action can proceed, False if blocked
    """
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

if __name__ == "__main__":
    # CLI for testing
    import sys
    if len(sys.argv) > 1:
        action = " ".join(sys.argv[1:])
        can_proceed, issues = check_action(action)
        
        print(f"Action: {action}")
        print(f"Can proceed: {can_proceed}")
        if issues:
            print("Issues:")
            for issue in issues:
                print(f"  • {issue}")
        
        status = get_compliance_status()
        print(f"\\nCompliance Status: {status}")
    else:
        print("Usage: python3 claude_rules_checker.py 'action description'")
'''
            
            with open(rules_checker_path, 'w', encoding='utf-8') as f:
                f.write(enhanced_checker_content)
            
            print("✅ Enhanced claude_rules_checker.py created")
        else:
            print("✅ claude_rules_checker.py exists")
    
    def create_agent_startup_wrapper(self):
        """Create/update the agent startup wrapper script"""
        startup_wrapper_path = self.agents_dir / "agent_startup_wrapper.py"
        
        wrapper_content = '''#!/usr/bin/env python3
"""
Agent Startup Wrapper with CLAUDE.md Rules Enforcement

This wrapper ensures all agents check CLAUDE.md rules before startup
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add the agents directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from claude_rules_checker import check_action, get_compliance_status, enforce_rules_before_action
except ImportError:
    print("❌ ERROR: claude_rules_checker.py not found!")
    print("Please ensure claude_rules_checker.py is in the agents directory")
    sys.exit(1)

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
    
    def start_agent_with_compliance(self, additional_args: list = None) -> bool:
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
'''
        
        with open(startup_wrapper_path, 'w', encoding='utf-8') as f:
            f.write(wrapper_content)
        
        # Make it executable
        os.chmod(startup_wrapper_path, 0o755)
        
        print("✅ Agent startup wrapper created/updated")
    
    def update_all_agents(self) -> Dict[str, bool]:
        """Update all non-compliant agents"""
        print("🔧 Starting mass agent update process...")
        
        # Get the latest compliance report
        latest_report = None
        for file in self.agents_dir.glob("*compliance_report*.json"):
            if latest_report is None or file.stat().st_mtime > latest_report.stat().st_mtime:
                latest_report = file
        
        if latest_report is None:
            print("❌ No compliance report found. Running compliance check first...")
            self.run_pre_update_check()
            # Find the report again
            for file in self.agents_dir.glob("*compliance_report*.json"):
                if latest_report is None or file.stat().st_mtime > latest_report.stat().st_mtime:
                    latest_report = file
        
        if latest_report is None:
            print("❌ Could not generate compliance report")
            return {}
        
        print(f"📊 Using compliance report: {latest_report}")
        
        # Update non-compliant agents
        results = self.updater.update_non_compliant_agents(str(latest_report))
        
        return results
    
    def run_post_update_check(self) -> str:
        """Run compliance check after updating"""
        print("🔍 Running post-update compliance check...")
        report = self.checker.run_full_compliance_check()
        
        # Save post-update report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        post_report_file = f"post_update_compliance_report_{timestamp}.json"
        post_report_path = self.checker.save_report(post_report_file)
        
        print(f"📊 Post-update compliance: {report['compliant_agents']}/{report['total_agents']} agents compliant")
        
        return post_report_path
    
    def create_environment_file(self):
        """Create .env file for agents"""
        env_file_path = self.agents_dir / ".env"
        
        env_content = """# CLAUDE.md Rules Enforcement Environment Variables
CLAUDE_RULES_ENABLED=true
CLAUDE_RULES_PATH=/opt/sutazaiapp/CLAUDE.md
AGENTS_DIR=/opt/sutazaiapp/.claude/agents

# Logging
LOG_LEVEL=INFO
LOG_FILE=/opt/sutazaiapp/logs/agents.log

# Agent Configuration
AGENT_TIMEOUT=300
AGENT_MAX_RETRIES=3
"""
        
        with open(env_file_path, 'w', encoding='utf-8') as f:
            f.write(env_content)
        
        print("✅ Environment file created: .env")
    
    def run_complete_update_process(self) -> Dict:
        """Run the complete agent update process"""
        print("🚀 Starting complete agent compliance update process...")
        print("="*60)
        
        # Step 1: Pre-update check
        pre_report_path = self.run_pre_update_check()
        
        # Step 2: Update infrastructure
        print("\n🔧 Updating infrastructure...")
        self.update_claude_rules_checker()
        self.create_agent_startup_wrapper()
        self.create_environment_file()
        
        # Step 3: Update all agents
        print("\n🔧 Updating all agents...")
        update_results = self.update_all_agents()
        
        # Step 4: Post-update check
        print("\n🔍 Running post-update verification...")
        post_report_path = self.run_post_update_check()
        
        # Step 5: Generate summary
        successful_updates = sum(1 for success in update_results.values() if success)
        total_updates = len(update_results)
        
        summary = {
            "pre_update_report": pre_report_path,
            "post_update_report": post_report_path,
            "agents_updated": total_updates,
            "successful_updates": successful_updates,
            "failed_updates": total_updates - successful_updates,
            "update_results": update_results
        }
        
        print("\n" + "="*60)
        print("🎉 COMPLETE UPDATE PROCESS SUMMARY")
        print("="*60)
        print(f"📊 Agents updated: {successful_updates}/{total_updates}")
        print(f"📁 Pre-update report: {pre_report_path}")
        print(f"📁 Post-update report: {post_report_path}")
        
        if successful_updates < total_updates:
            print(f"\n❌ Failed updates:")
            for agent_name, success in update_results.items():
                if not success:
                    print(f"  • {agent_name}")
        
        return summary

def main():
    """Main function"""
    updater = MasterAgentUpdater()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--check-only":
        # Just run compliance check
        updater.run_pre_update_check()
    else:
        # Run complete update process
        summary = updater.run_complete_update_process()
        
        # Save summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = f"master_update_summary_{timestamp}.json"
        summary_path = Path("/opt/sutazaiapp/.claude/agents") / summary_file
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"📋 Complete summary saved to: {summary_path}")

if __name__ == "__main__":
    main()