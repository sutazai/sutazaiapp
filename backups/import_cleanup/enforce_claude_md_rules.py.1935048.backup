#!/usr/bin/env python3
"""
Enforce CLAUDE.md Rules for All AI Agents

This script updates all AI agents to automatically check and follow
the rules defined in /opt/sutazaiapp/CLAUDE.md
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, List, Any

AGENTS_DIR = "/opt/sutazaiapp/.claude/agents"
CLAUDE_MD_PATH = "/opt/sutazaiapp/CLAUDE.md"
DOCKER_COMPOSE_FILES = [
    "/opt/sutazaiapp/docker-compose.yml",
    "/opt/sutazaiapp/docker-compose.agents-simple.yml",
    "/opt/sutazaiapp/docker-compose.complete-agents.yml"
]

def read_claude_md_rules() -> str:
    """Read the rules from CLAUDE.md"""
    with open(CLAUDE_MD_PATH, 'r') as f:
        return f.read()

def update_agent_yaml(agent_file: Path) -> bool:
    """Update agent YAML to include CLAUDE.md rules checking"""
    try:
        with open(agent_file, 'r') as f:
            content = f.read()
        
        # Split YAML frontmatter and content
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                yaml_content = parts[1]
                markdown_content = parts[2]
            else:
                return False
        else:
            return False
        
        # Parse YAML
        agent_config = yaml.safe_load(yaml_content)
        
        # Add CLAUDE.md rules reference
        if 'system_prompt' not in agent_config:
            agent_config['system_prompt'] = ""
        
        # Prepend CLAUDE.md rules checking instruction
        rules_instruction = """
IMPORTANT: Before executing any task, you MUST:
1. Check the codebase hygiene rules in /opt/sutazaiapp/CLAUDE.md
2. Ensure all actions comply with these rules
3. Follow the three-tier enforcement (BLOCKING/WARNING/GUIDANCE)
4. Never violate BLOCKING rules
5. Document any WARNING-level concerns
6. Apply GUIDANCE-level best practices

Rules file location: /opt/sutazaiapp/CLAUDE.md
"""
        
        if rules_instruction.strip() not in agent_config['system_prompt']:
            agent_config['system_prompt'] = rules_instruction + "\n" + agent_config.get('system_prompt', '')
        
        # Add rules_file reference
        agent_config['rules_file'] = CLAUDE_MD_PATH
        agent_config['enforce_rules'] = True
        
        # Write back
        updated_yaml = yaml.dump(agent_config, default_flow_style=False, sort_keys=False)
        updated_content = f"---\n{updated_yaml}---\n{markdown_content}"
        
        with open(agent_file, 'w') as f:
            f.write(updated_content)
        
        return True
    
    except Exception as e:
        print(f"Error updating {agent_file}: {e}")
        return False

def update_agent_python_files(agent_dir: Path) -> bool:
    """Update Python files in agent directory to check CLAUDE.md"""
    updated = False
    
    for py_file in agent_dir.glob("*.py"):
        try:
            with open(py_file, 'r') as f:
                content = f.read()
            
            # Check if already has rules checking
            if "CLAUDE.md" in content:
                continue
            
            # Add rules checking code after imports
            rules_code = '''
# CLAUDE.md Rules Enforcement
import os
from pathlib import Path

CLAUDE_MD_PATH = "/opt/sutazaiapp/CLAUDE.md"

def check_claude_rules():
    """Check and load CLAUDE.md rules"""
    if os.path.exists(CLAUDE_MD_PATH):
        with open(CLAUDE_MD_PATH, 'r') as f:
            return f.read()
    return None

# Load rules at startup
CLAUDE_RULES = check_claude_rules()
'''
            
            # Find where to insert (after imports)
            lines = content.split('\n')
            insert_index = 0
            
            for i, line in enumerate(lines):
                if line.strip() and not line.startswith('import') and not line.startswith('from'):
                    if i > 0:
                        insert_index = i
                        break
            
            # Insert rules code
            lines.insert(insert_index, rules_code)
            
            updated_content = '\n'.join(lines)
            
            with open(py_file, 'w') as f:
                f.write(updated_content)
            
            updated = True
            
        except Exception as e:
            print(f"Error updating {py_file}: {e}")
    
    return updated

def update_docker_compose_files() -> None:
    """Update docker-compose files to mount CLAUDE.md"""
    for compose_file in DOCKER_COMPOSE_FILES:
        if not os.path.exists(compose_file):
            continue
        
        try:
            with open(compose_file, 'r') as f:
                compose_data = yaml.safe_load(f)
            
            if not compose_data or 'services' not in compose_data:
                continue
            
            # Update each service
            for service_name, service_config in compose_data['services'].items():
                if 'agent' not in service_name and service_name != 'ollama':
                    continue
                
                # Ensure volumes list exists
                if 'volumes' not in service_config:
                    service_config['volumes'] = []
                
                # Add CLAUDE.md mount if not present
                claude_mount = "/opt/sutazaiapp/CLAUDE.md:/app/CLAUDE.md:ro"
                if claude_mount not in service_config['volumes']:
                    service_config['volumes'].append(claude_mount)
                
                # Add environment variable
                if 'environment' not in service_config:
                    service_config['environment'] = {}
                
                service_config['environment']['CLAUDE_RULES_PATH'] = '/app/CLAUDE.md'
                service_config['environment']['ENFORCE_CLAUDE_RULES'] = 'true'
            
            # Write back
            with open(compose_file, 'w') as f:
                yaml.dump(compose_data, f, default_flow_style=False, sort_keys=False)
            
            print(f"Updated {compose_file}")
            
        except Exception as e:
            print(f"Error updating {compose_file}: {e}")

def create_rules_checker_module() -> None:
    """Create a shared module for rules checking"""
    module_content = '''#!/usr/bin/env python3
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
        fantasy_keywords = ['configurator', 'specific implementation name (e.g., emailSender, dataProcessor)', 'spell', 'fantasy', 'mythical']
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
'''
    
    module_path = Path(AGENTS_DIR) / "claude_rules_checker.py"
    with open(module_path, 'w') as f:
        f.write(module_content)
    
    print(f"Created rules checker module: {module_path}")

def run_comprehensive_check() -> int:
    """Run comprehensive check of all CLAUDE.md rules"""
    print("🔍 Running comprehensive CLAUDE.md compliance check...")
    
    violations = []
    warnings = []
    
    # Check for garbage files (Rule 13)
    garbage_patterns = ['*.backup*', '*.tmp', '*.bak', '*~', '*.old', '*.agi_backup']
    for pattern in garbage_patterns:
        result = os.popen(f'find /opt/sutazaiapp -name "{pattern}" -type f 2>/dev/null | grep -v archive | grep -v .git').read()
        if result.strip():
            violations.append(f"Rule 13: Garbage files found:\n{result}")
    
    # Check for multiple deployment scripts (Rule 12)
    deploy_scripts = os.popen('find /opt/sutazaiapp -name "deploy*.sh" -o -name "*deploy*.py" | grep -v archive').read()
    if len(deploy_scripts.strip().split('\n')) > 1:
        violations.append(f"Rule 12: Multiple deployment scripts found:\n{deploy_scripts}")
    
    # Check Docker structure (Rule 11)
    dockerfiles = os.popen('find /opt/sutazaiapp -name "Dockerfile*" | grep -v archive | wc -l').read().strip()
    if int(dockerfiles) > 20:
        warnings.append(f"Rule 11: {dockerfiles} Dockerfiles found - consider consolidation")
    
    # Check Python documentation (Rule 8)
    py_files = os.popen('find /opt/sutazaiapp -name "*.py" | grep -v venv | grep -v __pycache__ | head -20').read()
    for py_file in py_files.strip().split('\n'):
        if py_file:
            try:
                with open(py_file, 'r') as f:
                    content = f.read(500)
                    if 'Purpose:' not in content and 'purpose:' not in content:
                        warnings.append(f"Rule 8: {py_file} missing documentation")
            except:
                pass
    
    # Report results
    if violations:
        print("\n❌ BLOCKING VIOLATIONS FOUND:")
        for v in violations:
            print(f"\n{v}")
        return 1
    
    if warnings:
        print("\n⚠️  WARNINGS:")
        for w in warnings[:10]:  # Limit output
            print(f"\n{w}")
    
    print("\n✅ Comprehensive check complete")
    return 0

def main():
    """Main function to enforce CLAUDE.md rules across all agents"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enforce CLAUDE.md rules")
    parser.add_argument('--comprehensive', action='store_true', help='Run comprehensive compliance check')
    args = parser.parse_args()
    
    if args.comprehensive:
        return run_comprehensive_check()
    
    print("Enforcing CLAUDE.md rules for all AI agents...")
    
    # Create shared rules checker module
    create_rules_checker_module()
    
    # Update all agent YAML files
    agent_files = list(Path(AGENTS_DIR).glob("*.md"))
    updated_count = 0
    
    for agent_file in agent_files:
        if agent_file.name == "CLAUDE.md":
            continue
        
        print(f"Updating {agent_file.name}...")
        if update_agent_yaml(agent_file):
            updated_count += 1
    
    print(f"\nUpdated {updated_count} agent YAML configurations")
    
    # Update Python files in agent directories
    agent_dirs = [d for d in Path(AGENTS_DIR).iterdir() if d.is_dir()]
    
    for agent_dir in agent_dirs:
        print(f"Updating Python files in {agent_dir.name}...")
        update_agent_python_files(agent_dir)
    
    # Update docker-compose files
    print("\nUpdating docker-compose files...")
    update_docker_compose_files()
    
    print("\nCreating agent startup wrapper...")
    
    # Create a startup wrapper that all agents will use
    wrapper_content = '''#!/usr/bin/env python3
"""
Agent Startup Wrapper
Ensures all agents check CLAUDE.md rules before starting
"""

import os
import sys
from pathlib import Path

# Add rules checker to path
sys.path.insert(0, '/opt/sutazaiapp/.claude/agents')

from claude_rules_checker import rules_checker, get_rules_summary

def initialize_with_rules():
    """Initialize agent with CLAUDE.md rules awareness"""
    print("Loading CLAUDE.md rules...")
    rules_summary = get_rules_summary()
    
    print("\\nBLOCKING Rules (must never violate):")
    for rule in rules_summary['blocking']:
        print(f"  - {rule}")
    
    print("\\nWARNING Rules (require careful consideration):")
    for rule in rules_summary['warning']:
        print(f"  - {rule}")
    
    print("\\nRules enforcement initialized successfully!")

if __name__ == "__main__":
    initialize_with_rules()
    
    # Continue with normal agent startup
    if len(sys.argv) > 1:
        import importlib.util
        spec = importlib.util.spec_from_file_location("agent_module", sys.argv[1])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
'''
    
    wrapper_path = Path(AGENTS_DIR) / "agent_startup_wrapper.py"
    with open(wrapper_path, 'w') as f:
        f.write(wrapper_content)
    
    os.chmod(wrapper_path, 0o755)
    print(f"Created agent startup wrapper: {wrapper_path}")
    
    print("\n✅ CLAUDE.md rules enforcement has been implemented for all agents!")
    print("\nNext steps:")
    print("1. Restart all agent containers to apply changes")
    print("2. Monitor agent logs to ensure rules are being checked")
    print("3. Test agent compliance with sample tasks")

if __name__ == "__main__":
    main()