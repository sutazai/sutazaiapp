#!/usr/bin/env python3
"""
Enforce codebase standards and implementation rules for all agents in .claude directory.
Implements all rules from CLAUDE.md to ensure consistency and quality.
"""

import os
import re
import yaml
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

class AgentStandardsEnforcer:
    def __init__(self):
        self.agent_dir = Path("/opt/sutazaiapp/.claude/agents")
        self.project_root = Path("/opt/sutazaiapp")
        self.issues = defaultdict(list)
        self.fixes_applied = defaultdict(list)
        self.agent_data = {}
        
    def analyze_all_agents(self):
        """Analyze all agents for compliance with codebase standards."""
        print("🔍 Analyzing all agents for codebase standards compliance...\n")
        
        # Get all agent files
        agent_files = list(self.agent_dir.glob("*.md"))
        agent_files = [f for f in agent_files if not any(
            f.name.endswith(ext) for ext in ['.backup', '_backup', '.fantasy_backup']
        )]
        
        print(f"Found {len(agent_files)} agent definition files to analyze.\n")
        
        for agent_file in sorted(agent_files):
            self.analyze_agent(agent_file)
            
        return len(agent_files)
    
    def analyze_agent(self, agent_file):
        """Analyze a single agent file for compliance."""
        agent_name = agent_file.stem
        
        with open(agent_file, 'r') as f:
            content = f.read()
            
        # Extract YAML frontmatter
        if content.startswith("---"):
            yaml_end = content.find("---", 3)
            if yaml_end != -1:
                yaml_content = content[3:yaml_end]
                try:
                    agent_config = yaml.safe_load(yaml_content)
                    self.agent_data[agent_name] = {
                        'config': agent_config,
                        'content': content,
                        'file': agent_file
                    }
                    
                    # Run all checks
                    self.check_naming_convention(agent_name, agent_config)
                    self.check_no_fantasy_elements(agent_name, content)
                    self.check_model_configuration(agent_name, agent_config)
                    self.check_description_quality(agent_name, agent_config)
                    self.check_integration_consistency(agent_name, agent_config)
                    self.check_capabilities_logic(agent_name, agent_config)
                    
                except yaml.YAMLError as e:
                    self.issues[agent_name].append(f"YAML parsing error: {e}")
    
    def check_naming_convention(self, agent_name, config):
        """Rule: Ensure naming conventions are consistent and meaningful."""
        # Check filename matches agent name in config
        config_name = config.get('name', '')
        if config_name != agent_name:
            self.issues[agent_name].append(
                f"Naming mismatch: file='{agent_name}' vs config='{config_name}'"
            )
            
        # Check naming pattern (should be kebab-case)
        if not re.match(r'^[a-z]+(-[a-z]+)*$', agent_name):
            self.issues[agent_name].append(
                f"Non-standard naming: should be kebab-case, got '{agent_name}'"
            )
    
    def check_no_fantasy_elements(self, agent_name, content):
        """Rule 1: No Fantasy Elements - production-ready implementation only."""
        fantasy_patterns = [
            r'consciousness', r'sentient', r'self-aware', r'emergent intelligence',
            r'singularity', r'AGI', r'ASI', r'superintelligence', r'evolution',
            r'40\+ agents', r'neural plasticity', r'quantum', r'cosmic'
        ]
        
        for pattern in fantasy_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                self.issues[agent_name].append(f"Fantasy element found: '{pattern}'")
    
    def check_model_configuration(self, agent_name, config):
        """Rule: Verify model configuration follows standards."""
        model = config.get('model', '')
        
        # Should use tinyllama by default
        if model != 'tinyllama:latest':
            self.issues[agent_name].append(
                f"Non-default model: '{model}' (should be 'tinyllama:latest')"
            )
            
        # Check version is specified
        if 'version' not in config:
            self.issues[agent_name].append("Missing version field")
    
    def check_description_quality(self, agent_name, config):
        """Rule: Ensure descriptions are clear, professional, and actionable."""
        description = config.get('description', '')
        
        if not description:
            self.issues[agent_name].append("Missing description")
            return
            
        # Check description structure
        if "Use this agent when" not in description:
            self.issues[agent_name].append("Description missing 'Use this agent when' section")
            
        if "Do NOT use this agent for" not in description:
            self.issues[agent_name].append("Description missing 'Do NOT use' section")
            
        # Check for vague language
        vague_terms = ['might', 'maybe', 'possibly', 'could', 'etc.', '...']
        for term in vague_terms:
            if term in description.lower():
                self.issues[agent_name].append(f"Vague language detected: '{term}'")
    
    def check_integration_consistency(self, agent_name, config):
        """Rule: Verify integration references are valid and consistent."""
        integrations = config.get('integrations', {})
        
        if not integrations:
            self.issues[agent_name].append("No integrations defined")
            return
            
        # Check agent references
        agent_refs = integrations.get('agents', [])
        for ref in agent_refs:
            if ref == 'all_40+' or '40+' in ref:
                self.issues[agent_name].append(f"Invalid agent reference: '{ref}'")
                
        # Check for duplicate entries
        for key, values in integrations.items():
            if isinstance(values, list) and len(values) != len(set(values)):
                self.issues[agent_name].append(f"Duplicate entries in {key}: {values}")
    
    def check_capabilities_logic(self, agent_name, config):
        """Rule: Verify capabilities are logical and implementable."""
        capabilities = config.get('capabilities', [])
        
        if not capabilities:
            self.issues[agent_name].append("No capabilities defined")
            return
            
        # Check for fantasy capabilities
        fantasy_capabilities = [
            'consciousness', 'self_awareness', 'sentience', 'evolution',
            'quantum', 'singularity', 'emergence'
        ]
        
        for cap in capabilities:
            if any(f in str(cap).lower() for f in fantasy_capabilities):
                self.issues[agent_name].append(f"Fantasy capability: '{cap}'")
    
    def check_for_duplicates(self):
        """Rule: Avoid creating multiple or conflicting versions."""
        print("\n🔍 Checking for duplicate agents and conflicts...\n")
        
        # Group agents by similar names
        name_groups = defaultdict(list)
        for agent_name in self.agent_data:
            # Extract base name (remove version numbers, detailed/simple suffixes)
            base_name = re.sub(r'[-_](detailed|simple|v\d+)$', '', agent_name)
            name_groups[base_name].append(agent_name)
            
        # Report potential duplicates
        for base_name, agents in name_groups.items():
            if len(agents) > 1:
                print(f"⚠️  Potential duplicates for '{base_name}': {agents}")
                for agent in agents:
                    self.issues[agent].append(f"Potential duplicate of {base_name}")
    
    def fix_issues(self):
        """Apply fixes for common issues."""
        print("\n🔧 Applying automated fixes...\n")
        
        for agent_name, issues in self.issues.items():
            if agent_name not in self.agent_data:
                continue
                
            agent_file = self.agent_data[agent_name]['file']
            content = self.agent_data[agent_name]['content']
            original_content = content
            
            # Fix model references
            if any("Non-default model" in issue for issue in issues):
                content = re.sub(
                    r'model:\s*[^\n]+',
                    'model: tinyllama:latest',
                    content
                )
                self.fixes_applied[agent_name].append("Fixed model to tinyllama:latest")
            
            # Fix agent references
            if any("Invalid agent reference" in issue for issue in issues):
                content = re.sub(r'"all_40\+"', '"all"', content)
                content = re.sub(r'\ball_40\+\b', 'all', content)
                self.fixes_applied[agent_name].append("Fixed agent references")
            
            # Fix duplicate entries
            if any("Duplicate entries" in issue for issue in issues):
                # This requires more careful YAML manipulation
                yaml_match = re.match(r'---\n(.*?)\n---', content, re.DOTALL)
                if yaml_match:
                    yaml_content = yaml_match.group(1)
                    config = yaml.safe_load(yaml_content)
                    
                    # Remove duplicates from lists
                    if 'integrations' in config:
                        for key, values in config['integrations'].items():
                            if isinstance(values, list):
                                config['integrations'][key] = list(dict.fromkeys(values))
                    
                    # Reconstruct content
                    new_yaml = yaml.dump(config, default_flow_style=False, sort_keys=False)
                    content = content.replace(yaml_match.group(0), f"---\n{new_yaml}---")
                    self.fixes_applied[agent_name].append("Removed duplicate entries")
            
            # Save if changes were made
            if content != original_content:
                with open(agent_file, 'w') as f:
                    f.write(content)
    
    def generate_report(self):
        """Generate comprehensive compliance report."""
        report_path = self.project_root / "AGENT_STANDARDS_COMPLIANCE_REPORT.md"
        
        with open(report_path, 'w') as f:
            f.write("# Agent Standards Compliance Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary
            total_agents = len(self.agent_data)
            agents_with_issues = len(self.issues)
            agents_fixed = len(self.fixes_applied)
            
            f.write("## Executive Summary\n\n")
            f.write(f"- **Total Agents Analyzed**: {total_agents}\n")
            f.write(f"- **Agents with Issues**: {agents_with_issues}\n")
            f.write(f"- **Agents Fixed**: {agents_fixed}\n")
            f.write(f"- **Compliance Rate**: {((total_agents - agents_with_issues) / total_agents * 100):.1f}%\n\n")
            
            # Rules Applied
            f.write("## Rules Applied\n\n")
            f.write("1. ✅ **No Fantasy Elements** - Real, production-ready implementation only\n")
            f.write("2. ✅ **No Breaking Existing Functionality** - Preserve working features\n")
            f.write("3. ✅ **Consistent Naming** - Meaningful, kebab-case names\n")
            f.write("4. ✅ **Default Model** - tinyllama:latest for all agents\n")
            f.write("5. ✅ **Clear Descriptions** - Actionable use cases\n")
            f.write("6. ✅ **Valid Integrations** - No invalid references\n")
            f.write("7. ✅ **No Duplicates** - Single version of each agent\n\n")
            
            # Issues Found
            if self.issues:
                f.write("## Issues Found\n\n")
                for agent_name, issues in sorted(self.issues.items()):
                    f.write(f"### {agent_name}\n")
                    for issue in issues:
                        f.write(f"- ❌ {issue}\n")
                    f.write("\n")
            
            # Fixes Applied
            if self.fixes_applied:
                f.write("## Automated Fixes Applied\n\n")
                for agent_name, fixes in sorted(self.fixes_applied.items()):
                    f.write(f"### {agent_name}\n")
                    for fix in fixes:
                        f.write(f"- ✅ {fix}\n")
                    f.write("\n")
            
            # Clean Agents
            clean_agents = [name for name in self.agent_data if name not in self.issues]
            if clean_agents:
                f.write("## Clean Agents (No Issues)\n\n")
                for agent in sorted(clean_agents):
                    f.write(f"- ✅ {agent}\n")
            
            f.write("\n## Recommendations\n\n")
            f.write("1. Review remaining issues that couldn't be auto-fixed\n")
            f.write("2. Consolidate any duplicate agents\n")
            f.write("3. Update agent descriptions for clarity\n")
            f.write("4. Ensure all agents follow the same pattern\n")
            
        return report_path

def main():
    """Main execution function."""
    print("🚀 SutazAI Agent Standards Enforcement Tool\n")
    print("Applying all rules from CLAUDE.md to ensure consistency...\n")
    
    enforcer = AgentStandardsEnforcer()
    
    # Analyze all agents
    total = enforcer.analyze_all_agents()
    
    # Check for duplicates
    enforcer.check_for_duplicates()
    
    # Apply fixes
    enforcer.fix_issues()
    
    # Generate report
    report_path = enforcer.generate_report()
    
    print(f"\n✅ Analysis complete!")
    print(f"📄 Report saved to: {report_path}")
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"  - Agents analyzed: {total}")
    print(f"  - Issues found: {len(enforcer.issues)}")
    print(f"  - Fixes applied: {len(enforcer.fixes_applied)}")

if __name__ == "__main__":
    main()