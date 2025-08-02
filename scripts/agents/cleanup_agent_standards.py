#!/usr/bin/env python3
"""
Agent Standards Compliance Script
Ensures all agents follow the codebase standards and rules
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
import yaml

class AgentStandardsEnforcer:
    def __init__(self, agents_dir: str = "/opt/sutazaiapp/.claude/agents"):
        self.agents_dir = Path(agents_dir)
        self.issues = []
        self.fixes_applied = []
        
        # Define fantasy terms to remove
        self.fantasy_terms = {
            'wizard', 'magic', 'spell', 'enchant', 'dragon', 'mythical', 
            'fantasy', 'supernatural', 'mystical', 'sorcerer', 'mage', 
            'alchemy', 'arcane', 'celestial', 'divine', 'ethereal', 
            'necromancer', 'warlock', 'enchantment', 'magical'
        }
        
        # Valid agent file patterns
        self.valid_patterns = {
            'standard': re.compile(r'^[a-z\-]+\.md$'),
            'detailed': re.compile(r'^[a-z\-]+-detailed\.md$'),
            'meta': re.compile(r'^[A-Z_]+\.md$'),  # For summary/protocol files
            'list': re.compile(r'^[a-z_]+\.txt$')
        }
        
    def run_full_compliance_check(self):
        """Run all compliance checks and fixes"""
        print("🔍 Starting Agent Standards Compliance Check...")
        
        # Step 1: Identify all issues
        self.check_file_naming()
        self.check_duplicates()
        self.check_fantasy_elements()
        self.check_file_structure()
        
        # Step 2: Apply fixes
        self.fix_duplicates()
        self.fix_naming_issues()
        self.remove_fantasy_elements()
        self.standardize_structure()
        
        # Step 3: Generate report
        self.generate_report()
        
    def check_file_naming(self):
        """Check for consistent naming conventions"""
        for file_path in self.agents_dir.glob("*"):
            if file_path.is_file():
                filename = file_path.name
                valid = any(pattern.match(filename) for pattern in self.valid_patterns.values())
                
                if not valid:
                    self.issues.append({
                        'type': 'naming',
                        'file': filename,
                        'issue': 'Does not match standard naming convention'
                    })
    
    def check_duplicates(self):
        """Check for duplicate agent definitions"""
        agents = {}
        for file_path in self.agents_dir.glob("*.md"):
            if file_path.name.endswith('-detailed.md'):
                base_name = file_path.name.replace('-detailed.md', '')
            else:
                base_name = file_path.name.replace('.md', '')
            
            if base_name not in agents:
                agents[base_name] = []
            agents[base_name].append(file_path.name)
        
        # Check for deploy vs deployment duplicates
        if 'deploy-automation-master' in agents and 'deployment-automation-master' in agents:
            self.issues.append({
                'type': 'duplicate',
                'files': ['deploy-automation-master*', 'deployment-automation-master*'],
                'issue': 'Duplicate agent with different naming'
            })
    
    def check_fantasy_elements(self):
        """Check for fantasy elements in agent files"""
        for file_path in self.agents_dir.glob("*.md"):
            content = file_path.read_text(encoding='utf-8', errors='ignore').lower()
            
            found_terms = []
            for term in self.fantasy_terms:
                # Skip legitimate technical uses
                if term == 'image' and 'container' in content:
                    continue
                if term == 'magic' and ('magic number' in content or 'imagemagick' in content):
                    continue
                    
                if term in content:
                    found_terms.append(term)
            
            if found_terms:
                self.issues.append({
                    'type': 'fantasy_elements',
                    'file': file_path.name,
                    'terms': found_terms,
                    'issue': f'Contains fantasy terms: {", ".join(found_terms)}'
                })
    
    def check_file_structure(self):
        """Check for consistent file structure"""
        required_sections = ['name:', 'description:', 'model:', 'version:', 'capabilities:']
        
        for file_path in self.agents_dir.glob("*.md"):
            if file_path.name.startswith(('AGENT_', 'COMPREHENSIVE_')) or file_path.name.endswith('.txt'):
                continue
                
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            # Check for YAML frontmatter
            if not content.strip().startswith('---'):
                self.issues.append({
                    'type': 'structure',
                    'file': file_path.name,
                    'issue': 'Missing YAML frontmatter'
                })
                continue
            
            # Check for required sections
            missing_sections = []
            for section in required_sections:
                if section not in content.split('---')[1]:
                    missing_sections.append(section)
            
            if missing_sections:
                self.issues.append({
                    'type': 'structure',
                    'file': file_path.name,
                    'issue': f'Missing required sections: {", ".join(missing_sections)}'
                })
    
    def fix_duplicates(self):
        """Consolidate duplicate agent files"""
        # Handle deploy vs deployment duplicate
        deploy_files = list(self.agents_dir.glob("deploy-automation-master*"))
        deployment_files = list(self.agents_dir.glob("deployment-automation-master*"))
        
        if deploy_files and deployment_files:
            # Keep deployment-automation-master as the canonical name
            for deploy_file in deploy_files:
                deploy_file.unlink()
                self.fixes_applied.append({
                    'type': 'duplicate_removed',
                    'file': deploy_file.name,
                    'action': 'Removed in favor of deployment-automation-master'
                })
    
    def fix_naming_issues(self):
        """Fix file naming convention issues"""
        for file_path in self.agents_dir.glob("*"):
            if file_path.is_file():
                filename = file_path.name
                new_name = None
                
                # Convert camelCase or PascalCase to kebab-case
                if re.search(r'[A-Z]', filename) and not filename.startswith(('AGENT_', 'COMPREHENSIVE_')):
                    # Convert to kebab-case
                    new_name = re.sub(r'(?<!^)(?=[A-Z])', '-', filename).lower()
                    new_name = re.sub(r'_', '-', new_name)
                
                if new_name and new_name != filename:
                    new_path = file_path.parent / new_name
                    if not new_path.exists():
                        file_path.rename(new_path)
                        self.fixes_applied.append({
                            'type': 'naming_fixed',
                            'old_name': filename,
                            'new_name': new_name
                        })
    
    def remove_fantasy_elements(self):
        """Remove fantasy elements from agent files"""
        replacements = {
            'wizard': 'expert',
            'magic': 'advanced',
            'spell': 'command',
            'enchant': 'enhance',
            'dragon': 'system',
            'mythical': 'advanced',
            'fantasy': 'system',
            'supernatural': 'advanced',
            'mystical': 'sophisticated',
            'sorcerer': 'specialist',
            'mage': 'engineer',
            'alchemy': 'optimization',
            'arcane': 'specialized',
            'celestial': 'distributed',
            'divine': 'optimal',
            'ethereal': 'lightweight',
            'necromancer': 'recovery specialist',
            'warlock': 'security specialist',
            'enchantment': 'enhancement',
            'magical': 'advanced'
        }
        
        for file_path in self.agents_dir.glob("*.md"):
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            original_content = content
            
            for term, replacement in replacements.items():
                # Case-insensitive replacement while preserving case
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                
                def replace_func(match):
                    original = match.group(0)
                    if original.isupper():
                        return replacement.upper()
                    elif original[0].isupper():
                        return replacement.capitalize()
                    else:
                        return replacement
                
                content = pattern.sub(replace_func, content)
            
            if content != original_content:
                file_path.write_text(content, encoding='utf-8')
                self.fixes_applied.append({
                    'type': 'fantasy_removed',
                    'file': file_path.name,
                    'action': 'Removed fantasy elements'
                })
    
    def standardize_structure(self):
        """Ensure all agent files have consistent structure"""
        template = """---
name: {name}
description: |
  {description}
model: tinyllama:latest
version: 1.0
capabilities:
  - capability1
  - capability2
integrations:
  systems: []
  frameworks: []
  languages: []
  tools: []
performance:
  metric1: value1
  metric2: value2
---

{content}
"""
        
        for file_path in self.agents_dir.glob("*.md"):
            if file_path.name.startswith(('AGENT_', 'COMPREHENSIVE_')) or file_path.name.endswith('.txt'):
                continue
            
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            # Skip if already has proper structure
            if content.strip().startswith('---') and content.count('---') >= 2:
                continue
            
            # Extract agent name from filename
            agent_name = file_path.stem.replace('-detailed', '').replace('-', ' ').title()
            
            # Try to extract description from content
            description = "Professional agent for specialized tasks"
            if 'description:' in content:
                desc_match = re.search(r'description:\s*(.+?)(?=\n|$)', content, re.MULTILINE)
                if desc_match:
                    description = desc_match.group(1).strip()
            
            # Create standardized content
            new_content = template.format(
                name=file_path.stem,
                description=description,
                content=content
            )
            
            file_path.write_text(new_content, encoding='utf-8')
            self.fixes_applied.append({
                'type': 'structure_fixed',
                'file': file_path.name,
                'action': 'Standardized file structure'
            })
    
    def generate_report(self):
        """Generate compliance report"""
        report = {
            'total_files': len(list(self.agents_dir.glob("*"))),
            'issues_found': len(self.issues),
            'fixes_applied': len(self.fixes_applied),
            'issues': self.issues,
            'fixes': self.fixes_applied
        }
        
        # Save report
        report_path = self.agents_dir / 'AGENT_STANDARDS_COMPLIANCE_REPORT.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n📊 Compliance Check Summary:")
        print(f"Total files checked: {report['total_files']}")
        print(f"Issues found: {report['issues_found']}")
        print(f"Fixes applied: {report['fixes_applied']}")
        
        if self.issues:
            print("\n⚠️ Remaining Issues:")
            for issue in self.issues[:5]:  # Show first 5 issues
                print(f"  - {issue.get('file', issue.get('files', 'Unknown'))}: {issue['issue']}")
            if len(self.issues) > 5:
                print(f"  ... and {len(self.issues) - 5} more")
        
        if self.fixes_applied:
            print("\n✅ Fixes Applied:")
            for fix in self.fixes_applied[:5]:  # Show first 5 fixes
                print(f"  - {fix.get('file', fix.get('old_name', 'Unknown'))}: {fix['action']}")
            if len(self.fixes_applied) > 5:
                print(f"  ... and {len(self.fixes_applied) - 5} more")
        
        print(f"\n📄 Full report saved to: {report_path}")
        
        return report


if __name__ == "__main__":
    enforcer = AgentStandardsEnforcer()
    enforcer.run_full_compliance_check()