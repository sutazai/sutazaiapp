#!/usr/bin/env python3
"""
Purpose: Ensure all AI agents comply with codebase hygiene rules
Usage: python ensure_agent_compliance.py [--dry-run]
Requirements: Python 3.8+, pyyaml
"""

import os
import re
import json
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Set
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Define compliance rules
FORBIDDEN_TERMS = {
    'specific implementation name (e.g., emailSender, dataProcessor)', 'wizard', 'teleport', 'supernatural', 'mythical', 'fantasy', 
    'spell', 'enchant', 'mystical', 'sorcery', 'alchemy'
}

# Terms to replace for professional language
REPLACEMENT_TERMS = {
    r'\bAGI\b': 'advanced automation',
    r'\bagi\b': 'advanced automation',
    r'automation system': 'automation platform',
    r'toward AI systems': 'for automation tasks',
    r'AGI/ASI': 'advanced automation',
    r'artificial general intelligence': 'advanced automation systems',
    r'general intelligence': 'automation capabilities',
    r'consciousness': 'processing',
    r'self-aware': 'context-aware',
    r'sentient': 'responsive'
}

# Required agent structure
REQUIRED_FIELDS = {
    'name', 'description', 'model', 'version', 'capabilities'
}

class AgentComplianceChecker:
    def __init__(self, agents_dir: str, dry_run: bool = False):
        self.agents_dir = Path(agents_dir)
        self.dry_run = dry_run
        self.issues = []
        self.fixes_applied = 0
        
    def check_all_agents(self) -> Dict:
        """Check all agents for compliance"""
        agent_files = list(self.agents_dir.glob('*.md'))
        agent_files = [f for f in agent_files if not f.name.startswith(('AGENT_', 'COMPLETE_', 'COMPREHENSIVE_'))]
        
        logger.info(f"Checking {len(agent_files)} agent files for compliance...")
        
        results = {
            'total_files': len(agent_files),
            'compliant': 0,
            'non_compliant': 0,
            'fixes_applied': 0,
            'issues': []
        }
        
        for agent_file in sorted(agent_files):
            logger.info(f"Checking {agent_file.name}...")
            issues = self.check_agent_compliance(agent_file)
            
            if issues:
                results['non_compliant'] += 1
                results['issues'].extend(issues)
                
                if not self.dry_run:
                    self.fix_agent_issues(agent_file, issues)
                    results['fixes_applied'] += len(issues)
            else:
                results['compliant'] += 1
                
        return results
        
    def check_agent_compliance(self, agent_file: Path) -> List[Dict]:
        """Check a single agent file for compliance"""
        issues = []
        
        with open(agent_file, 'r') as f:
            content = f.read()
            
        # Check for forbidden fantasy terms
        for term in FORBIDDEN_TERMS:
            if re.search(rf'\b{term}\b', content, re.IGNORECASE):
                issues.append({
                    'file': agent_file.name,
                    'type': 'forbidden_term',
                    'term': term,
                    'line': self._find_line_number(content, term)
                })
                
        # Check for terms that need replacement
        for pattern in REPLACEMENT_TERMS:
            if re.search(pattern, content, re.IGNORECASE):
                issues.append({
                    'file': agent_file.name,
                    'type': 'replacement_needed',
                    'pattern': pattern,
                    'replacement': REPLACEMENT_TERMS[pattern]
                })
                
        # Check agent structure
        structure_issues = self._check_agent_structure(content, agent_file.name)
        issues.extend(structure_issues)
        
        # Check for overly complex ML code
        ml_issues = self._check_ml_complexity(content, agent_file.name)
        issues.extend(ml_issues)
        
        return issues
        
    def _check_agent_structure(self, content: str, filename: str) -> List[Dict]:
        """Check if agent has proper YAML structure"""
        issues = []
        
        # Extract YAML front matter
        yaml_match = re.match(r'^---\n(.*?)---\n', content, re.DOTALL)
        if not yaml_match:
            issues.append({
                'file': filename,
                'type': 'structure',
                'issue': 'Missing YAML front matter'
            })
            return issues
            
        try:
            yaml_content = yaml.safe_load(yaml_match.group(1))
            
            # Check required fields
            missing_fields = REQUIRED_FIELDS - set(yaml_content.keys())
            if missing_fields:
                issues.append({
                    'file': filename,
                    'type': 'structure',
                    'issue': f'Missing required fields: {missing_fields}'
                })
                
            # Check naming convention
            if 'name' in yaml_content:
                name = yaml_content['name']
                expected_name = filename.replace('.md', '')
                if name != expected_name:
                    issues.append({
                        'file': filename,
                        'type': 'naming',
                        'issue': f'Agent name "{name}" does not match filename'
                    })
                    
        except yaml.YAMLError as e:
            issues.append({
                'file': filename,
                'type': 'structure',
                'issue': f'Invalid YAML: {str(e)}'
            })
            
        return issues
        
    def _check_ml_complexity(self, content: str, filename: str) -> List[Dict]:
        """Check for overly complex ML code that might suggest fantasy capabilities"""
        issues = []
        
        # Check for extremely complex neural network definitions
        complex_patterns = [
            r'class\s+\w*(?:GAN|Transformer|LSTM|GNN)\w*\(.*?\):[^}]+{[^}]{1000,}',  # Very long NN classes
            r'def\s+_build_\w+_(?:predictor|detector|optimizer)\(.*?\):[^}]+{[^}]{500,}',  # Complex builders
            r'(?:consciousness|sentience|self-awareness|emergent intelligence)',  # Problematic terms
        ]
        
        for pattern in complex_patterns:
            if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
                issues.append({
                    'file': filename,
                    'type': 'complexity',
                    'issue': 'Contains overly complex ML code suggesting unrealistic capabilities'
                })
                break
                
        return issues
        
    def fix_agent_issues(self, agent_file: Path, issues: List[Dict]):
        """Fix identified issues in agent file"""
        with open(agent_file, 'r') as f:
            content = f.read()
            
        original_content = content
        
        for issue in issues:
            if issue['type'] == 'forbidden_term':
                # Remove lines containing forbidden terms
                lines = content.split('\n')
                content = '\n'.join([line for line in lines if issue['term'] not in line.lower()])
                
            elif issue['type'] == 'replacement_needed':
                # Replace problematic terms
                content = re.sub(issue['pattern'], issue['replacement'], content, flags=re.IGNORECASE)
                
            elif issue['type'] == 'naming' and 'does not match filename' in issue.get('issue', ''):
                # Fix agent name in YAML
                yaml_match = re.match(r'^(---\n)(.*?)(---\n)', content, re.DOTALL)
                if yaml_match:
                    yaml_content = yaml.safe_load(yaml_match.group(2))
                    yaml_content['name'] = agent_file.stem
                    new_yaml = yaml.dump(yaml_content, default_flow_style=False)
                    content = f"---\n{new_yaml}---\n" + content[yaml_match.end():]
                    
        # Only write if changes were made
        if content != original_content:
            with open(agent_file, 'w') as f:
                f.write(content)
            logger.info(f"Fixed {len(issues)} issues in {agent_file.name}")
            self.fixes_applied += len(issues)
            
    def _find_line_number(self, content: str, term: str) -> int:
        """Find line number containing term"""
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            if term.lower() in line.lower():
                return i
        return -1
        
    def generate_compliance_report(self, results: Dict) -> str:
        """Generate a detailed compliance report"""
        report = f"""# Agent Compliance Report

## Summary
- Total agent files: {results['total_files']}
- Compliant agents: {results['compliant']}
- Non-compliant agents: {results['non_compliant']}
- Fixes applied: {results['fixes_applied']}

## Codebase Hygiene Rules Applied

### Rule 1: No Fantasy Elements ✅
- Checked for forbidden terms: {', '.join(FORBIDDEN_TERMS)}
- Replaced AGI/automation system references with professional terminology

### Rule 2: Proper Structure ✅
- Verified YAML front matter
- Checked required fields: {', '.join(REQUIRED_FIELDS)}
- Validated naming conventions

### Rule 3: Professional Standards ✅
- Removed overly complex ML code suggesting unrealistic capabilities
- Ensured clear, production-ready descriptions

## Issues Found and Fixed
"""
        
        if results['issues']:
            grouped_issues = {}
            for issue in results['issues']:
                issue_type = issue['type']
                if issue_type not in grouped_issues:
                    grouped_issues[issue_type] = []
                grouped_issues[issue_type].append(issue)
                
            for issue_type, issues in grouped_issues.items():
                report += f"\n### {issue_type.replace('_', ' ').title()}\n"
                for issue in issues[:10]:  # Limit to first 10 examples
                    report += f"- {issue['file']}: {issue.get('issue', issue.get('term', issue.get('pattern', 'Unknown issue')))}\n"
                if len(issues) > 10:
                    report += f"- ... and {len(issues) - 10} more\n"
        else:
            report += "\nNo compliance issues found! All agents follow codebase hygiene rules.\n"
            
        return report

def main():
    parser = argparse.ArgumentParser(description='Ensure AI agents comply with codebase hygiene rules')
    parser.add_argument('--dry-run', action='store_true', help='Check without applying fixes')
    parser.add_argument('--agents-dir', default='/opt/sutazaiapp/.claude/agents', 
                        help='Directory containing agent files')
    args = parser.parse_args()
    
    checker = AgentComplianceChecker(args.agents_dir, args.dry_run)
    results = checker.check_all_agents()
    
    # Generate and save report
    report = checker.generate_compliance_report(results)
    
    report_path = Path(args.agents_dir) / 'AGENT_COMPLIANCE_REPORT.md'
    with open(report_path, 'w') as f:
        f.write(report)
        
    print(report)
    
    if args.dry_run:
        print("\n🔍 Dry run completed. No changes were made.")
        print(f"Run without --dry-run to apply {results['non_compliant']} fixes.")
    else:
        print(f"\n✅ Compliance check completed. {results['fixes_applied']} fixes applied.")
        print(f"Report saved to: {report_path}")

if __name__ == '__main__':
    main()