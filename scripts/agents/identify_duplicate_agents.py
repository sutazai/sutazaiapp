#!/usr/bin/env python3
"""
Purpose: Identify and document duplicate agent definitions
Usage: python identify_duplicate_agents.py
Requirements: Python 3.8+
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Set
import json

def analyze_duplicate_agents(agents_dir: str) -> Dict:
    """Analyze duplicate agents and their relationships"""
    
    agents_path = Path(agents_dir)
    agent_files = list(agents_path.glob('*.md'))
    
    # Filter out non-agent files
    agent_files = [f for f in agent_files if not f.name.startswith(('AGENT_', 'COMPLETE_', 'COMPREHENSIVE_'))]
    
    # Group agents by base name
    agent_groups = {}
    for agent_file in agent_files:
        base_name = agent_file.stem.replace('-detailed', '')
        if base_name not in agent_groups:
            agent_groups[base_name] = []
        agent_groups[base_name].append(agent_file)
    
    # Find duplicates
    duplicates = {name: files for name, files in agent_groups.items() if len(files) > 1}
    
    # Analyze each duplicate
    duplicate_analysis = {}
    for base_name, files in duplicates.items():
        analysis = {
            'files': [f.name for f in files],
            'regular_version': None,
            'detailed_version': None,
            'content_comparison': {}
        }
        
        for file in files:
            with open(file, 'r') as f:
                content = f.read()
                
            # Check if it's detailed version
            if '-detailed' in file.name:
                analysis['detailed_version'] = {
                    'filename': file.name,
                    'size': len(content),
                    'has_yaml': content.startswith('---')
                }
            else:
                analysis['regular_version'] = {
                    'filename': file.name,
                    'size': len(content),
                    'has_yaml': content.startswith('---')
                }
                
            # Extract YAML if present
            if content.startswith('---'):
                yaml_match = content.split('---')[1] if len(content.split('---')) > 2 else ''
                try:
                    yaml_data = yaml.safe_load(yaml_match) if yaml_match else {}
                    analysis['content_comparison'][file.name] = {
                        'has_yaml': True,
                        'name': yaml_data.get('name', 'N/A'),
                        'model': yaml_data.get('model', 'N/A'),
                        'version': yaml_data.get('version', 'N/A')
                    }
                except:
                    analysis['content_comparison'][file.name] = {'has_yaml': False}
            else:
                analysis['content_comparison'][file.name] = {'has_yaml': False}
                
        duplicate_analysis[base_name] = analysis
    
    return {
        'total_agents': len(agent_files),
        'unique_agents': len(agent_groups),
        'duplicate_pairs': len(duplicates),
        'duplicates': duplicate_analysis
    }

def generate_duplicate_report(analysis: Dict) -> str:
    """Generate a report on duplicate agents"""
    
    report = f"""# Duplicate Agent Analysis Report

## Summary
- Total agent files: {analysis['total_agents']}
- Unique agent names: {analysis['unique_agents']}
- Duplicate pairs found: {analysis['duplicate_pairs']}

## Duplicate Agent Pairs

The following agents have both regular and detailed versions:

"""
    
    for agent_name, data in sorted(analysis['duplicates'].items()):
        report += f"\n### {agent_name}\n"
        report += f"- Files: {', '.join(data['files'])}\n"
        
        if data['regular_version']:
            reg = data['regular_version']
            report += f"- Regular version: {reg['filename']} ({reg['size']} bytes)\n"
            
        if data['detailed_version']:
            det = data['detailed_version']
            report += f"- Detailed version: {det['filename']} ({det['size']} bytes)\n"
            
        # Compare content
        if len(data['content_comparison']) == 2:
            reg_file = [f for f in data['files'] if not '-detailed' in f][0]
            det_file = [f for f in data['files'] if '-detailed' in f][0]
            
            if reg_file in data['content_comparison'] and det_file in data['content_comparison']:
                reg_info = data['content_comparison'][reg_file]
                det_info = data['content_comparison'][det_file]
                
                if reg_info['has_yaml'] and det_info['has_yaml']:
                    if reg_info['name'] == det_info['name']:
                        report += f"- ⚠️ Both versions have same agent name: {reg_info['name']}\n"
                    else:
                        report += f"- ✅ Different agent names: {reg_info['name']} vs {det_info['name']}\n"
                        
    report += "\n## Recommendations\n\n"
    report += "1. **Consolidate duplicate agents**: Each agent should have only one definition\n"
    report += "2. **Use consistent naming**: If detailed versions are needed, ensure they have different agent names\n"
    report += "3. **Document the purpose**: Clearly indicate why both versions exist if they're both necessary\n"
    report += "4. **Follow naming conventions**: Use either regular OR detailed, not both\n"
    
    return report

def main():
    agents_dir = '/opt/sutazaiapp/.claude/agents'
    
    print("Analyzing duplicate agents...")
    analysis = analyze_duplicate_agents(agents_dir)
    
    # Generate report
    report = generate_duplicate_report(analysis)
    
    # Save report
    report_path = Path(agents_dir) / 'DUPLICATE_AGENTS_REPORT.md'
    with open(report_path, 'w') as f:
        f.write(report)
        
    # Save JSON analysis
    json_path = Path(agents_dir) / 'duplicate_agents_analysis.json'
    with open(json_path, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
        
    print(report)
    print(f"\nReports saved to:")
    print(f"- {report_path}")
    print(f"- {json_path}")

if __name__ == '__main__':
    main()