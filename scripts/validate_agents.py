#!/usr/bin/env python3
"""
Validate agent YAML frontmatter format
Part of CLAUDE.md hygiene enforcement
"""

import os
import re
import sys
import yaml
from pathlib import Path

REQUIRED_FIELDS = ['name', 'model', 'temperature']
VALID_MODELS = [
    'gpt-oss'
]

def extract_yaml_frontmatter(content):
    """Extract YAML frontmatter from markdown content"""
    match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
    if match:
        return match.group(1)
    return None

def validate_agent_file(filepath):
    """Validate a single agent file"""
    violations = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for YAML frontmatter
        yaml_content = extract_yaml_frontmatter(content)
        if not yaml_content:
            violations.append("Missing YAML frontmatter (should start with ---)")
            return violations
        
        # Parse YAML
        try:
            data = yaml.safe_load(yaml_content)
        except yaml.YAMLError as e:
            violations.append(f"Invalid YAML: {e}")
            return violations
        
        # Check required fields
        for field in REQUIRED_FIELDS:
            if field not in data:
                violations.append(f"Missing required field: {field}")
        
        # Validate field values
        if 'name' in data:
            name = data['name']
            # Name should match filename (without .md)
            expected_name = os.path.splitext(os.path.basename(filepath))[0]
            if name != expected_name:
                violations.append(f"Name '{name}' doesn't match filename '{expected_name}'")
        
        if 'model' in data:
            model = data['model']
            if model not in VALID_MODELS:
                violations.append(f"Invalid model '{model}'. Valid models: {', '.join(VALID_MODELS)}")
        
        if 'temperature' in data:
            temp = data['temperature']
            try:
                temp_float = float(temp)
                if not 0 <= temp_float <= 2:
                    violations.append(f"Temperature {temp} out of range (0-2)")
            except ValueError:
                violations.append(f"Invalid temperature value: {temp}")
        
        # Check for consistent formatting
        lines = content.split('\n')
        if len(lines) > 0 and not lines[0] == '---':
            violations.append("File should start with --- for YAML frontmatter")
        
        # Check description after frontmatter
        content_after_yaml = content.split('---', 2)
        if len(content_after_yaml) > 2:
            main_content = content_after_yaml[2].strip()
            if not main_content:
                violations.append("No agent description after frontmatter")
        
    except Exception as e:
        violations.append(f"Error reading file: {e}")
    
    return violations

def main():
    """Validate all agent files"""
    agent_dir = Path('.claude/agents')
    if not agent_dir.exists():
        print("No .claude/agents directory found")
        return 0
    
    all_violations = []
    
    for filepath in agent_dir.glob('*.md'):
        # Skip backup files
        if any(pattern in str(filepath) for pattern in ['backup', 'fantasy', 'old']):
            all_violations.append((str(filepath), ["Backup/old file should not exist"]))
            continue
        
        violations = validate_agent_file(filepath)
        if violations:
            all_violations.append((str(filepath), violations))
    
    if all_violations:
        print("ERROR: Agent validation failed!")
        print("-" * 60)
        for filepath, violations in all_violations:
            print(f"\n{filepath}:")
            for v in violations:
                print(f"  - {v}")
        print("-" * 60)
        print(f"\nTotal files with violations: {len(all_violations)}")
        print("\nAgent files must have proper YAML frontmatter with:")
        print("  - name: agent-name")
        print("  - model: valid-model-name")
        print("  - temperature: 0-2")
        return 1
    
    print("✓ All agent files validated successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())