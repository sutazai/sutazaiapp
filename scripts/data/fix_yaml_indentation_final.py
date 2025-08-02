#!/usr/bin/env python3
"""
Fix YAML indentation issues where content in description field is not properly indented.
"""

import os
import re
from pathlib import Path
import yaml

def fix_yaml_indentation(file_path):
    """Fix indentation issues in YAML description fields."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    # Find YAML boundaries
    yaml_start = -1
    yaml_end = -1
    
    for i, line in enumerate(lines):
        if line.strip() == '---':
            if yaml_start == -1:
                yaml_start = i
            else:
                yaml_end = i
                break
    
    if yaml_start == -1 or yaml_end == -1:
        return False, "Missing YAML delimiters"
    
    # Process YAML section
    yaml_lines = lines[yaml_start+1:yaml_end]
    body_lines = lines[yaml_end+1:]
    
    fixed_yaml_lines = []
    in_description = False
    i = 0
    
    while i < len(yaml_lines):
        line = yaml_lines[i]
        
        if line.startswith('description: |'):
            fixed_yaml_lines.append(line)
            in_description = True
            i += 1
        elif in_description:
            # Check if this line should end the description
            if (line and not line.startswith(' ') and not line.startswith('-') and 
                ':' in line and not line.startswith('Do NOT use') and 
                not line.startswith('This agent')):
                # This is a new YAML field
                in_description = False
                fixed_yaml_lines.append(line)
                i += 1
            elif line.startswith('Do NOT use this agent for:'):
                # This should be indented
                fixed_yaml_lines.append('  ')  # Empty line before
                fixed_yaml_lines.append('  ' + line)
                i += 1
            elif line.startswith('This agent specializes') or line.startswith('This agent manages') or line.startswith('This agent masters') or line.startswith('This agent creates'):
                # This should be indented as last part of description
                fixed_yaml_lines.append('  ')  # Empty line before
                fixed_yaml_lines.append('  ' + line)
                in_description = False  # This ends the description
                i += 1
            elif line.strip() == '' and in_description:
                # Empty line in description
                fixed_yaml_lines.append('')
                i += 1
            elif line.startswith('- ') and in_description:
                # List item, ensure proper indentation
                fixed_yaml_lines.append('  ' + line)
                i += 1
            elif in_description:
                # Other content in description
                if line.strip():
                    if not line.startswith('  '):
                        fixed_yaml_lines.append('  ' + line)
                    else:
                        fixed_yaml_lines.append(line)
                else:
                    fixed_yaml_lines.append(line)
                i += 1
            else:
                fixed_yaml_lines.append(line)
                i += 1
        else:
            fixed_yaml_lines.append(line)
            i += 1
    
    # Reconstruct content
    yaml_content = '\n'.join(fixed_yaml_lines)
    body_content = '\n'.join(body_lines).strip()
    
    if body_content and not body_content.startswith('\n'):
        body_content = '\n' + body_content
    
    final_content = f"---\n{yaml_content}\n---{body_content}"
    
    # Validate YAML
    try:
        parts = final_content.split('---', 2)
        yaml.safe_load(parts[1])
    except Exception as e:
        return False, f"YAML validation failed: {e}"
    
    # Write back
    with open(file_path, 'w') as f:
        f.write(final_content)
    
    return True, "Fixed YAML indentation"

def main():
    """Process problem files."""
    agents_dir = Path('/opt/sutazaiapp/.claude/agents')
    
    problem_files = [
        'model-training-specialist.md',
        'federated-learning-coordinator.md',
        'intelligence-optimization-monitor.md',
        'senior-ai-engineer.md',
        'agi-system-architect.md',
        'jarvis-voice-interface.md',
        'dify-automation-specialist.md',
        'agentgpt-autonomous-executor.md',
        'processing-architecture-search.md',
        'edge-computing-optimizer.md',
        'knowledge-graph-builder.md',
        'data-pipeline-engineer.md',
        'langflow-workflow-designer.md',
        'advanced-computing-optimizer.md',
        'localagi-orchestration-manager.md',
        'bigagi-system-manager.md',
        'deployment-automation-master.md',
        'flowiseai-flow-manager.md',
        'agentzero-coordinator.md',
        'transformers-migration-specialist.md',
        'ai-product-manager.md',
        'deep-learning-coordinator-architect.md',
        'private-data-analyst.md',
        'ai-agent-orchestrator.md',
        'codebase-team-lead.md',
        'infrastructure-devops-manager.md',
    ]
    
    fixed_count = 0
    failed_count = 0
    
    for filename in problem_files:
        file_path = agents_dir / filename
        if not file_path.exists():
            print(f"File not found: {filename}")
            continue
            
        print(f"\nProcessing {filename}...")
        success, message = fix_yaml_indentation(file_path)
        
        if success:
            print(f"  ✓ {message}")
            fixed_count += 1
            
            # Validate by trying to parse
            with open(file_path, 'r') as f:
                content = f.read()
            try:
                parts = content.split('---', 2)
                yaml.safe_load(parts[1])
                print(f"  ✓ YAML validation passed")
            except Exception as e:
                print(f"  ⚠ YAML validation warning: {e}")
        else:
            print(f"  ✗ {message}")
            failed_count += 1
    
    print(f"\n\n=== Summary ===")
    print(f"Fixed: {fixed_count} files")  
    print(f"Failed: {failed_count} files")

if __name__ == '__main__':
    main()