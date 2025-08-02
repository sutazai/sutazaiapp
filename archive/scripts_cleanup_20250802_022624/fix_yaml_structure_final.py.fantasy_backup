#!/usr/bin/env python3
"""
Fix YAML structure where some YAML fields appear after what should be body content.
"""

import os
import re
from pathlib import Path
import yaml

def fix_yaml_structure(file_path):
    """Fix YAML structure by moving misplaced YAML fields."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    # Find the YAML frontmatter boundaries
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
    
    # Extract sections
    yaml_lines = lines[yaml_start+1:yaml_end]
    body_lines = lines[yaml_end+1:]
    
    # Look for YAML fields in the body
    yaml_field_pattern = re.compile(r'^(model|version|capabilities|tools):\s*(.*)$')
    
    additional_yaml_lines = []
    new_body_lines = []
    found_yaml_in_body = False
    
    for line in body_lines:
        if yaml_field_pattern.match(line) or (found_yaml_in_body and line.startswith('  - ')):
            # This is a YAML field that should be in frontmatter
            additional_yaml_lines.append(line)
            found_yaml_in_body = True
        else:
            # Check if we just finished extracting YAML fields
            if found_yaml_in_body and line.strip() == '':
                found_yaml_in_body = False
                continue  # Skip this empty line
            new_body_lines.append(line)
    
    # Combine YAML sections
    # First, find where the description ends in the original YAML
    desc_end_index = -1
    in_description = False
    
    for i, line in enumerate(yaml_lines):
        if line.startswith('description:'):
            in_description = True
        elif in_description and line and not line.startswith(' ') and not line.startswith('-'):
            desc_end_index = i
            break
    
    # If we found additional YAML fields, insert them before any existing fields after description
    if additional_yaml_lines:
        if desc_end_index > 0:
            # Insert after description block
            final_yaml_lines = yaml_lines[:desc_end_index] + [''] + additional_yaml_lines + yaml_lines[desc_end_index:]
        else:
            # Append at end
            final_yaml_lines = yaml_lines + [''] + additional_yaml_lines
    else:
        final_yaml_lines = yaml_lines
    
    # Clean up the final YAML
    final_yaml_lines = [line for line in final_yaml_lines if line.strip() != '' or (i > 0 and i < len(final_yaml_lines)-1)]
    
    # Reconstruct content
    yaml_content = '\n'.join(final_yaml_lines)
    body_content = '\n'.join(new_body_lines).strip()
    
    # Ensure proper spacing
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
    
    return True, "Fixed YAML structure"

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
        'neural-architecture-search.md',
        'edge-computing-optimizer.md',
        'knowledge-graph-builder.md',
        'data-pipeline-engineer.md',
        'langflow-workflow-designer.md',
        'quantum-computing-optimizer.md',
        'localagi-orchestration-manager.md',
        'bigagi-system-manager.md',
        'deployment-automation-master.md',
        'flowiseai-flow-manager.md',
        'agentzero-coordinator.md',
        'transformers-migration-specialist.md',
        'ai-product-manager.md',
        'deep-learning-brain-architect.md',
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
        success, message = fix_yaml_structure(file_path)
        
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