#!/usr/bin/env python3
"""
Fix files that are missing the closing YAML delimiter.
These files have frontmatter content but then have body content without the closing ---.
"""

import os
import re
from pathlib import Path
import yaml

def fix_missing_delimiter(file_path):
    """Fix files missing the closing --- delimiter."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if file starts with frontmatter
    if not content.startswith('---'):
        return False, "No frontmatter"
    
    # Split by --- to check structure
    parts = content.split('---', 2)
    
    if len(parts) >= 3:
        # Already has closing delimiter
        return False, "Already has closing delimiter"
    
    # File is missing closing delimiter
    lines = content.split('\n')
    yaml_lines = []
    body_lines = []
    in_yaml = True
    
    # Skip the first --- line
    for i, line in enumerate(lines[1:], 1):
        if in_yaml:
            # Check if this line looks like it should be body content
            # These patterns indicate we've left the YAML section
            body_indicators = [
                'This agent specializes',
                'This agent manages',
                'This agent masters',
                'This agent creates',
                'This agent builds',
                'You are',
                '## ',
                '### ',
            ]
            
            if any(line.strip().startswith(indicator) for indicator in body_indicators):
                # This is the start of body content
                in_yaml = False
                body_lines = lines[i:]  # Include this line and all following
            else:
                yaml_lines.append(line)
        else:
            # Should not reach here in first pass
            body_lines.append(line)
    
    # Reconstruct with proper delimiter
    yaml_content = '\n'.join(yaml_lines).rstrip()
    body_content = '\n'.join(body_lines)
    
    # Ensure body has proper spacing
    if body_content and not body_content.startswith('\n'):
        body_content = '\n' + body_content
    
    fixed_content = f"---\n{yaml_content}\n---{body_content}"
    
    # Validate YAML
    try:
        parts = fixed_content.split('---', 2)
        yaml.safe_load(parts[1])
    except Exception as e:
        return False, f"YAML validation failed: {e}"
    
    # Write back
    with open(file_path, 'w') as f:
        f.write(fixed_content)
    
    return True, "Fixed missing delimiter"

def main():
    """Process all agent files that need delimiter fixes."""
    agents_dir = Path('/opt/sutazaiapp/.claude/agents')
    
    # List of files that failed in the previous script
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
        success, message = fix_missing_delimiter(file_path)
        
        if success:
            print(f"  ✓ {message}")
            fixed_count += 1
        else:
            print(f"  ✗ {message}")
            failed_count += 1
    
    print(f"\n\n=== Summary ===")
    print(f"Fixed: {fixed_count} files")
    print(f"Failed: {failed_count} files")

if __name__ == '__main__':
    main()