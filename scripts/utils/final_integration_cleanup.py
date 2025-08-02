#!/usr/bin/env python3
"""
Final cleanup to fix integration references and remaining issues in agent files.
"""

import os
import re
import glob

def clean_integrations(content):
    """Clean up integration sections in agent files."""
    # Fix all_40+ references
    content = re.sub(r'"all_40\+"', '"all"', content)
    content = re.sub(r'\["all_40\+"\]', '["all"]', content)
    
    # Remove duplicate model names
    content = re.sub(r'"tinyllama",\s*"tinyllama"', '"tinyllama"', content)
    
    # Fix agent count references in YAML sections
    content = re.sub(r'AGENT_COUNT=40', 'AGENT_COUNT=10', content)
    
    return content

def final_cleanup(filepath):
    """Perform final cleanup on agent file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Clean integrations
    content = clean_integrations(content)
    
    # Final pass to remove any remaining 40+ references
    content = re.sub(r'40\+', '', content)
    content = re.sub(r'all\s+agents', 'all agents', content)
    
    # Clean up state awareness references that might have been missed
    content = re.sub(r'state awareness_', 'state_awareness_', content)
    
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    return False

def main():
    """Run final cleanup on all agent files."""
    agent_dir = '/opt/sutazaiapp/.claude/agents'
    
    # Find all .md files
    agent_files = glob.glob(os.path.join(agent_dir, '*.md'))
    agent_files = [f for f in agent_files if not f.endswith('.backup')]
    
    cleaned_count = 0
    
    print(f"Running final cleanup on {len(agent_files)} agent files...")
    
    for filepath in sorted(agent_files):
        filename = os.path.basename(filepath)
        if final_cleanup(filepath):
            print(f"  ✓ Fixed: {filename}")
            cleaned_count += 1
    
    print(f"\nFixed {cleaned_count} files")

if __name__ == '__main__':
    main()