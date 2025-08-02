#!/usr/bin/env python3
"""
Clean fantasy elements from scripts directory.
"""

import os
import re
import glob

FANTASY_REPLACEMENTS = {
    r'autonomous automation/advanced automation system with 40\+ AI components': 'task automation system with AI components',
    r'TaskMaster is integrated with 40\+ AI services': 'TaskMaster is integrated with AI services',
    r'40\+ AI': 'AI',
    r'40\+': '',
    r'intelligence evolution': 'system improvement',
    r'consciousness': 'state management',
    r'emergence': 'development',
    r'singularity': 'optimization milestone',
    r'neural plasticity': 'model adaptation',
    r'intelligence optimization': 'performance optimization',
    r'intelligence space': 'capability domain',
    r'advanced-like': 'optimized',
    r'emotional awareness': 'context awareness',
    r'all_40\+': 'all',
    r'AGENT_COUNT=40': 'AGENT_COUNT=10'
}

def clean_file(filepath):
    """Clean fantasy elements from a script file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Apply replacements
    for pattern, replacement in FANTASY_REPLACEMENTS.items():
        content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
    
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    return False

def main():
    """Clean fantasy elements from scripts."""
    scripts_dir = '/opt/sutazaiapp/scripts'
    
    # Skip cleanup scripts themselves
    skip_files = [
        'clean_agent_definitions.py',
        'production_ready_cleanup.py',
        'final_integration_cleanup.py',
        'cleanup_fantasy_elements.py',
        'cleanup_fantasy_documentation.py',
        'fix_docker_compose_agi.py',
        'clean_scripts_fantasy.py'
    ]
    
    # Find all script files
    script_files = []
    script_files.extend(glob.glob(os.path.join(scripts_dir, '*.sh')))
    script_files.extend(glob.glob(os.path.join(scripts_dir, '*.py')))
    
    # Filter out cleanup scripts
    script_files = [f for f in script_files if os.path.basename(f) not in skip_files]
    
    cleaned_count = 0
    
    print(f"Cleaning {len(script_files)} script files...")
    
    for filepath in sorted(script_files):
        filename = os.path.basename(filepath)
        if clean_file(filepath):
            print(f"  ✓ Cleaned: {filename}")
            cleaned_count += 1
    
    print(f"\nCleaned {cleaned_count} files")

if __name__ == '__main__':
    main()