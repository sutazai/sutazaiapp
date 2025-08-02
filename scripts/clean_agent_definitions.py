#!/usr/bin/env python3
"""
Clean up agent definitions by removing fantasy elements and focusing on production-ready implementation.
"""

import os
import re
import glob
import json

# Fantasy terms to replace or remove
FANTASY_REPLACEMENTS = {
    # Exaggerated numbers
    r'40\+ (agents|AI systems)': 'multiple agents',
    r'(\d+)\+ agents': 'multiple agents',
    
    # Fantasy terminology
    r'intelligence evolution': 'system improvement',
    r'intelligence space': 'capability domain',
    r'system optimization': 'performance optimization',
    r'advanced AI system': 'AI system',
    r'sophisticated goal decomposition': 'task decomposition',
    r'distributed processing': 'parallel processing',
    r'meta-learning': 'continuous learning',
    r'swarm intelligence': 'distributed coordination',
    r'consciousness modeling': 'state management',
    r'emergent intelligence': 'adaptive behavior',
    r'neural plasticity': 'model adaptation',
    r'cognitive architecture': 'system architecture',
    r'intelligence cores': 'processing units',
    r'synaptic': 'connection',
    r'neural homeostasis': 'system stability',
    r'consciousness': 'state awareness',
    
    # Overly ambitious claims
    r'pursue system optimization independently': 'execute tasks autonomously',
    r'evolve.*intelligence': 'improve performance',
    r'self-improving': 'continuously optimizing',
    r'goal-driven automation system systems': 'goal-driven automation',
    r'autonomous exploration of.*': 'systematic analysis of',
    r'optimized behavior detection': 'behavior monitoring',
    
    # Redundant phrases
    r'automation system system': 'automation system',
    r'system system': 'system',
}

def clean_agent_file(filepath):
    """Clean a single agent definition file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Apply replacements
    for pattern, replacement in FANTASY_REPLACEMENTS.items():
        content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
    
    # Remove duplicate model names (e.g., "tinyllama, tinyllama")
    content = re.sub(r'(tinyllama),\s*\1', r'\1', content)
    
    # Fix description formatting
    content = re.sub(r'SutazAI advanced AI system', 'SutazAI system', content)
    content = re.sub(r'SutazAI\'s advanced AI system', 'SutazAI system', content)
    
    # Only write if content changed
    if content != original_content:
        # Backup original
        backup_path = filepath + '.fantasy_cleanup_backup'
        if not os.path.exists(backup_path):
            with open(backup_path, 'w') as f:
                f.write(original_content)
        
        # Write cleaned content
        with open(filepath, 'w') as f:
            f.write(content)
        
        return True
    return False

def main():
    """Clean all agent definition files."""
    agent_dir = '/opt/sutazaiapp/.claude/agents'
    
    # Find all .md files (excluding backups)
    agent_files = glob.glob(os.path.join(agent_dir, '*.md'))
    agent_files = [f for f in agent_files if not f.endswith('.backup') and not f.endswith('.fantasy_backup')]
    
    cleaned_count = 0
    
    print(f"Cleaning {len(agent_files)} agent definition files...")
    
    for filepath in sorted(agent_files):
        filename = os.path.basename(filepath)
        if clean_agent_file(filepath):
            print(f"  ✓ Cleaned: {filename}")
            cleaned_count += 1
        else:
            print(f"  - No changes: {filename}")
    
    print(f"\nCleaned {cleaned_count} files")
    
    # Also remove old fantasy backup files
    fantasy_backups = glob.glob(os.path.join(agent_dir, '*.fantasy_backup'))
    if fantasy_backups:
        print(f"\nRemoving {len(fantasy_backups)} old fantasy backup files...")
        for backup in fantasy_backups:
            os.remove(backup)
            print(f"  ✗ Removed: {os.path.basename(backup)}")

if __name__ == '__main__':
    main()