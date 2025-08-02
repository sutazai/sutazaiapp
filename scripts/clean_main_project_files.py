#!/usr/bin/env python3
"""
Clean fantasy elements from main project documentation files.
"""

import os
import re
import glob

FANTASY_REPLACEMENTS = {
    r'All 40\+ agents': 'All agents',
    r'40\+ agents': 'agents',
    r'40\+ AI agents': 'AI agents',
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
    """Clean fantasy elements from a file."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"  ✗ Error reading {filepath}: {e}")
        return False
    
    original_content = content
    
    # Apply replacements
    for pattern, replacement in FANTASY_REPLACEMENTS.items():
        content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
    
    if content != original_content:
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            print(f"  ✗ Error writing {filepath}: {e}")
            return False
    return False

def main():
    """Clean fantasy elements from main project files."""
    root_dir = '/opt/sutazaiapp'
    
    # Files to check
    files_to_check = [
        'IMPLEMENTATION.md',
        'README.md',
        'COMPREHENSIVE_DEPLOYMENT_ARCHITECTURE.md',
        'COMPREHENSIVE_DEPLOYMENT_INSTALLATION_GUIDE.md',
        'COMPREHENSIVE_TESTING_VALIDATION_REPORT.md',
        'PRODUCT_REQUIREMENTS_DOCUMENT.md',
        'SUTAZAI_PRODUCT_REQUIREMENTS_DOCUMENT.md',
        'PRODUCTION_DEPLOYMENT_GUIDE.md',
        'OPERATIONAL_RUNBOOK.md',
        'OPERATIONAL_RUNBOOK_COMPLETE.md',
        'RESOURCE_MANAGEMENT_SCALING_GUIDE.md',
        'FINAL_SYSTEM_SUMMARY.md'
    ]
    
    # Also check docs directory
    docs_files = glob.glob(os.path.join(root_dir, 'docs', '*.md'))
    docs_files.extend(glob.glob(os.path.join(root_dir, 'docs', '**', '*.md')))
    
    all_files = [os.path.join(root_dir, f) for f in files_to_check if os.path.exists(os.path.join(root_dir, f))]
    all_files.extend(docs_files)
    
    # Remove duplicates
    all_files = list(set(all_files))
    
    cleaned_count = 0
    
    print(f"Cleaning {len(all_files)} documentation files...")
    
    for filepath in sorted(all_files):
        filename = os.path.relpath(filepath, root_dir)
        if os.path.exists(filepath) and clean_file(filepath):
            print(f"  ✓ Cleaned: {filename}")
            cleaned_count += 1
    
    print(f"\nCleaned {cleaned_count} files")

if __name__ == '__main__':
    main()