#!/usr/bin/env python3
"""
ULTRA SCRIPT CONSOLIDATION ANALYSIS TOOL
Purpose: Analyze and categorize all scripts in the codebase for safe consolidation
Author: Ultra System Architect
Date: 2025-08-10
"""

import os
import hashlib
from pathlib import Path
from collections import defaultdict
import json

def get_file_hash(filepath):
    """Generate hash of file content for duplicate detection"""
    try:
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return None

def analyze_scripts():
    """Perform comprehensive script analysis"""
    
    project_root = Path('/opt/sutazaiapp')
    
    # Collections for analysis
    scripts_by_type = defaultdict(list)
    scripts_by_directory = defaultdict(list)
    duplicate_hashes = defaultdict(list)
    small_scripts = []
    obsolete_patterns = ['_old', '_backup', 'deprecated', 'obsolete', '_v1', '_v2', '_copy', '_temp']
    
    # Directories to exclude
    exclude_dirs = {'node_modules', '.git', '__pycache__', 'venv', '.venv'}
    
    # Walk through all files
    for root, dirs, files in os.walk(project_root):
        # Remove excluded directories from walk
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            filepath = Path(root) / file
            
            # Skip if not a script
            if not file.endswith(('.sh', '.py', '.js')):
                continue
                
            # Skip if in excluded paths
            if any(exc in str(filepath) for exc in exclude_dirs):
                continue
            
            # Categorize by type
            ext = filepath.suffix
            scripts_by_type[ext].append(str(filepath))
            
            # Categorize by directory
            rel_path = filepath.relative_to(project_root)
            top_dir = rel_path.parts[0] if rel_path.parts else 'root'
            scripts_by_directory[top_dir].append(str(filepath))
            
            # Check for duplicates by hash
            file_hash = get_file_hash(filepath)
            if file_hash:
                duplicate_hashes[file_hash].append(str(filepath))
            
            # Check if small/stub (less than 10 lines)
            try:
                with open(filepath, 'r') as f:
                    lines = len(f.readlines())
                    if lines < 10:
                        small_scripts.append((str(filepath), lines))
            except:
                pass
    
    # Generate report
    report = {
        'total_scripts': sum(len(scripts) for scripts in scripts_by_type.values()),
        'by_type': {k: len(v) for k, v in scripts_by_type.items()},
        'by_directory': {k: len(v) for k, v in scripts_by_directory.items()},
        'duplicates': sum(1 for hashes in duplicate_hashes.values() if len(hashes) > 1),
        'duplicate_groups': len([h for h in duplicate_hashes.values() if len(h) > 1]),
        'small_scripts': len(small_scripts),
        'samples': {
            'duplicates': [files for files in duplicate_hashes.values() if len(files) > 1][:5],
            'small_scripts': small_scripts[:10]
        }
    }
    
    return report

def categorize_scripts():
    """Categorize scripts by function for consolidation"""
    
    categories = {
        'deployment': [],
        'testing': [],
        'monitoring': [],
        'maintenance': [],
        'security': [],
        'database': [],
        'docker': [],
        'agents': [],
        'utilities': [],
        'initialization': [],
        'validation': []
    }
    
    # Keywords for categorization
    keywords = {
        'deployment': ['deploy', 'rollout', 'release', 'install'],
        'testing': ['test', 'spec', 'validate', 'check', 'verify'],
        'monitoring': ['monitor', 'health', 'status', 'metrics', 'log'],
        'maintenance': ['cleanup', 'backup', 'restore', 'migrate', 'update'],
        'security': ['security', 'auth', 'permission', 'encrypt', 'audit'],
        'database': ['db', 'database', 'postgres', 'redis', 'neo4j', 'sql'],
        'docker': ['docker', 'container', 'compose', 'build', 'image'],
        'agents': ['agent', 'jarvis', 'ai_', 'ollama'],
        'utilities': ['util', 'helper', 'tool', 'convert', 'format'],
        'initialization': ['init', 'setup', 'bootstrap', 'start'],
        'validation': ['validate', 'lint', 'format', 'compliance']
    }
    
    project_root = Path('/opt/sutazaiapp')
    exclude_dirs = {'node_modules', '.git', '__pycache__', 'venv', '.venv'}
    
    uncategorized = []
    
    for root, dirs, files in os.walk(project_root):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            if not file.endswith(('.sh', '.py', '.js')):
                continue
                
            filepath = Path(root) / file
            if any(exc in str(filepath) for exc in exclude_dirs):
                continue
            
            categorized = False
            filename_lower = file.lower()
            
            for category, category_keywords in keywords.items():
                if any(kw in filename_lower for kw in category_keywords):
                    categories[category].append(str(filepath))
                    categorized = True
                    break
            
            if not categorized:
                uncategorized.append(str(filepath))
    
    return {
        'categories': {k: len(v) for k, v in categories.items()},
        'uncategorized': len(uncategorized),
        'total_categorized': sum(len(v) for v in categories.values())
    }

if __name__ == "__main__":
    print("=== ULTRA SCRIPT CONSOLIDATION ANALYSIS ===\n")
    
    # Run analysis
    analysis = analyze_scripts()
    categorization = categorize_scripts()
    
    # Print results
    print(f"TOTAL SCRIPTS FOUND: {analysis['total_scripts']}")
    print(f"\nBY TYPE:")
    for ext, count in analysis['by_type'].items():
        print(f"  {ext}: {count}")
    
    print(f"\nBY TOP DIRECTORY:")
    for dir_name, count in sorted(analysis['by_directory'].items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {dir_name}/: {count}")
    
    print(f"\nDUPLICATES:")
    print(f"  Duplicate groups: {analysis['duplicate_groups']}")
    print(f"  Total duplicate files: {analysis['duplicates']}")
    
    print(f"\nSMALL/STUB SCRIPTS: {analysis['small_scripts']}")
    
    print(f"\nCATEGORIZATION:")
    for category, count in categorization['categories'].items():
        print(f"  {category}: {count}")
    print(f"  uncategorized: {categorization['uncategorized']}")
    
    print(f"\nCONSOLIDATION POTENTIAL:")
    reduction_potential = analysis['duplicates'] + analysis['small_scripts']
    print(f"  Immediate reduction possible: {reduction_potential} scripts")
    print(f"  Target reduction: {int(analysis['total_scripts'] * 0.8)} scripts")
    print(f"  Final target: {int(analysis['total_scripts'] * 0.2)} scripts (~350)")
    
    # Save detailed report
    with open('/opt/sutazaiapp/script_consolidation_report.json', 'w') as f:
        json.dump({
            'analysis': analysis,
            'categorization': categorization
        }, f, indent=2)
    
    print("\nDetailed report saved to: /opt/sutazaiapp/script_consolidation_report.json")