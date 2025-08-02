#!/usr/bin/env python3
"""
Cleanup script to remove all fantasy elements from SutazAI codebase
Transforms automation/advanced automation system into practical task automation platform
"""

import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

# Fantasy terms to replace
FANTASY_REPLACEMENTS = {
    # automation/advanced automation terms
    r'\bAGI\b': 'automation system',
    r'\bASI\b': 'advanced automation',
    r'agi[_-]?coordinator': 'task_coordinator',
    r'coordinator[_-]?architecture': 'system_architecture',
    
    # System State terms
    r'system_state[_-]?level': 'system_health_score',
    r'system_state': 'system_state',
    r'active': 'active',
    r'self[_-]?aware': 'self_monitoring',
    
    # Cognitive terms
    r'cognitive[_-]?modules?': 'task_processors',
    r'cognitive[_-]?functions?': 'processing_functions',
    r'processing[_-]?engine': 'processing_engine',
    r'processing[_-]?network': 'ml_model',
    
    # Fantasy descriptors
    r'optimal': 'advanced',
    r'enhanced': 'optimized',
    r'comprehensive': 'comprehensive',
    r'powerful': 'powerful',
    r'sophisticated': 'sophisticated',
    r'advanced': 'advanced',
    r'system-wide': 'system-wide',
    r'high-performance': 'high-performance',
    r'cloud-based': 'cloud-based',
    r'lightweight': 'lightweight',
    
    # Emergence terms
    r'emergence': 'optimization',
    r'emergent[_-]?behavior': 'optimized_behavior',
    r'singularity': 'convergence',
    r'meta[_-]?learning': 'transfer_learning',
    r'self[_-]?improvement': 'continuous_improvement',
}

# Files to rename
FILE_RENAMES = {
    'task_coordinator.py': 'task_coordinator.py',
    'advanced_coordinator_architecture.py': 'system_architecture.py',
    'coordinator.py': 'system_api.py',
    'self_improvement.py': 'continuous_improvement.py',
    'autonomous_system_controller.py': 'system_controller.py',
    'processing_engine': 'processing_engine',
}

def clean_file_content(content: str) -> str:
    """Clean fantasy elements from file content"""
    cleaned = content
    
    for pattern, replacement in FANTASY_REPLACEMENTS.items():
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    
    return cleaned

def process_python_file(filepath: Path) -> Tuple[bool, str]:
    """Process a Python file and clean fantasy elements"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        cleaned = clean_file_content(content)
        
        if cleaned != original:
            # Backup original
            backup_path = filepath.with_suffix('.bak')
            shutil.copy2(filepath, backup_path)
            
            # Write cleaned content
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(cleaned)
            
            return True, f"Cleaned {filepath}"
        
        return False, f"No changes needed for {filepath}"
        
    except Exception as e:
        return False, f"Error processing {filepath}: {e}"

def rename_files(directory: Path) -> List[str]:
    """Rename files containing fantasy terms"""
    results = []
    
    for old_name, new_name in FILE_RENAMES.items():
        old_path = directory / old_name
        if old_path.exists():
            new_path = directory / new_name
            shutil.move(old_path, new_path)
            results.append(f"Renamed {old_name} to {new_name}")
    
    return results

def update_imports(directory: Path) -> List[str]:
    """Update import statements after file renames"""
    results = []
    
    for py_file in directory.rglob('*.py'):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            modified = False
            for old_name, new_name in FILE_RENAMES.items():
                old_import = old_name.replace('.py', '')
                new_import = new_name.replace('.py', '')
                
                # Update various import patterns
                patterns = [
                    (f'from .{old_import}', f'from .{new_import}'),
                    (f'import {old_import}', f'import {new_import}'),
                    (f'from {old_import}', f'from {new_import}'),
                ]
                
                for old_pattern, new_pattern in patterns:
                    if old_pattern in content:
                        content = content.replace(old_pattern, new_pattern)
                        modified = True
            
            if modified:
                with open(py_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                results.append(f"Updated imports in {py_file}")
                
        except Exception as e:
            results.append(f"Error updating {py_file}: {e}")
    
    return results

def archive_agi_docs(base_path: Path) -> List[str]:
    """Archive automation-related documentation"""
    results = []
    archive_dir = base_path / 'archive' / 'historical' / 'agi-concepts'
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    # Patterns for automation docs
    agi_patterns = ['*automation*.md', '*advanced automation*.md', '*system_state*.md', '*coordinator*.md']
    
    for pattern in agi_patterns:
        for doc in base_path.rglob(pattern):
            if 'archive' not in str(doc):
                dest = archive_dir / doc.name
                shutil.move(str(doc), str(dest))
                results.append(f"Archived {doc} to {dest}")
    
    return results

def main():
    """Main cleanup function"""
    base_path = Path('/opt/sutazaiapp')
    
    print("🧹 Starting SutazAI Fantasy Elements Cleanup")
    print("=" * 50)
    
    # Step 1: Clean Python files
    print("\n📝 Cleaning Python files...")
    backend_path = base_path / 'backend'
    python_files = list(backend_path.rglob('*.py'))
    
    for py_file in python_files:
        if 'archive' not in str(py_file):
            success, message = process_python_file(py_file)
            if success:
                print(f"  ✅ {message}")
    
    # Step 2: Rename files
    print("\n📁 Renaming files...")
    rename_results = rename_files(backend_path)
    for result in rename_results:
        print(f"  ✅ {result}")
    
    # Step 3: Update imports
    print("\n🔧 Updating imports...")
    import_results = update_imports(backend_path)
    for result in import_results[:10]:  # Show first 10
        print(f"  ✅ {result}")
    if len(import_results) > 10:
        print(f"  ... and {len(import_results) - 10} more files")
    
    # Step 4: Archive automation docs
    print("\n📦 Archiving automation documentation...")
    archive_results = archive_agi_docs(base_path)
    for result in archive_results:
        print(f"  ✅ {result}")
    
    print("\n✨ Cleanup complete!")
    print("\nNext steps:")
    print("1. Review the changes with `git diff`")
    print("2. Run tests to ensure functionality")
    print("3. Update any remaining configuration files")
    print("4. Commit the changes")

if __name__ == "__main__":
    main()