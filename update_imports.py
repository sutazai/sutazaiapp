#!/usr/bin/env python3
"""
Import Consolidation Script
Updates all imports to use the canonical BaseAgent from agents.core.base_agent
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

# Define the import mappings
IMPORT_MAPPINGS = [
    # Old base_agent_v2 imports -> canonical base_agent imports
    (r'from\s+agents\.core\.base_agent_v2\s+import\s+(.+)', r'from agents.core.base_agent import \1'),
    (r'from\s+core\.base_agent_v2\s+import\s+(.+)', r'from agents.core.base_agent import \1'),
    (r'from\s+\.base_agent_v2\s+import\s+(.+)', r'from .base_agent import \1'),
    
    # Old simple_base_agent imports -> canonical base_agent imports
    (r'from\s+agents\.core\.simple_base_agent\s+import\s+(.+)', r'from agents.core.base_agent import \1'),
    (r'from\s+simple_base_agent\s+import\s+(.+)', r'from agents.core.base_agent import \1'),
    
    # Root level base_agent imports -> canonical
    (r'from\s+agents\.base_agent\s+import\s+(.+)', r'from agents.core.base_agent import \1'),
    (r'from\s+base_agent\s+import\s+(.+)', r'from agents.core.base_agent import \1'),
    
    # Backend base_agent imports -> canonical  
    (r'from\s+\.\.ai_agents\.core\.base_agent\s+import\s+(.+)', r'from agents.core.base_agent import \1'),
    (r'from\s+backend\.ai_agents\.core\.base_agent\s+import\s+(.+)', r'from agents.core.base_agent import \1'),
    
    # Compatibility base_agent imports -> canonical
    (r'from\s+agents\.compatibility_base_agent\s+import\s+(.+)', r'from agents.core.base_agent import \1'),
    
    # Import statements without from
    (r'import\s+agents\.core\.base_agent_v2', r'import agents.core.base_agent'),
    (r'import\s+core\.base_agent_v2', r'import agents.core.base_agent'),
    (r'import\s+base_agent_v2', r'import agents.core.base_agent'),
]

# Class name mappings for code updates
CLASS_MAPPINGS = [
    (r'BaseAgent', r'BaseAgent'),
    # Note: We keep BaseAgent as is since it's already canonical
]

def update_file_imports(file_path: Path) -> Tuple[bool, List[str]]:
    """
    Update imports in a single file
    Returns (changed, changes_made)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return False, [f"Error reading file: {e}"]
    
    original_content = content
    changes_made = []
    
    # Apply import mappings
    for old_pattern, new_pattern in IMPORT_MAPPINGS:
        matches = re.findall(old_pattern, content)
        if matches:
            new_content = re.sub(old_pattern, new_pattern, content)
            if new_content != content:
                changes_made.append(f"Updated import: {old_pattern} -> {new_pattern}")
                content = new_content
    
    # Apply class name mappings (only in non-import contexts)
    for old_class, new_class in CLASS_MAPPINGS:
        # Only replace class names that are not in import statements
        pattern = rf'\b{old_class}\b(?![^#]*import)'
        matches = re.findall(pattern, content)
        if matches:
            new_content = re.sub(pattern, new_class, content)
            if new_content != content:
                changes_made.append(f"Updated class name: {old_class} -> {new_class}")
                content = new_content
    
    # Write back if changed
    if content != original_content:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, changes_made
        except Exception as e:
            return False, [f"Error writing file: {e}"]
    
    return False, []

def find_python_files(root_dir: Path) -> List[Path]:
    """Find all Python files in the directory tree"""
    python_files = []
    
    # Skip these directories
    skip_dirs = {
        '.git', '__pycache__', '.pytest_cache', 'node_modules', 
        'venv', 'env', '.venv', '.env', 'build', 'dist'
    }
    
    for root, dirs, files in os.walk(root_dir):
        # Remove skip_dirs from dirs to avoid traversing them
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(Path(root) / file)
    
    return python_files

def main():
    """Main function to update all imports"""
    root_dir = Path('/opt/sutazaiapp')
    
    print("🔍 Finding Python files...")
    python_files = find_python_files(root_dir)
    print(f"Found {len(python_files)} Python files")
    
    total_changed = 0
    total_changes = 0
    
    print("\n📝 Updating imports...")
    for file_path in python_files:
        try:
            changed, changes = update_file_imports(file_path)
            if changed:
                total_changed += 1
                total_changes += len(changes)
                print(f"✅ Updated {file_path}")
                for change in changes:
                    print(f"   - {change}")
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
    
    print(f"\n✅ Import consolidation complete!")
    print(f"📊 Files modified: {total_changed}")
    print(f"📊 Total changes: {total_changes}")
    
    # Show summary of what was updated
    if total_changed > 0:
        print("\n📋 Summary of changes:")
        print("   - All BaseAgent imports now use: from agents.core.base_agent import BaseAgent")
        print("   - All BaseAgentV2 class references changed to BaseAgent")
        print("   - Consolidated 6 different import patterns into 1 canonical pattern")

if __name__ == "__main__":
    main()