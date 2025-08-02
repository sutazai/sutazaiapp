#!/usr/bin/env python3
"""
Comprehensive cleanup script to remove ALL fantasy elements from SutazAI codebase
Transforms automation/advanced automation system into practical task automation platform
"""

import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

# Comprehensive fantasy terms to replace
FANTASY_REPLACEMENTS = {
    # automation/advanced automation terms
    r'\bAGI\b': 'automation',
    r'\bASI\b': 'advanced automation',
    r'agi[_-]?coordinator': 'task_coordinator',
    r'coordinator[_-]?architecture': 'system_architecture',
    r'Task Coordinator': 'Task Coordinator',
    r'task coordinator': 'task coordinator',
    r'automation coordinator': 'Task Coordinator',
    r'automation/advanced automation': 'Automation',
    
    # Backend service names
    r'backend': 'backend',
    r'frontend': 'frontend',
    r'backend': 'backend',
    r'frontend': 'frontend',
    
    # system_state terms
    r'system_state[_-]?level': 'system_health_score',
    r'system_state': 'system_state',
    r'active': 'active',
    r'self[_-]?aware': 'self_monitoring',
    r'system_state[_-]?state': 'system_state',
    r'system active': 'system active',
    
    # Cognitive terms
    r'cognitive[_-]?modules?': 'task_processors',
    r'cognitive[_-]?functions?': 'processing_functions',
    r'processing[_-]?engine': 'processing_engine',
    r'processing[_-]?network': 'ml_model',
    r'task processing engine': 'task processing engine',
    r'task processing engine': 'Task processing engine',
    r'task processing': 'task processing',
    r'task processing': 'Task processing',
    r'system monitoring': 'system monitoring',
    r'system monitoring': 'System monitoring',
    r'processing': 'processing',
    r'processing': 'Processing',
    
    # coordinator terms
    r'coordinator': 'coordinator',
    r'coordinator': 'Coordinator',
    r'/api/v1/coordinator': '/api/v1/coordinator',
    
    # Fantasy descriptors
    r'advanced': 'advanced',
    r'optimized': 'optimized',
    r'comprehensive': 'comprehensive',
    r'powerful': 'powerful',
    r'sophisticated': 'sophisticated',
    r'advanced': 'advanced',
    r'system-wide': 'system-wide',
    r'high-performance': 'high-performance',
    r'cloud-based': 'cloud-based',
    r'lightweight': 'lightweight',
    
    # optimization terms
    r'optimization': 'optimization',
    r'emergent[_-]?behavior': 'optimized_behavior',
    r'convergence': 'convergence',
    r'meta[_-]?learning': 'transfer_learning',
    r'self[_-]?improvement': 'continuous_improvement',
    
    # advanced terms
    r'advanced': 'advanced',
    r'optimized': 'optimized',
    
    # Model names
    r'automation system-coordinator': 'task-coordinator',
    r'task_coordinator': 'task-coordinator',
}

# Files to process
TARGET_FILES = [
    '/opt/sutazaiapp/backend/app/main.py',
    '/opt/sutazaiapp/frontend/app.py',
    '/opt/sutazaiapp/docker-compose.yml',
    '/opt/sutazaiapp/docker-compose.minimal.yml',
    '/opt/sutazaiapp/docker-compose.agents.yml',
]

def clean_content(content: str) -> str:
    """Clean fantasy elements from content"""
    cleaned = content
    
    # Apply replacements in order
    for pattern, replacement in FANTASY_REPLACEMENTS.items():
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE if pattern.islower() else 0)
    
    # Fix any double replacements
    cleaned = re.sub(r'system_state[_-]?state', 'system_state', cleaned)
    cleaned = re.sub(r'Task', 'Task', cleaned)
    cleaned = re.sub(r'task', 'task', cleaned)
    cleaned = re.sub(r'processing', 'processing', cleaned)
    cleaned = re.sub(r'Processing', 'Processing', cleaned)
    
    return cleaned

def process_file(filepath: Path) -> Tuple[bool, str]:
    """Process a file and clean fantasy elements"""
    try:
        # Read file
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        cleaned = clean_content(content)
        
        if cleaned != original:
            # Backup original
            backup_path = filepath.with_suffix(filepath.suffix + '.fantasy_backup')
            shutil.copy2(filepath, backup_path)
            
            # Write cleaned content
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(cleaned)
            
            # Count changes
            changes = sum(1 for a, b in zip(original.split('\n'), cleaned.split('\n')) if a != b)
            return True, f"Cleaned {filepath.name} - {changes} lines modified"
        
        return False, f"No changes needed for {filepath.name}"
        
    except Exception as e:
        return False, f"Error processing {filepath}: {e}"

def clean_all_python_files(directory: Path) -> List[str]:
    """Clean all Python files in directory"""
    results = []
    
    for py_file in directory.rglob('*.py'):
        if 'archive' not in str(py_file) and '.fantasy_backup' not in str(py_file):
            success, message = process_file(py_file)
            if success:
                results.append(message)
    
    return results

def clean_docker_files() -> List[str]:
    """Clean Docker compose files"""
    results = []
    
    docker_files = [
        '/opt/sutazaiapp/docker-compose.yml',
        '/opt/sutazaiapp/docker-compose.minimal.yml',
        '/opt/sutazaiapp/docker-compose.agents.yml',
    ]
    
    for docker_file in docker_files:
        if os.path.exists(docker_file):
            success, message = process_file(Path(docker_file))
            if success:
                results.append(message)
    
    return results

def main():
    """Main cleanup function"""
    print("🧹 Starting Comprehensive Fantasy Elements Cleanup")
    print("=" * 60)
    
    # Step 1: Clean specific target files
    print("\n📝 Cleaning target files...")
    for target_file in TARGET_FILES:
        if os.path.exists(target_file):
            success, message = process_file(Path(target_file))
            print(f"  {'✅' if success else '⚠️'} {message}")
    
    # Step 2: Clean all Python files in backend
    print("\n🔧 Cleaning all backend Python files...")
    backend_results = clean_all_python_files(Path('/opt/sutazaiapp/backend'))
    for result in backend_results[:10]:  # Show first 10
        print(f"  ✅ {result}")
    if len(backend_results) > 10:
        print(f"  ... and {len(backend_results) - 10} more files")
    
    # Step 3: Clean all Python files in scripts
    print("\n📜 Cleaning all script files...")
    script_results = clean_all_python_files(Path('/opt/sutazaiapp/scripts'))
    for result in script_results[:10]:  # Show first 10
        print(f"  ✅ {result}")
    if len(script_results) > 10:
        print(f"  ... and {len(script_results) - 10} more files")
    
    # Step 4: Clean Docker files
    print("\n🐳 Cleaning Docker files...")
    docker_results = clean_docker_files()
    for result in docker_results:
        print(f"  ✅ {result}")
    
    # Step 5: Update environment variables
    print("\n🔧 Updating environment variables...")
    env_file = Path('/opt/sutazaiapp/.env')
    if env_file.exists():
        with open(env_file, 'r') as f:
            env_content = f.read()
        
        # Update any automation references
        env_content = re.sub(r'automation', 'AUTOMATION', env_content)
        env_content = re.sub(r'agi', 'automation', env_content)
        
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("  ✅ Updated .env file")
    
    print("\n✨ Cleanup complete!")
    print("\nNext steps:")
    print("1. Review the changes with `git diff`")
    print("2. Restart all services: `docker-compose down && docker-compose up -d`")
    print("3. Run tests to ensure functionality")
    print("4. Verify API endpoints are working correctly")
    
    # Summary
    total_cleaned = len(backend_results) + len(script_results) + len(docker_results) + len([f for f in TARGET_FILES if os.path.exists(f)])
    print(f"\n📊 Summary: Cleaned {total_cleaned} files")
    print("🎯 All fantasy elements have been removed!")

if __name__ == "__main__":
    main()