#!/usr/bin/env python3
"""
Comprehensive script cleanup - remove fantasy elements and archive unused scripts
"""

import os
import re
import shutil
from pathlib import Path
from datetime import datetime
import subprocess

# Essential scripts to keep and clean
ESSENTIAL_SCRIPTS = [
    'live_logs.sh',
    'deploy_complete_system.sh',
    'system_test.sh',
    'health_check.sh',
    'verify_deployment.sh',
    'run_tests.sh',
    'cleanup_cache.sh',
    'setup.sh',
]

# Patterns indicating unused/fantasy scripts
UNUSED_PATTERNS = [
    r'brain',
    r'neural',
    r'agi',
    r'asi',
    r'consciousness',
    r'quantum',
    r'neuromorphic',
    r'divine',
    r'transcendent',
    r'omniscient',
    r'self_improvement',
    r'emergence',
    r'singularity',
]

# Fantasy terms to replace in scripts
SCRIPT_REPLACEMENTS = {
    r'\bAGI\b': 'automation',
    r'\bASI\b': 'advanced automation',
    r'agi[_-]?brain': 'task_coordinator',
    r'brain': 'coordinator',
    r'Brain': 'Coordinator',
    r'neural': 'processing',
    r'Neural': 'Processing',
    r'consciousness': 'system_state',
    r'Consciousness': 'System State',
    r'self[_-]?improvement': 'continuous_improvement',
    r'quantum': 'advanced',
    r'neuromorphic': 'optimized',
    r'backend-agi': 'backend',
    r'frontend-agi': 'frontend',
    r'sutazai-backend-agi': 'sutazai-backend',
    r'sutazai-frontend-agi': 'sutazai-frontend',
}

def is_essential(script_name):
    """Check if script is essential"""
    return script_name in ESSENTIAL_SCRIPTS

def has_fantasy_elements(filepath):
    """Check if file contains fantasy elements"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read().lower()
        
        for pattern in UNUSED_PATTERNS:
            if re.search(pattern, content):
                return True
        return False
    except:
        return False

def clean_script_content(content):
    """Remove fantasy elements from script content"""
    cleaned = content
    
    for pattern, replacement in SCRIPT_REPLACEMENTS.items():
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE if pattern.islower() else 0)
    
    return cleaned

def analyze_scripts(scripts_dir):
    """Analyze all scripts and categorize them"""
    essential = []
    fantasy = []
    utility = []
    
    for script in Path(scripts_dir).rglob('*'):
        if script.is_file() and not script.name.startswith('.'):
            rel_path = script.relative_to(scripts_dir)
            
            if is_essential(script.name):
                essential.append(rel_path)
            elif has_fantasy_elements(script):
                fantasy.append(rel_path)
            else:
                # Check if it's a utility script
                if any(keyword in str(rel_path).lower() for keyword in ['test', 'monitor', 'check', 'verify', 'validate']):
                    utility.append(rel_path)
                else:
                    fantasy.append(rel_path)  # Archive by default if not clearly useful
    
    return essential, utility, fantasy

def clean_essential_scripts(scripts_dir, essential_scripts):
    """Clean fantasy elements from essential scripts"""
    cleaned_count = 0
    
    for script_path in essential_scripts:
        full_path = Path(scripts_dir) / script_path
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            cleaned = clean_script_content(content)
            
            if cleaned != content:
                # Backup original
                backup_path = full_path.with_suffix(full_path.suffix + '.pre_cleanup')
                shutil.copy2(full_path, backup_path)
                
                # Write cleaned content
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned)
                
                cleaned_count += 1
                print(f"  ✅ Cleaned: {script_path}")
            else:
                print(f"  ✓ Already clean: {script_path}")
                
        except Exception as e:
            print(f"  ❌ Error cleaning {script_path}: {e}")
    
    return cleaned_count

def archive_scripts(scripts_dir, scripts_to_archive):
    """Archive unused scripts"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    archive_dir = Path(scripts_dir).parent / 'archive' / f'scripts_cleanup_{timestamp}'
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    archived_count = 0
    
    for script_path in scripts_to_archive:
        full_path = Path(scripts_dir) / script_path
        
        if full_path.exists():
            try:
                # Create subdirectory structure in archive
                archive_path = archive_dir / script_path
                archive_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Move file
                shutil.move(str(full_path), str(archive_path))
                archived_count += 1
                
            except Exception as e:
                print(f"  ❌ Error archiving {script_path}: {e}")
    
    return archived_count, archive_dir

def main():
    scripts_dir = Path('/opt/sutazaiapp/scripts')
    
    print("🧹 SutazAI Scripts Cleanup")
    print("=" * 60)
    
    # Step 1: Analyze scripts
    print("\n📊 Analyzing scripts...")
    essential, utility, fantasy = analyze_scripts(scripts_dir)
    
    print(f"\n📋 Script Analysis:")
    print(f"  Essential scripts: {len(essential)}")
    print(f"  Utility scripts: {len(utility)}")
    print(f"  Fantasy/unused scripts: {len(fantasy)}")
    
    # Step 2: Clean essential scripts
    print(f"\n🔧 Cleaning essential scripts...")
    cleaned_count = clean_essential_scripts(scripts_dir, essential)
    print(f"  Cleaned {cleaned_count} scripts")
    
    # Step 3: Clean utility scripts that we're keeping
    print(f"\n🔧 Cleaning utility scripts...")
    utility_cleaned = clean_essential_scripts(scripts_dir, utility)
    print(f"  Cleaned {utility_cleaned} utility scripts")
    
    # Step 4: Archive fantasy/unused scripts
    print(f"\n📦 Archiving {len(fantasy)} fantasy/unused scripts...")
    if fantasy:
        archived_count, archive_dir = archive_scripts(scripts_dir, fantasy)
        print(f"  ✅ Archived {archived_count} scripts to {archive_dir}")
    
    # Step 5: Clean up empty directories
    print("\n🗑️ Cleaning up empty directories...")
    empty_dirs = []
    for dirpath, dirnames, filenames in os.walk(scripts_dir, topdown=False):
        if not dirnames and not filenames and dirpath != str(scripts_dir):
            empty_dirs.append(dirpath)
            os.rmdir(dirpath)
    
    if empty_dirs:
        print(f"  ✅ Removed {len(empty_dirs)} empty directories")
    
    # Step 6: Update script permissions
    print("\n🔐 Updating script permissions...")
    for script in essential + utility:
        full_path = scripts_dir / script
        if full_path.exists() and full_path.suffix == '.sh':
            os.chmod(full_path, 0o755)
    
    # Summary
    print("\n📊 Cleanup Summary:")
    print(f"  Scripts kept: {len(essential) + len(utility)}")
    print(f"  Scripts cleaned: {cleaned_count + utility_cleaned}")
    print(f"  Scripts archived: {archived_count if fantasy else 0}")
    print(f"  Empty dirs removed: {len(empty_dirs)}")
    
    # List remaining scripts
    print("\n✅ Essential scripts preserved:")
    for script in sorted(essential):
        print(f"  - {script}")
    
    print("\n📁 Final script count:")
    remaining_count = len(list(scripts_dir.rglob('*')))
    print(f"  Total files remaining: {remaining_count}")

if __name__ == "__main__":
    main()