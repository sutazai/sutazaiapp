#!/usr/bin/env python3
"""
Manual cleanup of docker-compose files to remove AGI/ASI references
"""

import os
import re
from pathlib import Path
import shutil

def fix_docker_compose_manual(filepath: Path):
    """Manually fix docker-compose files with text replacement"""
    print(f"\n🔧 Processing: {filepath}")
    
    try:
        # Read file content
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Store original for comparison
        original = content
        
        # Backend-agi replacements
        content = re.sub(r'\bbackend-agi\b', 'backend', content)
        content = re.sub(r'\bsutazai-backend-agi\b', 'sutazai-backend', content)
        content = re.sub(r'backend_agi', 'backend', content)
        content = re.sub(r'BACKEND_AGI', 'BACKEND', content)
        
        # LocalAGI/BigAGI removal (comment out or remove)
        content = re.sub(r'(\s*)localagi:.*?(?=\n\s*\w+:|$)', r'\1# LocalAGI service removed', content, flags=re.DOTALL)
        content = re.sub(r'(\s*)bigagi:.*?(?=\n\s*\w+:|$)', r'\1# BigAGI service removed', content, flags=re.DOTALL)
        
        # Jarvis-agi replacements
        content = re.sub(r'\bjarvis-agi\b', 'jarvis', content)
        content = re.sub(r'\bsutazai-jarvis-agi\b', 'sutazai-jarvis', content)
        
        # AGI/ASI terms in comments and descriptions
        content = re.sub(r'AGI system', 'automation system', content, flags=re.IGNORECASE)
        content = re.sub(r'ASI capabilities', 'advanced automation', content, flags=re.IGNORECASE)
        content = re.sub(r'AGI integration', 'automation integration', content, flags=re.IGNORECASE)
        content = re.sub(r'neural network', 'processing network', content, flags=re.IGNORECASE)
        content = re.sub(r'brain service', 'coordinator service', content, flags=re.IGNORECASE)
        content = re.sub(r'consciousness', 'system state', content, flags=re.IGNORECASE)
        content = re.sub(r'quantum', 'optimization', content, flags=re.IGNORECASE)
        
        # Environment variable replacements
        content = re.sub(r'AGI_MODE', 'AUTOMATION_MODE', content)
        content = re.sub(r'ASI_ENABLED', 'ADVANCED_AUTOMATION_ENABLED', content)
        content = re.sub(r'NEURAL_', 'PROCESSING_', content)
        content = re.sub(r'BRAIN_', 'COORDINATOR_', content)
        content = re.sub(r'CONSCIOUSNESS_', 'SYSTEM_STATE_', content)
        content = re.sub(r'QUANTUM_', 'OPTIMIZATION_', content)
        
        # Fix depends_on references
        content = re.sub(r'- backend-agi\b', '- backend', content)
        content = re.sub(r'- localagi\b', '# - localagi (removed)', content)
        content = re.sub(r'- bigagi\b', '# - bigagi (removed)', content)
        content = re.sub(r'- jarvis-agi\b', '- jarvis', content)
        
        # Fix volume references
        content = re.sub(r'backend-agi_data', 'backend_data', content)
        content = re.sub(r'localagi_data', '# localagi_data (removed)', content)
        content = re.sub(r'bigagi_data', '# bigagi_data (removed)', content)
        content = re.sub(r'jarvis-agi_data', 'jarvis_data', content)
        
        # Fix network references
        content = re.sub(r'agi-network', 'automation-network', content)
        content = re.sub(r'neural-network', 'processing-network', content)
        
        # Fix build contexts
        content = re.sub(r'./backend-agi', './backend', content)
        content = re.sub(r'./docker/backend-agi', './docker/backend', content)
        content = re.sub(r'./services/backend-agi', './services/backend', content)
        
        # Fix image names
        content = re.sub(r'sutazai/backend-agi', 'sutazai/backend', content)
        content = re.sub(r'sutazai/localagi', '# sutazai/localagi (removed)', content)
        content = re.sub(r'sutazai/bigagi', '# sutazai/bigagi (removed)', content)
        content = re.sub(r'sutazai/jarvis-agi', 'sutazai/jarvis', content)
        
        if content != original:
            # Backup original
            backup_path = filepath.with_suffix(filepath.suffix + '.agi_backup')
            if not backup_path.exists():
                shutil.copy2(filepath, backup_path)
                print(f"  📋 Created backup: {backup_path}")
            
            # Write cleaned content
            with open(filepath, 'w') as f:
                f.write(content)
            
            # Count changes
            changes = len([m for m in re.finditer(r'agi|ASI|neural|brain|consciousness|quantum', original, re.IGNORECASE)])
            print(f"  ✅ Fixed {filepath} ({changes} potential AGI/fantasy references cleaned)")
            return True
        else:
            print(f"  ℹ️  No changes needed for {filepath}")
            return False
            
    except Exception as e:
        print(f"  ❌ Error processing {filepath}: {e}")
        return False

def main():
    """Main cleanup function"""
    print("🚀 Docker Compose AGI/ASI Manual Cleanup")
    print("=" * 60)
    
    root_path = Path('/opt/sutazaiapp')
    
    # Priority files to clean
    priority_files = [
        '/opt/sutazaiapp/docker-compose.yml',
        '/opt/sutazaiapp/docker-compose.minimal.yml',
        '/opt/sutazaiapp/docker-compose.agents.yml',
        '/opt/sutazaiapp/docker-compose.tinyllama.yml',
        '/opt/sutazaiapp/config/docker/docker-compose.yml',
        '/opt/sutazaiapp/config/docker/docker-compose.tinyllama.yml',
    ]
    
    # Clean priority files first
    print("\n📋 Cleaning main docker-compose files...")
    cleaned_count = 0
    
    for filepath in priority_files:
        if Path(filepath).exists():
            if fix_docker_compose_manual(Path(filepath)):
                cleaned_count += 1
    
    # Find all other docker-compose files
    all_compose_files = list(root_path.glob('**/docker-compose*.yml'))
    all_compose_files.extend(root_path.glob('**/docker-compose*.yaml'))
    
    print(f"\n📋 Found {len(all_compose_files)} total docker-compose files")
    
    # Clean remaining files
    for compose_file in all_compose_files:
        if str(compose_file) not in priority_files and 'backup' not in str(compose_file):
            if fix_docker_compose_manual(compose_file):
                cleaned_count += 1
    
    # Also clean the jarvis-agi script
    jarvis_file = root_path / 'scripts/docker/services/docker_jarvis-agi_jarvis_super_system.py'
    if jarvis_file.exists():
        print(f"\n🤖 Cleaning JARVIS AGI script...")
        try:
            with open(jarvis_file, 'r') as f:
                content = f.read()
            
            # Replace fantasy elements
            original = content
            content = re.sub(r'JARVIS Super Intelligence', 'JARVIS Automation System', content)
            content = re.sub(r'🧠', '🤖', content)
            content = re.sub(r'intelligence', 'automation', content, flags=re.IGNORECASE)
            content = re.sub(r'neural', 'processing', content, flags=re.IGNORECASE)
            content = re.sub(r'brain', 'coordinator', content, flags=re.IGNORECASE)
            content = re.sub(r'consciousness', 'system state', content, flags=re.IGNORECASE)
            content = re.sub(r'quantum', 'optimization', content, flags=re.IGNORECASE)
            content = re.sub(r'AGI', 'automation', content)
            content = re.sub(r'ASI', 'advanced automation', content)
            
            if content != original:
                # Backup and write
                backup_path = jarvis_file.with_suffix('.py.agi_backup')
                shutil.copy2(jarvis_file, backup_path)
                
                with open(jarvis_file, 'w') as f:
                    f.write(content)
                
                print(f"  ✅ Cleaned JARVIS script")
                cleaned_count += 1
                
        except Exception as e:
            print(f"  ❌ Error cleaning JARVIS script: {e}")
    
    # Clean Dockerfiles
    print("\n🐳 Cleaning Dockerfiles...")
    dockerfiles = list(root_path.glob('**/Dockerfile*'))
    
    for dockerfile in dockerfiles:
        if 'backup' not in str(dockerfile):
            try:
                with open(dockerfile, 'r') as f:
                    content = f.read()
                
                original = content
                
                # Replace backend-agi references
                content = re.sub(r'backend-agi', 'backend', content)
                content = re.sub(r'BACKEND_AGI', 'BACKEND', content)
                content = re.sub(r'backend_agi', 'backend', content)
                content = re.sub(r'localagi', '# localagi removed', content)
                content = re.sub(r'bigagi', '# bigagi removed', content)
                content = re.sub(r'jarvis-agi', 'jarvis', content)
                
                # Replace AGI/ASI references
                content = re.sub(r'AGI system', 'automation system', content)
                content = re.sub(r'ASI capabilities', 'advanced automation', content)
                
                if content != original:
                    # Backup
                    backup_path = dockerfile.with_suffix(dockerfile.suffix + '.agi_backup')
                    if not backup_path.exists():
                        shutil.copy2(dockerfile, backup_path)
                    
                    # Write cleaned content
                    with open(dockerfile, 'w') as f:
                        f.write(content)
                    
                    print(f"  ✅ Cleaned: {dockerfile}")
                    cleaned_count += 1
                    
            except Exception as e:
                print(f"  ❌ Error cleaning {dockerfile}: {e}")
    
    print("\n" + "=" * 60)
    print(f"✅ Cleanup Complete!")
    print(f"📊 Cleaned {cleaned_count} files total")
    
    # Final verification
    print("\n🔍 Running final verification...")
    remaining = []
    
    for compose_file in root_path.glob('**/docker-compose*.yml'):
        if 'backup' not in str(compose_file):
            try:
                with open(compose_file, 'r') as f:
                    content = f.read().lower()
                
                if any(term in content for term in ['backend-agi', 'localagi', 'bigagi', 'jarvis-agi', ' agi ', ' asi ']):
                    remaining.append(compose_file)
            except:
                pass
    
    if remaining:
        print(f"⚠️  Found {len(remaining)} files with possible remaining AGI/ASI references:")
        for f in remaining[:10]:
            print(f"  - {f}")
    else:
        print("✅ All files are clean!")
    
    print("\n🔍 Next steps:")
    print("1. Review changes: git diff")
    print("2. Test configuration: docker-compose config")
    print("3. Restart services: docker-compose down && docker-compose up -d")

if __name__ == "__main__":
    main()