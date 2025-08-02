#!/usr/bin/env python3
"""
Final cleanup to remove all AGI/ASI/LocalAGI/BigAGI references from Docker configurations
"""

import re
from pathlib import Path

def clean_docker_compose(filepath: Path):
    """Remove AGI services and references from docker-compose files"""
    print(f"🔧 Cleaning {filepath}")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    original = content
    
    # Remove entire LocalAGI service blocks
    content = re.sub(
        r'\s*localagi.*?(?=^\s*\w+:|\Z)',
        '\n  # LocalAGI - Service removed\n',
        content,
        flags=re.MULTILINE | re.DOTALL
    )
    
    # Remove entire BigAGI service blocks
    content = re.sub(
        r'\s*bigagi.*?(?=^\s*\w+:|\Z)',
        '\n  # BigAGI - Service removed\n',
        content,
        flags=re.MULTILINE | re.DOTALL
    )
    
    # Fix backend-agi references
    content = re.sub(r'Dockerfile\.agi', 'Dockerfile', content)
    content = re.sub(r'backend-agi', 'backend', content)
    
    # Remove LocalAGI/BigAGI from depends_on
    content = re.sub(r'\s*- localagi\n', '', content)
    content = re.sub(r'\s*- bigagi\n', '', content)
    
    # Remove LocalAGI/BigAGI from health check services
    content = re.sub(r',sutazai-localagi', '', content)
    content = re.sub(r',sutazai-bigagi', '', content)
    content = re.sub(r'sutazai-localagi,', '', content)
    content = re.sub(r'sutazai-bigagi,', '', content)
    
    if content != original:
        # Backup
        backup_path = filepath.with_suffix(filepath.suffix + '.final_backup')
        with open(backup_path, 'w') as f:
            f.write(original)
        
        # Write cleaned content
        with open(filepath, 'w') as f:
            f.write(content)
        
        print(f"  ✅ Cleaned {filepath}")
        return True
    
    print(f"  ℹ️  No changes needed")
    return False

def main():
    """Clean all main docker-compose files"""
    files_to_clean = [
        '/opt/sutazaiapp/docker-compose.yml',
        '/opt/sutazaiapp/config/docker/docker-compose.yml',
        '/opt/sutazaiapp/docker-compose-agents-complete.yml'
    ]
    
    print("🚀 Final Docker Cleanup")
    print("=" * 60)
    
    for filepath in files_to_clean:
        if Path(filepath).exists():
            clean_docker_compose(Path(filepath))
    
    # Also clean the agents Dockerfiles
    agent_dockerfiles = Path('/opt/sutazaiapp/docker/agents').glob('Dockerfile.*')
    for dockerfile in agent_dockerfiles:
        if 'localagi' in str(dockerfile) or 'bigagi' in str(dockerfile):
            print(f"  🗑️  Removing {dockerfile}")
            dockerfile.unlink()
    
    print("\n✅ Final cleanup complete!")

if __name__ == "__main__":
    main()