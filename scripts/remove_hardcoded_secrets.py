#!/usr/bin/env python3
"""
Purpose: Remove hardcoded default passwords from docker-compose files
Usage: python scripts/remove_hardcoded_secrets.py
Requirements: Python 3.8+
"""
import re
import os
from pathlib import Path


def remove_hardcoded_secrets(file_path):
    """Remove hardcoded secrets from a docker-compose file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Patterns to replace - remove the default values after :-
    patterns = [
        # Password patterns
        (r'\$\{POSTGRES_PASSWORD:-[^}]+\}', '${POSTGRES_PASSWORD}'),
        (r'\$\{REDIS_PASSWORD:-[^}]+\}', '${REDIS_PASSWORD}'),
        (r'\$\{NEO4J_PASSWORD:-[^}]+\}', '${NEO4J_PASSWORD}'),
        (r'\$\{DB_PASSWORD:-[^}]+\}', '${DB_PASSWORD}'),
        (r'\$\{GRAFANA_ADMIN_PASSWORD:-[^}]+\}', '${GRAFANA_ADMIN_PASSWORD}'),
        
        # Secret key patterns
        (r'\$\{SECRET_KEY:-[^}]+\}', '${SECRET_KEY}'),
        (r'\$\{JWT_SECRET:-[^}]+\}', '${JWT_SECRET}'),
        
        # Keep defaults for non-sensitive values
        # (r'\$\{POSTGRES_USER:-sutazai\}', '${POSTGRES_USER:-sutazai}'),
        # (r'\$\{POSTGRES_DB:-sutazai\}', '${POSTGRES_DB:-sutazai}'),
    ]
    
    changes_made = False
    for pattern, replacement in patterns:
        new_content = re.sub(pattern, replacement, content)
        if new_content != content:
            changes_made = True
            content = new_content
    
    if changes_made:
        # Backup original file
        backup_path = f"{file_path}.bak"
        with open(backup_path, 'w') as f:
            f.write(original_content)
        
        # Write updated content
        with open(file_path, 'w') as f:
            f.write(content)
        
        print(f"✅ Updated {file_path}")
        print(f"   Backup saved to {backup_path}")
        return True
    else:
        print(f"ℹ️  No changes needed for {file_path}")
        return False


def main():
    """Main function to process all docker-compose files."""
    print("🔒 Removing hardcoded secrets from docker-compose files...")
    print("=" * 60)
    
    # Find all docker-compose files
    docker_compose_files = []
    
    # Main docker-compose files
    for pattern in ['docker-compose*.yml', 'docker-compose*.yaml']:
        docker_compose_files.extend(Path('/opt/sutazaiapp').glob(pattern))
    
    # Also check in subdirectories
    for subdir in ['config/docker', 'deployment', 'workflows/deployments']:
        subdir_path = Path(f'/opt/sutazaiapp/{subdir}')
        if subdir_path.exists():
            docker_compose_files.extend(subdir_path.rglob('docker-compose*.yml'))
            docker_compose_files.extend(subdir_path.rglob('docker-compose*.yaml'))
    
    # Remove duplicates and sort
    docker_compose_files = sorted(set(docker_compose_files))
    
    print(f"Found {len(docker_compose_files)} docker-compose files to process\n")
    
    updated_count = 0
    for file_path in docker_compose_files:
        # Skip archive and backup files
        if 'archive' in str(file_path) or '.bak' in str(file_path):
            print(f"⏭️  Skipping archived file: {file_path}")
            continue
            
        if remove_hardcoded_secrets(file_path):
            updated_count += 1
    
    print("\n" + "=" * 60)
    print(f"✅ Updated {updated_count} files")
    print("\n⚠️  IMPORTANT: Before deploying:")
    print("1. Copy .env.example to .env")
    print("2. Update all passwords in .env with secure values")
    print("3. Never commit .env to version control")
    print("4. Add .env to .gitignore if not already there")


if __name__ == "__main__":
    main()