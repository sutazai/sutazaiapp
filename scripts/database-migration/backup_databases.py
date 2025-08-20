#!/usr/bin/env python3
"""
Backup all SQLite databases before migration
"""

import os
import shutil
import json
from datetime import datetime
from pathlib import Path
import hashlib

def find_databases(base_path="/opt/sutazaiapp"):
    """Find all SQLite databases"""
    databases = {
        'memory_dbs': [],
        'extended_memory': None,
        'n8n': None,
        'flowise': None
    }
    
    # Find memory.db files
    for root, dirs, files in os.walk(base_path):
        # Skip backup and migration directories
        if 'backups' in root or 'scripts/database-migration' in root:
            continue
        
        if 'memory.db' in files:
            db_path = os.path.join(root, 'memory.db')
            if os.path.exists(db_path):
                databases['memory_dbs'].append(db_path)
    
    # Check for other databases
    extended = "/opt/sutazaiapp/data/mcp/extended-memory/extended_memory.db"
    if os.path.exists(extended):
        databases['extended_memory'] = extended
    
    n8n = "/opt/sutazaiapp/data/n8n/database.sqlite"
    if os.path.exists(n8n):
        databases['n8n'] = n8n
        
    flowise = "/opt/sutazaiapp/data/flowise/database.sqlite"
    if os.path.exists(flowise):
        databases['flowise'] = flowise
    
    return databases

def backup_databases():
    """Backup all databases"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"/opt/sutazaiapp/backups/databases/{timestamp}"
    
    print(f"=== Database Backup ===")
    print(f"Backup directory: {backup_dir}")
    print()
    
    # Create backup directory
    os.makedirs(backup_dir, exist_ok=True)
    
    # Find all databases
    databases = find_databases()
    
    # Backup memory databases
    backed_up = 0
    print(f"Found {len(databases['memory_dbs'])} memory.db files")
    
    for db_path in databases['memory_dbs']:
        relative_path = db_path.replace("/opt/sutazaiapp/", "")
        backup_path = os.path.join(backup_dir, "memory_databases", relative_path)
        
        # Create directory structure
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        
        # Copy file
        print(f"Backing up: {db_path}")
        shutil.copy2(db_path, backup_path)
        
        # Verify
        if os.path.exists(backup_path):
            size = os.path.getsize(backup_path)
            print(f"  ✓ Backed up ({size} bytes)")
            backed_up += 1
    
    # Backup extended memory
    if databases['extended_memory']:
        backup_path = os.path.join(backup_dir, "extended_memory", "extended_memory.db")
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        print(f"Backing up extended memory database...")
        shutil.copy2(databases['extended_memory'], backup_path)
        print(f"  ✓ Extended memory backed up")
        backed_up += 1
    
    # Backup N8N
    if databases['n8n']:
        backup_path = os.path.join(backup_dir, "application_databases", "n8n_database.sqlite")
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        print(f"Backing up N8N database...")
        shutil.copy2(databases['n8n'], backup_path)
        print(f"  ✓ N8N database backed up")
        backed_up += 1
    
    # Backup Flowise
    if databases['flowise']:
        backup_path = os.path.join(backup_dir, "application_databases", "flowise_database.sqlite")
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        print(f"Backing up Flowise database...")
        shutil.copy2(databases['flowise'], backup_path)
        print(f"  ✓ Flowise database backed up")
        backed_up += 1
    
    # Create manifest
    manifest = {
        'backup_date': datetime.now().isoformat(),
        'backup_directory': backup_dir,
        'databases_backed_up': backed_up,
        'memory_dbs': len(databases['memory_dbs']),
        'files': []
    }
    
    # List all backed up files
    for root, dirs, files in os.walk(backup_dir):
        for file in files:
            if file.endswith(('.db', '.sqlite')):
                manifest['files'].append(os.path.join(root, file).replace(backup_dir + '/', ''))
    
    # Save manifest
    with open(os.path.join(backup_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print()
    print(f"=== Backup Complete ===")
    print(f"Total databases backed up: {backed_up}")
    print(f"Backup location: {backup_dir}")
    print(f"Manifest: {backup_dir}/manifest.json")
    
    return backup_dir

if __name__ == "__main__":
    backup_databases()