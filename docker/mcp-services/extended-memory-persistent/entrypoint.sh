#!/bin/bash
set -e

# Initialize database if needed
echo "Initializing Extended Memory MCP Service..."
echo "Database path: ${SQLITE_PATH}"
echo "Service port: ${SERVICE_PORT}"
echo "Cache enabled: ${ENABLE_CACHE}"

# Ensure the directory exists and has correct permissions
mkdir -p $(dirname ${SQLITE_PATH})
touch ${SQLITE_PATH}

# Check if we need to migrate from old data
if [ -f "/var/lib/mcp/memory.json" ]; then
    echo "Found legacy memory.json, migrating to SQLite..."
    python3 -c "
import json
import sqlite3
import os
from datetime import datetime

db_path = os.environ.get('SQLITE_PATH', '/var/lib/mcp/extended_memory.db')
legacy_path = '/var/lib/mcp/memory.json'

if os.path.exists(legacy_path):
    with open(legacy_path, 'r') as f:
        legacy_data = json.load(f)
    
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS memory_store (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            type TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            access_count INTEGER DEFAULT 1
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Migrate data
    for key, value in legacy_data.items():
        value_str = json.dumps(value)
        type_str = type(value).__name__
        cursor.execute(
            'INSERT OR REPLACE INTO memory_store (key, value, type) VALUES (?, ?, ?)',
            (key, value_str, type_str)
        )
    
    conn.commit()
    conn.close()
    
    # Rename legacy file
    os.rename(legacy_path, legacy_path + '.migrated')
    print(f'Migrated {len(legacy_data)} items from legacy storage')
"
fi

# Start the server
echo "Starting Extended Memory MCP Server..."
exec python3 /app/server.py