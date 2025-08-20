#!/usr/bin/env python3
"""
SQLite to PostgreSQL Migration Script
Consolidates multiple SQLite databases into unified PostgreSQL database
"""

import os
import sys
import json
import sqlite3
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from datetime import datetime
import hashlib
import argparse
import logging
from typing import Dict, List, Tuple, Any
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatabaseMigrator:
    """Handles migration from SQLite to PostgreSQL"""
    
    def __init__(self, pg_config: Dict[str, str], dry_run: bool = False):
        """
        Initialize the migrator
        
        Args:
            pg_config: PostgreSQL connection configuration
            dry_run: If True, only simulate migration without actual data transfer
        """
        self.pg_config = pg_config
        self.dry_run = dry_run
        self.pg_conn = None
        self.pg_cursor = None
        self.stats = {
            'total_records': 0,
            'migrated_records': 0,
            'failed_records': 0,
            'databases_processed': 0
        }
        
    def connect_postgres(self):
        """Establish PostgreSQL connection"""
        try:
            self.pg_conn = psycopg2.connect(**self.pg_config)
            self.pg_cursor = self.pg_conn.cursor(cursor_factory=RealDictCursor)
            logger.info("Connected to PostgreSQL successfully")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            sys.exit(1)
            
    def close_postgres(self):
        """Close PostgreSQL connection"""
        if self.pg_cursor:
            self.pg_cursor.close()
        if self.pg_conn:
            self.pg_conn.close()
            
    def calculate_checksum(self, data: Dict) -> str:
        """Calculate checksum for data verification"""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
        
    def migrate_memory_db(self, sqlite_path: str) -> Tuple[int, int]:
        """
        Migrate a single memory.db SQLite database
        
        Returns:
            Tuple of (success_count, failure_count)
        """
        logger.info(f"Processing: {sqlite_path}")
        
        if not os.path.exists(sqlite_path):
            logger.warning(f"Database not found: {sqlite_path}")
            return 0, 0
            
        success_count = 0
        failure_count = 0
        source_db_name = os.path.basename(os.path.dirname(sqlite_path))
        
        try:
            # Connect to SQLite
            sqlite_conn = sqlite3.connect(sqlite_path)
            sqlite_conn.row_factory = sqlite3.Row
            sqlite_cursor = sqlite_conn.cursor()
            
            # Get all records
            sqlite_cursor.execute("""
                SELECT key, value, namespace, metadata, 
                       created_at, updated_at, accessed_at, 
                       access_count, ttl, expires_at
                FROM memory_entries
            """)
            
            records = sqlite_cursor.fetchall()
            total_records = len(records)
            logger.info(f"Found {total_records} records to migrate")
            
            # Start migration transaction
            migration_start = datetime.now()
            
            if not self.dry_run:
                # Record migration metadata
                self.pg_cursor.execute("""
                    INSERT INTO migration_metadata 
                    (source_file, records_migrated, migration_started, status)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                """, (sqlite_path, 0, migration_start, 'in_progress'))
                migration_id = self.pg_cursor.fetchone()['id']
            
            # Migrate each record
            for record in records:
                try:
                    # Convert SQLite row to dict
                    row_dict = dict(record)
                    
                    # Parse metadata if it exists
                    metadata = None
                    if row_dict['metadata']:
                        try:
                            metadata = json.loads(row_dict['metadata'])
                        except:
                            metadata = {'raw': row_dict['metadata']}
                    
                    # Parse value
                    try:
                        value = json.loads(row_dict['value'])
                    except:
                        value = {'raw': row_dict['value']}
                    
                    # Determine data type
                    data_type = 'json'
                    if isinstance(value, str):
                        data_type = 'string'
                    elif isinstance(value, (int, float)):
                        data_type = 'number'
                    elif isinstance(value, bool):
                        data_type = 'boolean'
                    elif isinstance(value, list):
                        data_type = 'array'
                    elif isinstance(value, dict):
                        data_type = 'object'
                    
                    if not self.dry_run:
                        # Insert into PostgreSQL
                        self.pg_cursor.execute("""
                            INSERT INTO unified_memory 
                            (key, value, namespace, data_type, metadata, 
                             source_db, source_path, created_at, updated_at, 
                             accessed_at, access_count, ttl, expires_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, 
                                    to_timestamp(%s), to_timestamp(%s), 
                                    to_timestamp(%s), %s, %s, 
                                    CASE WHEN %s IS NOT NULL THEN to_timestamp(%s) ELSE NULL END)
                            ON CONFLICT (key, namespace) 
                            DO UPDATE SET 
                                value = EXCLUDED.value,
                                updated_at = EXCLUDED.updated_at,
                                access_count = unified_memory.access_count + 1
                        """, (
                            row_dict['key'],
                            Json(value),
                            row_dict['namespace'],
                            data_type,
                            Json(metadata) if metadata else None,
                            source_db_name,
                            sqlite_path,
                            row_dict['created_at'],
                            row_dict['updated_at'] or row_dict['created_at'],
                            row_dict['accessed_at'] or row_dict['created_at'],
                            row_dict['access_count'] or 0,
                            row_dict['ttl'],
                            row_dict['expires_at'],
                            row_dict['expires_at']
                        ))
                    
                    success_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to migrate record {row_dict.get('key', 'unknown')}: {e}")
                    failure_count += 1
            
            if not self.dry_run:
                # Update migration metadata
                self.pg_cursor.execute("""
                    UPDATE migration_metadata 
                    SET records_migrated = %s, 
                        migration_completed = %s,
                        status = %s,
                        checksum = %s
                    WHERE id = %s
                """, (
                    success_count, 
                    datetime.now(), 
                    'completed' if failure_count == 0 else 'completed_with_errors',
                    self.calculate_checksum({'count': success_count}),
                    migration_id
                ))
                
                # Commit transaction
                self.pg_conn.commit()
            
            sqlite_conn.close()
            
            logger.info(f"Migration complete: {success_count} success, {failure_count} failures")
            
        except Exception as e:
            logger.error(f"Migration failed for {sqlite_path}: {e}")
            if not self.dry_run:
                self.pg_conn.rollback()
            return 0, total_records
            
        return success_count, failure_count
        
    def migrate_extended_memory(self, sqlite_path: str) -> Tuple[int, int]:
        """
        Migrate extended memory database with different schema
        """
        logger.info(f"Processing extended memory: {sqlite_path}")
        
        if not os.path.exists(sqlite_path):
            logger.warning(f"Extended memory database not found: {sqlite_path}")
            return 0, 0
            
        success_count = 0
        failure_count = 0
        
        try:
            sqlite_conn = sqlite3.connect(sqlite_path)
            sqlite_conn.row_factory = sqlite3.Row
            sqlite_cursor = sqlite_conn.cursor()
            
            # Get all records from memory_store
            sqlite_cursor.execute("""
                SELECT key, value, type, created_at, updated_at, 
                       accessed_at, access_count
                FROM memory_store
            """)
            
            records = sqlite_cursor.fetchall()
            total_records = len(records)
            logger.info(f"Found {total_records} extended memory records to migrate")
            
            for record in records:
                try:
                    row_dict = dict(record)
                    
                    # Parse value based on type
                    value = row_dict['value']
                    data_type = row_dict['type']
                    
                    try:
                        if data_type in ['dict', 'list']:
                            value = json.loads(value)
                        elif data_type == 'int':
                            value = int(value)
                        elif data_type == 'float':
                            value = float(value)
                        elif data_type == 'bool':
                            value = value.lower() == 'true'
                        elif data_type == 'none':
                            value = None
                    except:
                        pass
                    
                    if not self.dry_run:
                        self.pg_cursor.execute("""
                            INSERT INTO unified_memory 
                            (key, value, namespace, data_type, 
                             source_db, source_path, created_at, 
                             updated_at, accessed_at, access_count)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (key, namespace) 
                            DO UPDATE SET 
                                value = EXCLUDED.value,
                                updated_at = EXCLUDED.updated_at,
                                access_count = unified_memory.access_count + 1
                        """, (
                            row_dict['key'],
                            Json(value) if value is not None else Json({}),
                            'extended_memory',
                            data_type,
                            'extended_memory',
                            sqlite_path,
                            row_dict['created_at'],
                            row_dict['updated_at'],
                            row_dict['accessed_at'],
                            row_dict['access_count'] or 1
                        ))
                    
                    success_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to migrate extended memory record {row_dict.get('key', 'unknown')}: {e}")
                    failure_count += 1
            
            if not self.dry_run:
                self.pg_conn.commit()
            
            sqlite_conn.close()
            
        except Exception as e:
            logger.error(f"Extended memory migration failed: {e}")
            if not self.dry_run:
                self.pg_conn.rollback()
            return 0, total_records
            
        return success_count, failure_count
        
    def run_migration(self, sqlite_databases: List[str], extended_memory_path: str = None):
        """
        Run the complete migration process
        """
        self.connect_postgres()
        
        try:
            # Migrate regular memory databases
            for db_path in sqlite_databases:
                if os.path.exists(db_path):
                    success, failure = self.migrate_memory_db(db_path)
                    self.stats['migrated_records'] += success
                    self.stats['failed_records'] += failure
                    self.stats['databases_processed'] += 1
            
            # Migrate extended memory if provided
            if extended_memory_path:
                success, failure = self.migrate_extended_memory(extended_memory_path)
                self.stats['migrated_records'] += success
                self.stats['failed_records'] += failure
                self.stats['databases_processed'] += 1
            
            # Print summary
            logger.info("=" * 60)
            logger.info("MIGRATION SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Databases processed: {self.stats['databases_processed']}")
            logger.info(f"Records migrated: {self.stats['migrated_records']}")
            logger.info(f"Records failed: {self.stats['failed_records']}")
            logger.info(f"Success rate: {(self.stats['migrated_records'] / (self.stats['migrated_records'] + self.stats['failed_records']) * 100):.2f}%" if (self.stats['migrated_records'] + self.stats['failed_records']) > 0 else "N/A")
            
        finally:
            self.close_postgres()

def find_all_memory_databases(base_path: str = '/opt/sutazaiapp') -> List[str]:
    """Find all memory.db files in the project (excluding backups)"""
    databases = []
    for root, dirs, files in os.walk(base_path):
        # Skip backup and archive directories
        if 'backups' in root or 'archives' in root:
            continue
        if 'memory.db' in files:
            databases.append(os.path.join(root, 'memory.db'))
    return sorted(databases)

def main():
    parser = argparse.ArgumentParser(description='Migrate SQLite databases to PostgreSQL')
    parser.add_argument('--dry-run', action='store_true', help='Simulate migration without actual data transfer')
    parser.add_argument('--host', default='localhost', help='PostgreSQL host')
    parser.add_argument('--port', type=int, default=10000, help='PostgreSQL port')
    parser.add_argument('--database', default='sutazai', help='PostgreSQL database name')
    parser.add_argument('--user', default='sutazai', help='PostgreSQL username')
    parser.add_argument('--password', default='change_me_secure', help='PostgreSQL password')
    
    args = parser.parse_args()
    
    # PostgreSQL configuration
    # Check if we can connect via Docker or directly
    import subprocess
    try:
        # Try to get Docker container IP
        result = subprocess.run(['docker', 'inspect', 'sutazai-postgres', '-f', '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}'], 
                              capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            # Use Docker container IP
            pg_host = result.stdout.strip()
            pg_port = 5432  # Internal PostgreSQL port
        else:
            pg_host = args.host
            pg_port = args.port
    except:
        pg_host = args.host
        pg_port = args.port
    
    pg_config = {
        'host': pg_host,
        'port': pg_port,
        'database': args.database,
        'user': args.user,
        'password': args.password
    }
    
    # Find all databases
    memory_databases = find_all_memory_databases()
    extended_memory_path = '/opt/sutazaiapp/data/mcp/extended-memory/extended_memory.db'
    
    logger.info(f"Found {len(memory_databases)} memory.db files")
    
    # Create migrator and run
    migrator = DatabaseMigrator(pg_config, dry_run=args.dry_run)
    migrator.run_migration(memory_databases, extended_memory_path)

if __name__ == '__main__':
    main()