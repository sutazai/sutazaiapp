#!/usr/bin/env python3
"""
SutazAI Database Operations Script
Provides backup, restore, and maintenance operations for PostgreSQL
"""

import os
import sys
import subprocess
import psycopg2
from datetime import datetime, timedelta
from pathlib import Path
import logging
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': '10000',
    'database': 'sutazai',
    'user': 'sutazai',
    'password': 'sutazai_secure_2024'
}

BACKUP_DIR = Path('/opt/sutazaiapp/backups/database')
BACKUP_RETENTION_DAYS = 7  # Keep backups for 7 days

def ensure_backup_dir():
    """Ensure backup directory exists"""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

def create_backup(backup_type='full'):
    """Create database backup"""
    ensure_backup_dir()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_filename = f"sutazai_backup_{backup_type}_{timestamp}.sql"
    backup_path = BACKUP_DIR / backup_filename
    
    # Set environment variable for password
    env = os.environ.copy()
    env['PGPASSWORD'] = DB_CONFIG['password']
    
    cmd = [
        'docker', 'exec', 'sutazai-postgres', 'pg_dump',
        '-h', 'localhost',
        '-p', '5432',
        '-U', DB_CONFIG['user'],
        '-d', DB_CONFIG['database'],
        '--verbose',
        '--no-password'
    ]
    
    if backup_type == 'schema_only':
        cmd.append('--schema-only')
    elif backup_type == 'data_only':
        cmd.append('--data-only')
    
    try:
        logger.info(f"Creating {backup_type} backup: {backup_filename}")
        
        with open(backup_path, 'w') as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, 
                                 env=env, text=True, check=True)
        
        # Verify backup was created and has content
        if backup_path.exists() and backup_path.stat().st_size > 1000:
            logger.info(f"✅ Backup created successfully: {backup_path}")
            logger.info(f"💾 Backup size: {backup_path.stat().st_size / (1024*1024):.2f} MB")
            return backup_path
        else:
            logger.error("❌ Backup file is empty or too small")
            return None
            
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Backup failed: {e}")
        logger.error(f"❌ Error output: {e.stderr}")
        return None
    except Exception as e:
        logger.error(f"❌ Backup failed: {e}")
        return None

def restore_backup(backup_path: Path):
    """Restore database from backup"""
    if not backup_path.exists():
        logger.error(f"❌ Backup file not found: {backup_path}")
        return False
    
    env = os.environ.copy()
    env['PGPASSWORD'] = DB_CONFIG['password']
    
    # First, copy the backup file to container
    docker_cp_cmd = ['docker', 'cp', str(backup_path), 'sutazai-postgres:/tmp/restore.sql']
    
    try:
        subprocess.run(docker_cp_cmd, check=True)
        logger.info(f"📋 Copied backup file to container")
        
        # Then restore from inside container
        restore_cmd = [
            'docker', 'exec', 'sutazai-postgres',
            'psql', '-U', DB_CONFIG['user'], '-d', DB_CONFIG['database'],
            '-f', '/tmp/restore.sql'
        ]
        
        result = subprocess.run(restore_cmd, capture_output=True, text=True, env=env)
        
        if result.returncode == 0:
            logger.info("✅ Database restored successfully")
            return True
        else:
            logger.error(f"❌ Restore failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Restore failed: {e}")
        return False

def cleanup_old_backups():
    """Remove old backup files"""
    if not BACKUP_DIR.exists():
        return
    
    cutoff_date = datetime.now() - timedelta(days=BACKUP_RETENTION_DAYS)
    
    deleted_count = 0
    total_size_freed = 0
    
    for backup_file in BACKUP_DIR.glob('*.sql'):
        try:
            file_mtime = datetime.fromtimestamp(backup_file.stat().st_mtime)
            if file_mtime < cutoff_date:
                file_size = backup_file.stat().st_size
                backup_file.unlink()
                deleted_count += 1
                total_size_freed += file_size
                logger.info(f"🗑️ Deleted old backup: {backup_file.name}")
        except Exception as e:
            logger.error(f"❌ Failed to delete {backup_file.name}: {e}")
    
    if deleted_count > 0:
        logger.info(f"✅ Cleaned up {deleted_count} old backups, freed {total_size_freed/(1024*1024):.2f} MB")
    else:
        logger.info("✅ No old backups to clean up")

def vacuum_analyze():
    """Perform VACUUM ANALYZE on all tables"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = True
        
        with conn.cursor() as cur:
            # Get all user tables
            cur.execute("""
                SELECT tablename 
                FROM pg_tables 
                WHERE schemaname = 'public'
                ORDER BY tablename
            """)
            tables = [row[0] for row in cur.fetchall()]
            
            logger.info(f"🧹 Starting VACUUM ANALYZE on {len(tables)} tables...")
            
            for table in tables:
                logger.info(f"  Vacuuming {table}...")
                cur.execute(f"VACUUM ANALYZE {table}")
            
            logger.info("✅ VACUUM ANALYZE completed")
            
    except Exception as e:
        logger.error(f"❌ VACUUM ANALYZE failed: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

def update_table_statistics():
    """Update table statistics"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        
        with conn.cursor() as cur:
            # Update statistics
            cur.execute("ANALYZE")
            
            # Get table statistics
            cur.execute("""
                SELECT 
                    schemaname,
                    tablename,
                    n_tup_ins as inserts,
                    n_tup_upd as updates,
                    n_tup_del as deletes,
                    n_live_tup as live_rows,
                    n_dead_tup as dead_rows,
                    last_vacuum,
                    last_analyze
                FROM pg_stat_user_tables 
                WHERE schemaname = 'public'
                ORDER BY tablename
            """)
            
            stats = cur.fetchall()
            
            logger.info("📊 Table Statistics:")
            for stat in stats:
                schema, table, ins, upd, del_, live, dead, vacuum, analyze = stat
                logger.info(f"  {table}: {live} live rows, {dead} dead rows, last vacuum: {vacuum}")
        
        conn.close()
        logger.info("✅ Statistics updated")
        
    except Exception as e:
        logger.error(f"❌ Statistics update failed: {e}")

def check_database_size():
    """Check database and table sizes"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        
        with conn.cursor() as cur:
            # Database size
            cur.execute("SELECT pg_size_pretty(pg_database_size('sutazai'))")
            db_size = cur.fetchone()[0]
            logger.info(f"💾 Database size: {db_size}")
            
            # Table sizes
            cur.execute("""
                SELECT 
                    tablename,
                    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
                    pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
                FROM pg_tables 
                WHERE schemaname = 'public'
                ORDER BY size_bytes DESC
            """)
            
            tables = cur.fetchall()
            logger.info("📊 Table sizes:")
            for table, size, size_bytes in tables[:10]:  # Top 10 tables
                logger.info(f"  {table}: {size}")
        
        conn.close()
        
    except Exception as e:
        logger.error(f"❌ Size check failed: {e}")

def check_active_connections():
    """Check active database connections"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    datname,
                    usename,
                    application_name,
                    client_addr,
                    state,
                    query_start,
                    state_change
                FROM pg_stat_activity 
                WHERE datname = 'sutazai'
                AND state = 'active'
                ORDER BY query_start
            """)
            
            connections = cur.fetchall()
            
            if connections:
                logger.info(f"🔌 Active connections ({len(connections)}):")
                for conn_info in connections:
                    db, user, app, addr, state, start, change = conn_info
                    logger.info(f"  {user}@{addr or 'local'} via {app}: {state}")
            else:
                logger.info("🔌 No active connections (besides this one)")
        
        conn.close()
        
    except Exception as e:
        logger.error(f"❌ Connection check failed: {e}")

def create_monitoring_view():
    """Create or update monitoring views"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        
        with conn.cursor() as cur:
            # Database health monitoring view
            cur.execute("""
                CREATE OR REPLACE VIEW db_health_monitor AS
                SELECT 
                    'Database Size' as metric,
                    pg_size_pretty(pg_database_size('sutazai')) as value,
                    CURRENT_TIMESTAMP as updated_at
                UNION ALL
                SELECT 
                    'Active Connections' as metric,
                    COUNT(*)::text as value,
                    CURRENT_TIMESTAMP as updated_at
                FROM pg_stat_activity 
                WHERE datname = 'sutazai' AND state = 'active'
                UNION ALL
                SELECT 
                    'Total Tables' as metric,
                    COUNT(*)::text as value,
                    CURRENT_TIMESTAMP as updated_at
                FROM pg_tables 
                WHERE schemaname = 'public'
                UNION ALL
                SELECT 
                    'Total Indexes' as metric,
                    COUNT(*)::text as value,
                    CURRENT_TIMESTAMP as updated_at
                FROM pg_indexes 
                WHERE schemaname = 'public'
            """)
            
            conn.commit()
            logger.info("✅ Monitoring views updated")
        
        conn.close()
        
    except Exception as e:
        logger.error(f"❌ Monitoring view creation failed: {e}")

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python database_operations.py <command> [options]")
        print("\nCommands:")
        print("  backup [full|schema|data]  - Create database backup")
        print("  restore <backup_file>      - Restore from backup")
        print("  cleanup                    - Remove old backups")
        print("  vacuum                     - Run VACUUM ANALYZE")
        print("  stats                      - Update table statistics")
        print("  size                       - Check database sizes")
        print("  connections               - Check active connections")
        print("  monitor                    - Update monitoring views")
        print("  maintenance               - Run full maintenance")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    logger.info(f"🔧 SutazAI Database Operations - {command.upper()}")
    logger.info("=" * 50)
    
    if command == 'backup':
        backup_type = sys.argv[2] if len(sys.argv) > 2 else 'full'
        if backup_type not in ['full', 'schema', 'data']:
            logger.error("❌ Invalid backup type. Use: full, schema, or data")
            sys.exit(1)
        
        backup_path = create_backup(backup_type)
        if backup_path:
            logger.info(f"✅ Backup completed: {backup_path}")
        else:
            logger.error("❌ Backup failed")
            sys.exit(1)
    
    elif command == 'restore':
        if len(sys.argv) < 3:
            logger.error("❌ Please specify backup file path")
            sys.exit(1)
        
        backup_path = Path(sys.argv[2])
        if restore_backup(backup_path):
            logger.info("✅ Restore completed successfully")
        else:
            logger.error("❌ Restore failed")
            sys.exit(1)
    
    elif command == 'cleanup':
        cleanup_old_backups()
    
    elif command == 'vacuum':
        vacuum_analyze()
    
    elif command == 'stats':
        update_table_statistics()
    
    elif command == 'size':
        check_database_size()
    
    elif command == 'connections':
        check_active_connections()
    
    elif command == 'monitor':
        create_monitoring_view()
    
    elif command == 'maintenance':
        logger.info("🔧 Running full database maintenance...")
        vacuum_analyze()
        update_table_statistics()
        check_database_size()
        check_active_connections()
        create_monitoring_view()
        cleanup_old_backups()
        logger.info("✅ Full maintenance completed")
    
    else:
        logger.error(f"❌ Unknown command: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()