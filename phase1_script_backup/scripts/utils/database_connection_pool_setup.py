#!/usr/bin/env python3
"""
Database Connection Pool Setup and Configuration
Optimizes PostgreSQL for the SutazAI application workload
"""

import psycopg2
import logging
from contextlib import contextmanager
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_CONFIG = {
    'host': 'localhost',
    'port': 10000,
    'database': 'sutazai',
    'user': 'sutazai',
    'password': 'sutazai_secure_2024'
}

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        yield conn
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        raise e
    finally:
        if conn:
            conn.close()

def optimize_postgresql_settings():
    """Apply PostgreSQL optimizations for SutazAI workload"""
    
    optimizations = {
        # Connection settings
        'max_connections': '100',  # Reasonable for container environment
        'superuser_reserved_connections': '3',
        
        # Memory settings (optimized for container)
        'shared_buffers': '128MB',  # 25% of available container memory (512MB)
        'effective_cache_size': '384MB',  # 75% of available memory
        'work_mem': '4MB',  # Per operation memory
        'maintenance_work_mem': '64MB',
        
        # WAL settings
        'wal_buffers': '16MB',
        'checkpoint_completion_target': '0.9',
        'max_wal_size': '1GB',
        'min_wal_size': '80MB',
        
        # Query planner settings
        'random_page_cost': '1.1',  # SSD optimized
        'effective_io_concurrency': '200',  # SSD optimized
        'default_statistics_target': '100',
        
        # Logging settings
        'log_statement': 'none',  # Reduce logging overhead
        'log_duration': 'off',
        'log_min_duration_statement': '1000',  # Log slow queries (>1s)
        
        # Autovacuum settings
        'autovacuum': 'on',
        'autovacuum_max_workers': '3',
        'autovacuum_naptime': '1min',
        'autovacuum_vacuum_scale_factor': '0.2',
        'autovacuum_analyze_scale_factor': '0.1',
        
        # Connection timeout settings
        'tcp_keepalives_idle': '600',
        'tcp_keepalives_interval': '30',
        'tcp_keepalives_count': '3',
        'statement_timeout': '300000',  # 5 minutes
        'idle_in_transaction_session_timeout': '600000',  # 10 minutes
        
        # JSON/JSONB optimizations
        'default_text_search_config': 'pg_catalog.english'
    }
    
    logger.info("🔧 Applying PostgreSQL optimizations...")
    
    try:
        with get_db_connection() as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                # Apply settings that can be changed at runtime
                runtime_settings = [
                    'work_mem', 'maintenance_work_mem', 'default_statistics_target',
                    'log_statement', 'log_duration', 'log_min_duration_statement',
                    'statement_timeout', 'idle_in_transaction_session_timeout'
                ]
                
                for setting, value in optimizations.items():
                    if setting in runtime_settings:
                        try:
                            cur.execute(f"ALTER SYSTEM SET {setting} = %s", (value,))
                            logger.info(f"✅ Set {setting} = {value}")
                        except Exception as e:
                            logger.warning(f"⚠️ Could not set {setting}: {e}")
                
                # Reload configuration
                cur.execute("SELECT pg_reload_conf()")
                logger.info("✅ PostgreSQL configuration reloaded")
                
        logger.info("✅ Runtime optimizations applied")
        
    except Exception as e:
        logger.error(f"❌ Failed to apply optimizations: {e}")

def create_connection_monitoring():
    """Create monitoring for database connections"""
    
    monitoring_queries = """
    -- Connection monitoring view
    CREATE OR REPLACE VIEW connection_stats AS
    SELECT 
        datname,
        usename,
        application_name,
        client_addr,
        state,
        query_start,
        state_change,
        EXTRACT(EPOCH FROM (now() - query_start)) as query_duration_seconds
    FROM pg_stat_activity 
    WHERE datname = 'sutazai'
    ORDER BY query_start DESC;
    
    -- Connection summary view
    CREATE OR REPLACE VIEW connection_summary AS
    SELECT 
        state,
        COUNT(*) as connection_count,
        AVG(EXTRACT(EPOCH FROM (now() - state_change))) as avg_state_duration
    FROM pg_stat_activity 
    WHERE datname = 'sutazai'
    GROUP BY state
    ORDER BY connection_count DESC;
    
    -- Database activity monitoring
    CREATE OR REPLACE VIEW db_activity_monitor AS
    SELECT 
        'Total Connections' as metric,
        COUNT(*)::text as value
    FROM pg_stat_activity 
    WHERE datname = 'sutazai'
    UNION ALL
    SELECT 
        'Active Queries' as metric,
        COUNT(*)::text as value
    FROM pg_stat_activity 
    WHERE datname = 'sutazai' AND state = 'active'
    UNION ALL
    SELECT 
        'Idle Connections' as metric,
        COUNT(*)::text as value
    FROM pg_stat_activity 
    WHERE datname = 'sutazai' AND state = 'idle'
    UNION ALL
    SELECT 
        'Long Running Queries' as metric,
        COUNT(*)::text as value
    FROM pg_stat_activity 
    WHERE datname = 'sutazai' 
    AND state = 'active' 
    AND query_start < NOW() - INTERVAL '30 seconds';
    """
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(monitoring_queries)
                logger.info("✅ Connection monitoring views created")
    
    except Exception as e:
        logger.error(f"❌ Failed to create monitoring views: {e}")

def setup_database_extensions():
    """Enable useful PostgreSQL extensions"""
    extensions = [
        'uuid-ossp',      # UUID generation
        'btree_gin',      # Better JSONB indexing
        'pg_trgm',        # Trigram matching for text search
        'unaccent'        # Remove accents from text
    ]
    
    logger.info("📦 Setting up database extensions...")
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                for extension in extensions:
                    try:
                        cur.execute(f"CREATE EXTENSION IF NOT EXISTS \"{extension}\"")
                        logger.info(f"✅ Extension enabled: {extension}")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not enable {extension}: {e}")
    
    except Exception as e:
        logger.error(f"❌ Failed to setup extensions: {e}")

def create_performance_indexes():
    """Create additional performance indexes"""
    
    index_queries = [
        # JSONB performance indexes
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_capabilities_gin ON agents USING gin(capabilities)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tasks_payload_gin ON tasks USING gin(payload)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tasks_result_gin ON tasks USING gin(result)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_system_metrics_tags_gin ON system_metrics USING gin(tags)",
        
        # Composite indexes for common queries
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tasks_status_created_at ON tasks(status, created_at DESC)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_health_agent_status ON agent_health(agent_id, status, last_heartbeat DESC)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chat_history_user_created ON chat_history(user_id, created_at DESC)",
        
        # Partial indexes for active records
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_active ON agents(name) WHERE is_active = true",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_active ON users(username) WHERE is_active = true",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sessions_active ON sessions(user_id, expires_at) WHERE is_active = true",
        
        # Text search indexes
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tasks_title_trgm ON tasks USING gin(title gin_trgm_ops)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_documents_title_trgm ON knowledge_documents USING gin(title gin_trgm_ops)"
    ]
    
    logger.info("🚀 Creating performance indexes...")
    
    try:
        with get_db_connection() as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                for query in index_queries:
                    try:
                        logger.info(f"  Creating index: {query.split('idx_')[1].split(' ')[0] if 'idx_' in query else 'unnamed'}")
                        cur.execute(query)
                        logger.info("  ✅ Index created")
                    except Exception as e:
                        if "already exists" in str(e):
                            logger.info("  ⚠️ Index already exists")
                        else:
                            logger.error(f"  ❌ Failed to create index: {e}")
    
    except Exception as e:
        logger.error(f"❌ Failed to create indexes: {e}")

def analyze_database_performance():
    """Analyze current database performance"""
    
    analysis_queries = {
        'table_sizes': """
            SELECT 
                tablename,
                pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
            FROM pg_tables 
            WHERE schemaname = 'public'
            ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
            LIMIT 10
        """,
        
        'index_usage': """
            SELECT 
                schemaname,
                tablename,
                indexname,
                idx_tup_read,
                idx_tup_fetch,
                idx_scan
            FROM pg_stat_user_indexes
            WHERE schemaname = 'public'
            ORDER BY idx_scan DESC
            LIMIT 10
        """,
        
        'slow_queries': """
            SELECT 
                query,
                calls,
                total_time,
                mean_time,
                rows
            FROM pg_stat_statements
            WHERE query NOT LIKE '%pg_stat_%'
            ORDER BY mean_time DESC
            LIMIT 5
        """,
        
        'database_stats': """
            SELECT 
                'Database Size' as metric,
                pg_size_pretty(pg_database_size('sutazai')) as value
            UNION ALL
            SELECT 
                'Cache Hit Ratio' as metric,
                ROUND(
                    (blks_hit::float / (blks_hit + blks_read) * 100)::numeric, 2
                )::text || '%' as value
            FROM pg_stat_database 
            WHERE datname = 'sutazai'
        """
    }
    
    logger.info("📊 Analyzing database performance...")
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                for analysis_name, query in analysis_queries.items():
                    try:
                        cur.execute(query)
                        results = cur.fetchall()
                        
                        logger.info(f"\n{analysis_name.upper().replace('_', ' ')}:")
                        for row in results:
                            logger.info(f"  {' | '.join(map(str, row))}")
                    
                    except Exception as e:
                        logger.warning(f"⚠️ Could not run {analysis_name}: {e}")
    
    except Exception as e:
        logger.error(f"❌ Performance analysis failed: {e}")

def create_maintenance_functions():
    """Create database maintenance functions"""
    
    functions = """
    -- Function to update agent health
    CREATE OR REPLACE FUNCTION update_agent_health_status(
        agent_name_param TEXT,
        status_param TEXT,
        cpu_usage_param DECIMAL DEFAULT NULL,
        memory_usage_param DECIMAL DEFAULT NULL,
        response_time_param DECIMAL DEFAULT NULL
    ) RETURNS BOOLEAN AS $$
    DECLARE
        agent_id_var INTEGER;
    BEGIN
        -- Get agent ID
        SELECT id INTO agent_id_var FROM agents WHERE name = agent_name_param;
        
        IF agent_id_var IS NULL THEN
            RETURN FALSE;
        END IF;
        
        -- Insert or update health record
        INSERT INTO agent_health (
            agent_id, status, last_heartbeat, cpu_usage, memory_usage, response_time
        ) VALUES (
            agent_id_var, status_param, CURRENT_TIMESTAMP, 
            cpu_usage_param, memory_usage_param, response_time_param
        )
        ON CONFLICT (agent_id) DO UPDATE SET
            status = EXCLUDED.status,
            last_heartbeat = EXCLUDED.last_heartbeat,
            cpu_usage = COALESCE(EXCLUDED.cpu_usage, agent_health.cpu_usage),
            memory_usage = COALESCE(EXCLUDED.memory_usage, agent_health.memory_usage),
            response_time = COALESCE(EXCLUDED.response_time, agent_health.response_time);
        
        RETURN TRUE;
    END;
    $$ LANGUAGE plpgsql;
    
    -- Function to log API usage
    CREATE OR REPLACE FUNCTION log_api_usage(
        endpoint_param TEXT,
        method_param TEXT,
        response_code_param INTEGER,
        response_time_param DECIMAL,
        user_id_param INTEGER DEFAULT NULL,
        agent_id_param INTEGER DEFAULT NULL
    ) RETURNS VOID AS $$
    BEGIN
        INSERT INTO api_usage_logs (
            endpoint, method, response_code, response_time, user_id, agent_id
        ) VALUES (
            endpoint_param, method_param, response_code_param, 
            response_time_param, user_id_param, agent_id_param
        );
    END;
    $$ LANGUAGE plpgsql;
    
    -- Function to create system alert
    CREATE OR REPLACE FUNCTION create_system_alert(
        alert_type_param TEXT,
        severity_param TEXT,
        title_param TEXT,
        description_param TEXT,
        source_param TEXT DEFAULT NULL,
        metadata_param JSONB DEFAULT '{}'
    ) RETURNS INTEGER AS $$
    DECLARE
        alert_id INTEGER;
    BEGIN
        INSERT INTO system_alerts (
            alert_type, severity, title, description, source, metadata
        ) VALUES (
            alert_type_param, severity_param, title_param, 
            description_param, source_param, metadata_param
        ) RETURNING id INTO alert_id;
        
        RETURN alert_id;
    END;
    $$ LANGUAGE plpgsql;
    """
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(functions)
                logger.info("✅ Maintenance functions created")
    
    except Exception as e:
        logger.error(f"❌ Failed to create functions: {e}")

def main():
    """Main setup function"""
    logger.info("🔧 SutazAI Database Connection Pool Setup")
    logger.info("=" * 50)
    
    try:
        # Step 1: Setup extensions
        setup_database_extensions()
        
        # Step 2: Apply PostgreSQL optimizations
        optimize_postgresql_settings()
        
        # Step 3: Create performance indexes
        create_performance_indexes()
        
        # Step 4: Setup monitoring
        create_connection_monitoring()
        
        # Step 5: Create maintenance functions
        create_maintenance_functions()
        
        # Step 6: Analyze performance
        analyze_database_performance()
        
        logger.info("\n" + "=" * 50)
        logger.info("✅ Database optimization completed successfully!")
        logger.info("🚀 PostgreSQL is now optimized for SutazAI workload")
        
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())