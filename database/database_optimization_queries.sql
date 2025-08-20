-- 🔥 ULTRA DATABASE OPTIMIZATION QUERIES
-- SutazAI PostgreSQL Performance Optimization
-- Execute these queries to implement recommended optimizations

-- =============================================
-- SECTION 1: IMMEDIATE MAINTENANCE (Run First)
-- =============================================

-- 1.1: Check current dead tuple situation
SELECT 
    schemaname,
    relname as table_name,
    n_live_tup as live_rows,
    n_dead_tup as dead_rows,
    ROUND((n_dead_tup::float / NULLIF(n_live_tup + n_dead_tup, 0)) * 100, 2) as dead_percentage,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||relname)) as table_size,
    last_vacuum,
    last_autovacuum,
    last_analyze,
    last_autoanalyze
FROM pg_stat_user_tables 
ORDER BY dead_percentage DESC NULLS LAST;

-- 1.2: Fix critical dead tuple bloat (EXECUTE IMMEDIATELY)
VACUUM FULL users;           -- Fix 40% dead tuple bloat
VACUUM FULL agent_health;    -- Fix 100% dead tuple bloat

-- 1.3: Reindex affected tables after VACUUM FULL
REINDEX TABLE users;
REINDEX TABLE agent_health;

-- 1.4: Update table statistics
ANALYZE users;
ANALYZE agent_health;
ANALYZE agents;
ANALYZE tasks;

-- =============================================
-- SECTION 2: POSTGRESQL CONFIGURATION OPTIMIZATION
-- =============================================

-- 2.1: Enable JIT for complex query optimization
ALTER SYSTEM SET jit = on;
ALTER SYSTEM SET jit_above_cost = 100000;
ALTER SYSTEM SET jit_inline_above_cost = 500000;
ALTER SYSTEM SET jit_optimize_above_cost = 500000;

-- 2.2: Optimize memory settings for current workload
-- Current: shared_buffers=524288*8kB=4GB, work_mem=100MB 
-- Optimized for low concurrency, small dataset:
ALTER SYSTEM SET shared_buffers = '1GB';        -- Reduce from 4GB
ALTER SYSTEM SET work_mem = '16MB';              -- Reduce from 100MB  
ALTER SYSTEM SET maintenance_work_mem = '256MB'; -- Increase from 128MB
ALTER SYSTEM SET effective_cache_size = '2GB';  -- Reduce from 512MB*8kB

-- 2.3: Optimize autovacuum for better dead tuple management
ALTER SYSTEM SET autovacuum_vacuum_scale_factor = 0.1;    -- More aggressive (was 0.2)
ALTER SYSTEM SET autovacuum_analyze_scale_factor = 0.05;  -- More frequent stats (was 0.1)
ALTER SYSTEM SET autovacuum_vacuum_cost_limit = 2000;     -- Increase vacuum speed (was 200)
ALTER SYSTEM SET autovacuum_max_workers = 2;              -- Sufficient for small DB (was 3)

-- 2.4: Connection and performance tuning
ALTER SYSTEM SET max_connections = 200;                   -- Reduce from default
ALTER SYSTEM SET checkpoint_completion_target = 0.9;      -- Better I/O distribution
ALTER SYSTEM SET wal_buffers = '16MB';                    -- Optimize WAL writes
ALTER SYSTEM SET random_page_cost = 1.1;                  -- SSD optimization

-- 2.5: Apply configuration changes
SELECT pg_reload_conf();

-- Verify configuration changes applied
SELECT name, setting, unit, context 
FROM pg_settings 
WHERE name IN (
    'jit', 'shared_buffers', 'work_mem', 'maintenance_work_mem',
    'autovacuum_vacuum_scale_factor', 'autovacuum_analyze_scale_factor'
);

-- =============================================
-- SECTION 3: INDEX ANALYSIS AND OPTIMIZATION
-- =============================================

-- 3.1: Check index usage statistics
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_tup_read,
    idx_tup_fetch,
    idx_scan,
    CASE 
        WHEN idx_scan = 0 THEN 'UNUSED - Consider dropping'
        WHEN idx_scan < 100 THEN 'LOW usage'
        ELSE 'ACTIVE'
    END as index_status,
    pg_size_pretty(pg_relation_size(indexname::regclass)) as index_size
FROM pg_stat_user_indexes
ORDER BY idx_scan ASC;

-- 3.2: Find missing indexes (check for sequential scans on large tables)
SELECT 
    schemaname,
    relname as table_name,
    seq_scan,
    seq_tup_read,
    idx_scan,
    idx_tup_fetch,
    ROUND((seq_tup_read::float / NULLIF(seq_scan, 0)), 2) as avg_seq_reads_per_scan,
    CASE 
        WHEN seq_scan > idx_scan AND seq_tup_read > 1000 THEN 'NEEDS INDEX'
        ELSE 'OK'
    END as recommendation
FROM pg_stat_user_tables
WHERE seq_scan > 0
ORDER BY seq_tup_read DESC;

-- 3.3: Check for duplicate or redundant indexes
SELECT 
    t.schemaname,
    t.tablename,
    array_agg(t.indexname) as duplicate_indexes,
    array_agg(t.indexdef) as index_definitions
FROM (
    SELECT 
        schemaname,
        tablename,
        indexname,
        array_agg(attname ORDER BY attnum) as index_columns,
        indexdef
    FROM pg_indexes i
    JOIN pg_attribute a ON a.attrelid = (schemaname||'.'||tablename)::regclass
    WHERE i.schemaname = 'public'
    GROUP BY schemaname, tablename, indexname, indexdef
) t
GROUP BY t.schemaname, t.tablename, t.index_columns
HAVING count(*) > 1;

-- =============================================
-- SECTION 4: QUERY PERFORMANCE MONITORING
-- =============================================

-- 4.1: Enable pg_stat_statements for query monitoring (if not already enabled)
-- Add to postgresql.conf: shared_preload_libraries = 'pg_stat_statements'
-- Then restart PostgreSQL and run:
-- CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- 4.2: Monitor slow queries (requires pg_stat_statements)
-- Uncomment after enabling extension:
/*
SELECT 
    query,
    calls,
    total_exec_time,
    mean_exec_time,
    stddev_exec_time,
    rows,
    100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
FROM pg_stat_statements 
WHERE calls > 1
ORDER BY mean_exec_time DESC 
LIMIT 10;
*/

-- 4.3: Current active queries and their duration
SELECT 
    pid,
    now() - pg_stat_activity.query_start AS duration,
    query,
    state,
    client_addr
FROM pg_stat_activity 
WHERE (now() - pg_stat_activity.query_start) > interval '5 minutes'
AND state != 'idle';

-- =============================================
-- SECTION 5: TABLE MAINTENANCE AUTOMATION
-- =============================================

-- 5.1: Create function for automated maintenance reporting
CREATE OR REPLACE FUNCTION check_table_health()
RETURNS TABLE (
    table_name text,
    live_rows bigint,
    dead_rows bigint,
    dead_percentage numeric,
    table_size text,
    needs_vacuum boolean,
    last_vacuum timestamp,
    needs_analyze boolean,
    last_analyze timestamp
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        s.relname::text,
        s.n_live_tup,
        s.n_dead_tup,
        ROUND((s.n_dead_tup::float / NULLIF(s.n_live_tup + s.n_dead_tup, 0)) * 100, 2),
        pg_size_pretty(pg_total_relation_size(s.schemaname||'.'||s.relname)),
        (s.n_dead_tup::float / NULLIF(s.n_live_tup + s.n_dead_tup, 0)) > 0.1, -- >10% dead
        s.last_vacuum,
        s.last_analyze < (now() - interval '1 day'), -- analyze if >1 day old
        s.last_analyze
    FROM pg_stat_user_tables s
    ORDER BY (s.n_dead_tup::float / NULLIF(s.n_live_tup + s.n_dead_tup, 0)) DESC NULLS LAST;
END;
$$ LANGUAGE plpgsql;

-- 5.2: Run health check
SELECT * FROM check_table_health();

-- =============================================
-- SECTION 6: CONNECTION MONITORING
-- =============================================

-- 6.1: Monitor connection pool utilization
SELECT 
    datname,
    usename,
    client_addr,
    client_port,
    application_name,
    state,
    state_change,
    now() - state_change as state_duration,
    now() - query_start as query_duration,
    query
FROM pg_stat_activity 
WHERE datname = 'sutazai'
ORDER BY state_change DESC;

-- 6.2: Connection pool statistics
SELECT 
    count(*) as total_connections,
    count(*) FILTER (WHERE state = 'active') as active_connections,
    count(*) FILTER (WHERE state = 'idle') as idle_connections,
    count(*) FILTER (WHERE state = 'idle in transaction') as idle_in_transaction,
    max(now() - query_start) as longest_query_duration,
    max(now() - state_change) as longest_idle_duration
FROM pg_stat_activity 
WHERE datname = 'sutazai';

-- =============================================
-- SECTION 7: DATABASE SIZE AND GROWTH MONITORING
-- =============================================

-- 7.1: Database size breakdown
SELECT 
    'Total Database Size' as metric,
    pg_size_pretty(pg_database_size('sutazai')) as size
UNION ALL
SELECT 
    'Tables Size' as metric,
    pg_size_pretty(sum(pg_total_relation_size(schemaname||'.'||relname))) as size
FROM pg_stat_user_tables
UNION ALL
SELECT 
    'Indexes Size' as metric,
    pg_size_pretty(sum(pg_total_relation_size(indexname::regclass))) as size
FROM pg_stat_user_indexes;

-- 7.2: Largest tables and indexes
SELECT 
    'TABLE' as type,
    schemaname||'.'||relname as object_name,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||relname)) as size,
    pg_total_relation_size(schemaname||'.'||relname) as bytes
FROM pg_stat_user_tables
UNION ALL
SELECT 
    'INDEX' as type,
    indexname as object_name,
    pg_size_pretty(pg_relation_size(indexname::regclass)) as size,
    pg_relation_size(indexname::regclass) as bytes
FROM pg_indexes 
WHERE schemaname = 'public'
ORDER BY bytes DESC
LIMIT 10;

-- =============================================
-- SECTION 8: VERIFICATION QUERIES (Run After Changes)
-- =============================================

-- 8.1: Verify configuration changes took effect
SELECT 
    name,
    setting,
    unit,
    source,
    applied
FROM pg_settings 
WHERE name IN (
    'jit',
    'shared_buffers',
    'work_mem', 
    'maintenance_work_mem',
    'autovacuum_vacuum_scale_factor',
    'autovacuum_analyze_scale_factor'
);

-- 8.2: Verify dead tuple cleanup
SELECT 
    relname,
    n_live_tup,
    n_dead_tup,
    ROUND((n_dead_tup::float / NULLIF(n_live_tup + n_dead_tup, 0)) * 100, 2) as dead_percentage
FROM pg_stat_user_tables 
WHERE n_dead_tup > 0
ORDER BY dead_percentage DESC;

-- 8.3: Check autovacuum activity (run 24 hours after changes)
SELECT 
    relname,
    last_vacuum,
    last_autovacuum,
    last_analyze,
    last_autoanalyze,
    vacuum_count,
    autovacuum_count,
    analyze_count,
    autoanalyze_count
FROM pg_stat_user_tables 
ORDER BY last_autovacuum DESC;

-- =============================================
-- EXECUTION SUMMARY
-- =============================================

/*
EXECUTION ORDER:
1. Run SECTION 1 (Immediate Maintenance) - CRITICAL
2. Run SECTION 2 (Configuration) - Requires restart for some settings
3. Run SECTION 3 (Index Analysis) - Review output
4. Run SECTION 5 (Health Check Function) - Creates monitoring
5. Run SECTION 8 (Verification) - Confirm changes

EXPECTED RESULTS:
- Dead tuple percentage < 5% on all tables
- JIT enabled for complex queries
- Reduced memory footprint
- Better autovacuum frequency
- Improved query performance

MONITORING:
- Run health check weekly: SELECT * FROM check_table_health();
- Monitor connection usage daily
- Check for slow queries if pg_stat_statements enabled
*/