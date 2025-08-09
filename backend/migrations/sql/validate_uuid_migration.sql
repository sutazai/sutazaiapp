-- =====================================================================
-- UUID MIGRATION VALIDATION SCRIPT
-- =====================================================================
-- Purpose: Comprehensive validation of the INTEGER to UUID migration
-- Use: Run this after the migration to ensure everything worked correctly
--
-- Author: Claude Code (Senior Backend Developer)  
-- Date: 2025-08-09
-- =====================================================================

-- Set up better output formatting
\x off
\pset border 2
\pset format aligned

SELECT 'UUID MIGRATION VALIDATION REPORT' as report_title, NOW() as validation_time;

-- =====================================================================
-- TEST 1: VERIFY TABLE STRUCTURE (UUID COLUMNS)
-- =====================================================================

SELECT 'TEST 1: TABLE STRUCTURE VALIDATION' as test_name;

-- Check that all primary keys are UUID type
SELECT 
    table_name,
    column_name,
    data_type,
    is_nullable,
    column_default,
    CASE 
        WHEN data_type = 'uuid' THEN '✅ CORRECT'
        ELSE '❌ WRONG TYPE!'
    END as validation_status
FROM information_schema.columns 
WHERE table_schema = 'public' 
    AND column_name = 'id' 
    AND table_name IN ('users', 'agents', 'tasks', 'chat_history', 'agent_executions', 'system_metrics')
ORDER BY table_name;

-- Check that foreign key columns are UUID type
SELECT 
    table_name,
    column_name,
    data_type,
    CASE 
        WHEN data_type = 'uuid' THEN '✅ CORRECT'
        ELSE '❌ WRONG TYPE!'
    END as validation_status
FROM information_schema.columns 
WHERE table_schema = 'public' 
    AND column_name IN ('user_id', 'agent_id', 'task_id')
    AND table_name IN ('tasks', 'chat_history', 'agent_executions')
ORDER BY table_name, column_name;

-- =====================================================================
-- TEST 2: DATA INTEGRITY VERIFICATION
-- =====================================================================

SELECT 'TEST 2: DATA INTEGRITY VERIFICATION' as test_name;

-- Count records in original backup tables vs current tables
SELECT 'DATA COUNT COMPARISON' as check_type;

WITH backup_counts AS (
    SELECT 'users' as table_name, COUNT(*) as backup_count FROM users_backup_pre_uuid
    UNION ALL
    SELECT 'agents', COUNT(*) FROM agents_backup_pre_uuid
    UNION ALL 
    SELECT 'tasks', COUNT(*) FROM tasks_backup_pre_uuid
    UNION ALL
    SELECT 'chat_history', COUNT(*) FROM chat_history_backup_pre_uuid
    UNION ALL
    SELECT 'agent_executions', COUNT(*) FROM agent_executions_backup_pre_uuid
    UNION ALL
    SELECT 'system_metrics', COUNT(*) FROM system_metrics_backup_pre_uuid
),
current_counts AS (
    SELECT 'users' as table_name, COUNT(*) as current_count FROM users
    UNION ALL
    SELECT 'agents', COUNT(*) FROM agents
    UNION ALL
    SELECT 'tasks', COUNT(*) FROM tasks  
    UNION ALL
    SELECT 'chat_history', COUNT(*) FROM chat_history
    UNION ALL
    SELECT 'agent_executions', COUNT(*) FROM agent_executions
    UNION ALL
    SELECT 'system_metrics', COUNT(*) FROM system_metrics
)
SELECT 
    b.table_name,
    b.backup_count,
    c.current_count,
    CASE 
        WHEN b.backup_count = c.current_count THEN '✅ MATCH'
        ELSE '❌ MISMATCH!'
    END as validation_status
FROM backup_counts b
JOIN current_counts c ON b.table_name = c.table_name
ORDER BY b.table_name;

-- =====================================================================
-- TEST 3: UUID FORMAT VALIDATION
-- =====================================================================

SELECT 'TEST 3: UUID FORMAT VALIDATION' as test_name;

-- Check that all UUIDs are properly formatted (36 characters with hyphens)
SELECT 'USERS UUID FORMAT CHECK' as check_type;
SELECT 
    COUNT(*) as total_users,
    COUNT(CASE WHEN LENGTH(id::text) = 36 THEN 1 END) as valid_uuid_format,
    COUNT(CASE WHEN id IS NOT NULL THEN 1 END) as non_null_ids,
    CASE 
        WHEN COUNT(*) = COUNT(CASE WHEN LENGTH(id::text) = 36 THEN 1 END) AND COUNT(*) = COUNT(CASE WHEN id IS NOT NULL THEN 1 END) THEN '✅ ALL VALID'
        ELSE '❌ INVALID UUIDs FOUND!'
    END as validation_status
FROM users;

SELECT 'AGENTS UUID FORMAT CHECK' as check_type;
SELECT 
    COUNT(*) as total_agents,
    COUNT(CASE WHEN LENGTH(id::text) = 36 THEN 1 END) as valid_uuid_format,
    COUNT(CASE WHEN id IS NOT NULL THEN 1 END) as non_null_ids,
    CASE 
        WHEN COUNT(*) = COUNT(CASE WHEN LENGTH(id::text) = 36 THEN 1 END) AND COUNT(*) = COUNT(CASE WHEN id IS NOT NULL THEN 1 END) THEN '✅ ALL VALID'
        ELSE '❌ INVALID UUIDs FOUND!'
    END as validation_status
FROM agents;

-- =====================================================================
-- TEST 4: FOREIGN KEY RELATIONSHIPS
-- =====================================================================

SELECT 'TEST 4: FOREIGN KEY RELATIONSHIP VALIDATION' as test_name;

-- Check that all foreign key constraints exist
SELECT 'FOREIGN KEY CONSTRAINTS CHECK' as check_type;
SELECT 
    tc.table_name,
    tc.constraint_name,
    kcu.column_name,
    ccu.table_name AS foreign_table_name,
    ccu.column_name AS foreign_column_name,
    '✅ EXISTS' as validation_status
FROM information_schema.table_constraints AS tc
JOIN information_schema.key_column_usage AS kcu
    ON tc.constraint_name = kcu.constraint_name
    AND tc.table_schema = kcu.table_schema
JOIN information_schema.constraint_column_usage AS ccu
    ON ccu.constraint_name = tc.constraint_name
    AND ccu.table_schema = tc.table_schema
WHERE tc.constraint_type = 'FOREIGN KEY'
    AND tc.table_schema = 'public'
    AND tc.table_name IN ('tasks', 'chat_history', 'agent_executions')
ORDER BY tc.table_name, tc.constraint_name;

-- Test foreign key data integrity - check for orphaned records
SELECT 'FOREIGN KEY DATA INTEGRITY CHECK' as check_type;

-- Check for tasks with invalid user_id references
SELECT 
    'tasks.user_id orphans' as check_name,
    COUNT(*) as orphaned_records,
    CASE WHEN COUNT(*) = 0 THEN '✅ NO ORPHANS' ELSE '❌ ORPHANS FOUND!' END as validation_status
FROM tasks t
LEFT JOIN users u ON t.user_id = u.id
WHERE t.user_id IS NOT NULL AND u.id IS NULL;

-- Check for tasks with invalid agent_id references  
SELECT 
    'tasks.agent_id orphans' as check_name,
    COUNT(*) as orphaned_records,
    CASE WHEN COUNT(*) = 0 THEN '✅ NO ORPHANS' ELSE '❌ ORPHANS FOUND!' END as validation_status
FROM tasks t
LEFT JOIN agents a ON t.agent_id = a.id
WHERE t.agent_id IS NOT NULL AND a.id IS NULL;

-- Check for chat_history with invalid user_id references
SELECT 
    'chat_history.user_id orphans' as check_name,
    COUNT(*) as orphaned_records,
    CASE WHEN COUNT(*) = 0 THEN '✅ NO ORPHANS' ELSE '❌ ORPHANS FOUND!' END as validation_status
FROM chat_history ch
LEFT JOIN users u ON ch.user_id = u.id
WHERE ch.user_id IS NOT NULL AND u.id IS NULL;

-- =====================================================================
-- TEST 5: INDEX VERIFICATION
-- =====================================================================

SELECT 'TEST 5: INDEX VERIFICATION' as test_name;

-- Check that all expected indexes exist
SELECT 
    schemaname,
    tablename,
    indexname,
    indexdef,
    '✅ EXISTS' as validation_status
FROM pg_indexes 
WHERE schemaname = 'public' 
    AND tablename IN ('users', 'agents', 'tasks', 'chat_history', 'agent_executions', 'system_metrics')
ORDER BY tablename, indexname;

-- =====================================================================
-- TEST 6: DEFAULT VALUE VERIFICATION  
-- =====================================================================

SELECT 'TEST 6: DEFAULT VALUE VERIFICATION' as test_name;

-- Check that UUID columns have gen_random_uuid() as default
SELECT 
    table_name,
    column_name,
    column_default,
    CASE 
        WHEN column_default LIKE '%gen_random_uuid%' THEN '✅ CORRECT DEFAULT'
        ELSE '❌ MISSING DEFAULT!'
    END as validation_status
FROM information_schema.columns 
WHERE table_schema = 'public' 
    AND column_name = 'id'
    AND table_name IN ('users', 'agents', 'tasks', 'chat_history', 'agent_executions', 'system_metrics')
ORDER BY table_name;

-- =====================================================================
-- TEST 7: TRIGGER VERIFICATION
-- =====================================================================

SELECT 'TEST 7: TRIGGER VERIFICATION' as test_name;

-- Check that update triggers exist
SELECT 
    trigger_name,
    event_object_table as table_name,
    action_timing,
    event_manipulation,
    '✅ EXISTS' as validation_status
FROM information_schema.triggers
WHERE trigger_schema = 'public'
    AND trigger_name LIKE '%updated_at%'
ORDER BY event_object_table;

-- =====================================================================
-- TEST 8: MAPPING TABLE VERIFICATION
-- =====================================================================

SELECT 'TEST 8: MAPPING TABLE VERIFICATION' as test_name;

-- Verify mapping tables exist and have correct data
SELECT 'MAPPING TABLES EXISTENCE' as check_type;
SELECT 
    table_name,
    '✅ EXISTS' as validation_status
FROM information_schema.tables 
WHERE table_schema = 'public' 
    AND table_name LIKE 'uuid_migration_mapping_%'
ORDER BY table_name;

-- Verify mapping table counts match original data
SELECT 'MAPPING TABLE COUNTS' as check_type;
SELECT 
    'users_mapping' as mapping_table,
    COUNT(*) as mapping_count,
    (SELECT COUNT(*) FROM users_backup_pre_uuid) as original_count,
    CASE 
        WHEN COUNT(*) = (SELECT COUNT(*) FROM users_backup_pre_uuid) THEN '✅ MATCH'
        ELSE '❌ MISMATCH!'
    END as validation_status
FROM uuid_migration_mapping_users
UNION ALL
SELECT 
    'agents_mapping',
    COUNT(*),
    (SELECT COUNT(*) FROM agents_backup_pre_uuid),
    CASE 
        WHEN COUNT(*) = (SELECT COUNT(*) FROM agents_backup_pre_uuid) THEN '✅ MATCH'
        ELSE '❌ MISMATCH!'
    END
FROM uuid_migration_mapping_agents;

-- =====================================================================
-- TEST 9: SAMPLE DATA VERIFICATION
-- =====================================================================

SELECT 'TEST 9: SAMPLE DATA VERIFICATION' as test_name;

-- Show sample users with their new UUIDs
SELECT 'SAMPLE USERS DATA' as check_type;
SELECT 
    id as new_uuid,
    username,
    email,
    created_at,
    '✅ DATA PRESENT' as validation_status
FROM users 
LIMIT 3;

-- Show sample agents with their new UUIDs
SELECT 'SAMPLE AGENTS DATA' as check_type;
SELECT 
    id as new_uuid,
    name,
    type,
    endpoint,
    '✅ DATA PRESENT' as validation_status
FROM agents 
LIMIT 3;

-- =====================================================================
-- TEST 10: EXTENSION VERIFICATION
-- =====================================================================

SELECT 'TEST 10: EXTENSION VERIFICATION' as test_name;

-- Check that required extensions are installed
SELECT 
    extname as extension_name,
    extversion as version,
    '✅ INSTALLED' as validation_status
FROM pg_extension 
WHERE extname IN ('pgcrypto', 'uuid-ossp')
ORDER BY extname;

-- =====================================================================
-- FINAL SUMMARY
-- =====================================================================

SELECT 'VALIDATION SUMMARY' as summary_title;

-- Overall health check
WITH validation_summary AS (
    SELECT 
        -- Check table structure
        CASE WHEN EXISTS (
            SELECT 1 FROM information_schema.columns 
            WHERE table_schema = 'public' AND column_name = 'id' 
            AND data_type = 'uuid' AND table_name = 'users'
        ) THEN 1 ELSE 0 END as structure_ok,
        
        -- Check data integrity
        CASE WHEN (
            SELECT COUNT(*) FROM users
        ) = (
            SELECT COUNT(*) FROM users_backup_pre_uuid
        ) THEN 1 ELSE 0 END as data_integrity_ok,
        
        -- Check foreign keys
        CASE WHEN EXISTS (
            SELECT 1 FROM information_schema.table_constraints 
            WHERE constraint_type = 'FOREIGN KEY' 
            AND table_name = 'tasks' AND constraint_name = 'tasks_user_id_fkey'
        ) THEN 1 ELSE 0 END as foreign_keys_ok,
        
        -- Check extensions
        CASE WHEN EXISTS (
            SELECT 1 FROM pg_extension WHERE extname = 'pgcrypto'
        ) THEN 1 ELSE 0 END as extensions_ok
)
SELECT 
    'TABLE STRUCTURE' as component,
    CASE WHEN structure_ok = 1 THEN '✅ PASS' ELSE '❌ FAIL' END as status
FROM validation_summary
UNION ALL
SELECT 
    'DATA INTEGRITY',
    CASE WHEN data_integrity_ok = 1 THEN '✅ PASS' ELSE '❌ FAIL' END
FROM validation_summary
UNION ALL
SELECT 
    'FOREIGN KEYS',
    CASE WHEN foreign_keys_ok = 1 THEN '✅ PASS' ELSE '❌ FAIL' END
FROM validation_summary
UNION ALL
SELECT 
    'EXTENSIONS',
    CASE WHEN extensions_ok = 1 THEN '✅ PASS' ELSE '❌ FAIL' END
FROM validation_summary;

SELECT 'UUID MIGRATION VALIDATION COMPLETED' as completion_status, NOW() as completion_time;

-- =====================================================================
-- END OF VALIDATION
-- =====================================================================