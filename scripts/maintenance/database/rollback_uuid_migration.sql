-- ULTRAFIX: Rollback UUID Migration to SERIAL Primary Keys
-- Emergency rollback script if UUID migration needs to be reversed
-- EXECUTION TIME: ~2-3 minutes
-- USE CASE: If issues are discovered after UUID migration

BEGIN;

-- Safety check - confirm UUID migration exists
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'migration_log') THEN
        RAISE EXCEPTION 'Migration log table not found. Cannot proceed with rollback.';
    END IF;
    
    IF NOT EXISTS (
        SELECT 1 FROM migration_log 
        WHERE operation IN ('UUID_MIGRATION', 'PK_SWITCHOVER') 
        AND status = 'COMPLETED'
    ) THEN
        RAISE EXCEPTION 'No completed UUID migration found. Nothing to rollback.';
    END IF;
END $$;

-- Log rollback start
INSERT INTO migration_log (table_name, operation, status) 
VALUES ('ALL_TABLES', 'ROLLBACK_UUID', 'STARTED');

RAISE NOTICE '=== ULTRAFIX UUID MIGRATION ROLLBACK STARTING ===';
RAISE NOTICE 'This will restore SERIAL primary keys and remove UUID columns';

-- =========================================
-- PHASE 1: DROP UUID-BASED FOREIGN KEY CONSTRAINTS
-- =========================================

DO $$
BEGIN
    RAISE NOTICE 'Phase 1: Dropping UUID-based foreign key constraints...';
    
    -- Drop UUID foreign key constraints
    ALTER TABLE tasks DROP CONSTRAINT IF EXISTS fk_tasks_user_id;
    ALTER TABLE tasks DROP CONSTRAINT IF EXISTS fk_tasks_agent_id;
    ALTER TABLE chat_history DROP CONSTRAINT IF EXISTS fk_chat_history_user_id;
    ALTER TABLE agent_executions DROP CONSTRAINT IF EXISTS fk_agent_executions_agent_id;
    ALTER TABLE agent_executions DROP CONSTRAINT IF EXISTS fk_agent_executions_task_id;
    ALTER TABLE sessions DROP CONSTRAINT IF EXISTS fk_sessions_user_id;
    ALTER TABLE agent_health DROP CONSTRAINT IF EXISTS fk_agent_health_agent_id;
    
    RAISE NOTICE 'Phase 1 completed: UUID foreign key constraints dropped';
END $$;

-- =========================================
-- PHASE 2: RESTORE SERIAL PRIMARY KEYS
-- =========================================

-- Check if we're rolling back from full switchover or just the UUID addition
DO $$
DECLARE
    has_old_id_columns BOOLEAN;
BEGIN
    -- Check if old_id columns exist (indicates full switchover was completed)
    SELECT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'users' AND column_name = 'old_id'
    ) INTO has_old_id_columns;
    
    IF has_old_id_columns THEN
        RAISE NOTICE 'Detected full UUID switchover - rolling back column names and constraints...';
        
        -- Users table rollback
        ALTER TABLE users DROP CONSTRAINT users_pkey;
        ALTER TABLE users RENAME COLUMN id TO uuid_id;
        ALTER TABLE users RENAME COLUMN old_id TO id;
        ALTER TABLE users ADD CONSTRAINT users_pkey PRIMARY KEY (id);
        
        -- Agents table rollback  
        ALTER TABLE agents DROP CONSTRAINT agents_pkey;
        ALTER TABLE agents RENAME COLUMN id TO uuid_id;
        ALTER TABLE agents RENAME COLUMN old_id TO id;
        ALTER TABLE agents ADD CONSTRAINT agents_pkey PRIMARY KEY (id);
        
        -- Tasks table rollback
        ALTER TABLE tasks DROP CONSTRAINT tasks_pkey;
        ALTER TABLE tasks RENAME COLUMN id TO uuid_id;
        ALTER TABLE tasks RENAME COLUMN old_id TO id;
        ALTER TABLE tasks RENAME COLUMN agent_id TO uuid_agent_id;
        ALTER TABLE tasks RENAME COLUMN old_agent_id TO agent_id;
        ALTER TABLE tasks RENAME COLUMN user_id TO uuid_user_id;
        ALTER TABLE tasks RENAME COLUMN old_user_id TO user_id;
        ALTER TABLE tasks ADD CONSTRAINT tasks_pkey PRIMARY KEY (id);
        
        -- Chat history rollback
        ALTER TABLE chat_history DROP CONSTRAINT chat_history_pkey;
        ALTER TABLE chat_history RENAME COLUMN id TO uuid_id;
        ALTER TABLE chat_history RENAME COLUMN old_id TO id;
        ALTER TABLE chat_history RENAME COLUMN user_id TO uuid_user_id;
        ALTER TABLE chat_history RENAME COLUMN old_user_id TO user_id;
        ALTER TABLE chat_history ADD CONSTRAINT chat_history_pkey PRIMARY KEY (id);
        
        -- Agent executions rollback
        ALTER TABLE agent_executions DROP CONSTRAINT agent_executions_pkey;
        ALTER TABLE agent_executions RENAME COLUMN id TO uuid_id;
        ALTER TABLE agent_executions RENAME COLUMN old_id TO id;
        ALTER TABLE agent_executions RENAME COLUMN agent_id TO uuid_agent_id;
        ALTER TABLE agent_executions RENAME COLUMN old_agent_id TO agent_id;
        ALTER TABLE agent_executions RENAME COLUMN task_id TO uuid_task_id;
        ALTER TABLE agent_executions RENAME COLUMN old_task_id TO task_id;
        ALTER TABLE agent_executions ADD CONSTRAINT agent_executions_pkey PRIMARY KEY (id);
        
        -- System metrics rollback
        ALTER TABLE system_metrics DROP CONSTRAINT system_metrics_pkey;
        ALTER TABLE system_metrics RENAME COLUMN id TO uuid_id;
        ALTER TABLE system_metrics RENAME COLUMN old_id TO id;
        ALTER TABLE system_metrics ADD CONSTRAINT system_metrics_pkey PRIMARY KEY (id);
        
        RAISE NOTICE 'SERIAL primary keys restored from old_id columns';
        
    ELSE
        RAISE NOTICE 'Detected UUID column addition only - removing UUID columns...';
        
        -- Just remove the UUID columns that were added
        ALTER TABLE users DROP COLUMN IF EXISTS new_id;
        ALTER TABLE agents DROP COLUMN IF EXISTS new_id;
        ALTER TABLE tasks DROP COLUMN IF EXISTS new_id, DROP COLUMN IF EXISTS new_agent_id, DROP COLUMN IF EXISTS new_user_id;
        ALTER TABLE chat_history DROP COLUMN IF EXISTS new_id, DROP COLUMN IF EXISTS new_user_id;
        ALTER TABLE agent_executions DROP COLUMN IF EXISTS new_id, DROP COLUMN IF EXISTS new_agent_id, DROP COLUMN IF EXISTS new_task_id;
        ALTER TABLE system_metrics DROP COLUMN IF EXISTS new_id;
        
        RAISE NOTICE 'UUID columns removed, SERIAL primary keys preserved';
    END IF;
END $$;

-- =========================================
-- PHASE 3: RESTORE ORIGINAL FOREIGN KEY CONSTRAINTS
-- =========================================

DO $$
BEGIN
    RAISE NOTICE 'Phase 3: Restoring original SERIAL-based foreign key constraints...';
    
    -- Recreate original foreign key constraints with SERIAL columns
    ALTER TABLE tasks ADD CONSTRAINT fk_tasks_user_id_serial 
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE;
    
    ALTER TABLE tasks ADD CONSTRAINT fk_tasks_agent_id_serial 
        FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE SET NULL;
        
    ALTER TABLE chat_history ADD CONSTRAINT fk_chat_history_user_id_serial 
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE;
        
    ALTER TABLE agent_executions ADD CONSTRAINT fk_agent_executions_agent_id_serial 
        FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE CASCADE;
        
    ALTER TABLE agent_executions ADD CONSTRAINT fk_agent_executions_task_id_serial 
        FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE;
    
    -- Extended schema constraints if tables exist
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'sessions') THEN
        IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'sessions' AND column_name = 'old_user_id') THEN
            ALTER TABLE sessions RENAME COLUMN user_id TO uuid_user_id;
            ALTER TABLE sessions RENAME COLUMN old_user_id TO user_id;
        END IF;
        ALTER TABLE sessions ADD CONSTRAINT fk_sessions_user_id_serial 
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE;
    END IF;
    
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'agent_health') THEN
        IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'agent_health' AND column_name = 'old_agent_id') THEN
            ALTER TABLE agent_health RENAME COLUMN agent_id TO uuid_agent_id;
            ALTER TABLE agent_health RENAME COLUMN old_agent_id TO agent_id;
        END IF;
        ALTER TABLE agent_health ADD CONSTRAINT fk_agent_health_agent_id_serial 
            FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE CASCADE;
    END IF;
    
    RAISE NOTICE 'Phase 3 completed: Original foreign key constraints restored';
END $$;

-- =========================================
-- PHASE 4: CLEANUP UUID INDEXES
-- =========================================

DO $$
BEGIN
    RAISE NOTICE 'Phase 4: Cleaning up UUID indexes...';
    
    -- Drop UUID-specific indexes
    DROP INDEX IF EXISTS users_new_id_unique;
    DROP INDEX IF EXISTS agents_new_id_unique;
    DROP INDEX IF EXISTS tasks_new_id_unique;
    DROP INDEX IF EXISTS tasks_new_user_id_idx;
    DROP INDEX IF EXISTS tasks_new_agent_id_idx;
    DROP INDEX IF EXISTS chat_history_new_id_unique;
    DROP INDEX IF EXISTS agent_executions_new_id_unique;
    DROP INDEX IF EXISTS system_metrics_new_id_unique;
    DROP INDEX IF EXISTS sessions_new_id_unique;
    DROP INDEX IF EXISTS agent_health_new_id_unique;
    
    -- Drop UUID-optimized indexes
    DROP INDEX IF EXISTS idx_tasks_status_optimized;
    DROP INDEX IF EXISTS idx_tasks_user_id_uuid;
    DROP INDEX IF EXISTS idx_tasks_agent_id_uuid;
    DROP INDEX IF EXISTS idx_chat_history_user_id_uuid;
    DROP INDEX IF EXISTS idx_agent_executions_agent_id_uuid;
    DROP INDEX IF EXISTS idx_system_metrics_name_time;
    DROP INDEX IF EXISTS idx_tasks_status_created_at;
    DROP INDEX IF EXISTS idx_tasks_user_priority;
    DROP INDEX IF EXISTS idx_agent_executions_status_time;
    
    -- Recreate original indexes
    CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
    CREATE INDEX IF NOT EXISTS idx_tasks_user_id ON tasks(user_id);
    CREATE INDEX IF NOT EXISTS idx_chat_history_user_id ON chat_history(user_id);
    CREATE INDEX IF NOT EXISTS idx_agent_executions_agent_id ON agent_executions(agent_id);
    CREATE INDEX IF NOT EXISTS idx_system_metrics_name ON system_metrics(metric_name);
    CREATE INDEX IF NOT EXISTS idx_system_metrics_recorded_at ON system_metrics(recorded_at);
    
    RAISE NOTICE 'Phase 4 completed: UUID indexes cleaned up, original indexes restored';
END $$;

-- =========================================
-- PHASE 5: FINAL CLEANUP OF UUID COLUMNS
-- =========================================

DO $$
BEGIN
    RAISE NOTICE 'Phase 5: Final cleanup of remaining UUID columns...';
    
    -- Remove any remaining UUID columns (after switchover rollback)
    ALTER TABLE users DROP COLUMN IF EXISTS uuid_id;
    ALTER TABLE agents DROP COLUMN IF EXISTS uuid_id;
    ALTER TABLE tasks DROP COLUMN IF EXISTS uuid_id, DROP COLUMN IF EXISTS uuid_agent_id, DROP COLUMN IF EXISTS uuid_user_id;
    ALTER TABLE chat_history DROP COLUMN IF EXISTS uuid_id, DROP COLUMN IF EXISTS uuid_user_id;
    ALTER TABLE agent_executions DROP COLUMN IF EXISTS uuid_id, DROP COLUMN IF EXISTS uuid_agent_id, DROP COLUMN IF EXISTS uuid_task_id;
    ALTER TABLE system_metrics DROP COLUMN IF EXISTS uuid_id;
    
    -- Extended schema cleanup
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'sessions') THEN
        ALTER TABLE sessions DROP COLUMN IF EXISTS uuid_id, DROP COLUMN IF EXISTS uuid_user_id;
    END IF;
    
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'agent_health') THEN
        ALTER TABLE agent_health DROP COLUMN IF EXISTS uuid_id, DROP COLUMN IF EXISTS uuid_agent_id;
    END IF;
    
    RAISE NOTICE 'Phase 5 completed: All UUID columns removed';
END $$;

-- =========================================
-- FINAL VERIFICATION AND LOGGING
-- =========================================

DO $$
DECLARE
    rollback_time INTEGER;
    table_count INTEGER;
BEGIN
    -- Log rollback completion
    UPDATE migration_log 
    SET status = 'COMPLETED', completed_at = CURRENT_TIMESTAMP 
    WHERE table_name = 'ALL_TABLES' AND operation = 'ROLLBACK_UUID';
    
    -- Calculate rollback time
    SELECT EXTRACT(EPOCH FROM (
        completed_at - started_at
    )) * 1000 INTO rollback_time
    FROM migration_log 
    WHERE table_name = 'ALL_TABLES' AND operation = 'ROLLBACK_UUID';
    
    -- Count tables with SERIAL primary keys
    SELECT COUNT(*) INTO table_count
    FROM information_schema.columns 
    WHERE table_schema = 'public' 
    AND column_name = 'id' 
    AND data_type = 'integer';
    
    RAISE NOTICE '=== ULTRAFIX UUID MIGRATION ROLLBACK COMPLETED ===';
    RAISE NOTICE 'Rollback execution time: % ms (% seconds)', rollback_time, rollback_time/1000;
    RAISE NOTICE 'Tables with SERIAL primary keys restored: %', table_count;
    RAISE NOTICE 'All UUID columns and constraints removed';
    RAISE NOTICE 'Original SERIAL-based schema restored';
    RAISE NOTICE '';
    RAISE NOTICE '=== VERIFICATION ===';
    RAISE NOTICE 'Check table structures: \d+ users';
    RAISE NOTICE 'Verify rollback: SELECT * FROM migration_log WHERE operation = '\''ROLLBACK_UUID'\'';';
    RAISE NOTICE '';
    RAISE NOTICE 'STATUS: UUID MIGRATION SUCCESSFULLY ROLLED BACK TO SERIAL PKs';
END $$;

COMMIT;

-- Final verification
\echo '=== ROLLBACK VERIFICATION ==='
\echo 'Run these commands to verify the rollback:'
\echo 'SELECT table_name, column_name, data_type FROM information_schema.columns WHERE column_name = '\''id'\'' AND table_schema = '\''public'\'';'
\echo 'SELECT COUNT(*) as uuid_columns FROM information_schema.columns WHERE column_name LIKE '\''%uuid%'\'' AND table_schema = '\''public'\'';'
\echo 'SELECT * FROM migration_log WHERE operation = '\''ROLLBACK_UUID'\'';'