-- =====================================================================
-- EMERGENCY ROLLBACK SCRIPT: UUID IDs back to INTEGER IDs  
-- =====================================================================
-- Purpose: Rollback the UUID migration in case of critical issues
-- Risk Level: HIGH - Only use if UUID migration caused critical failures
--
-- PREREQUISITES:
-- 1. The UUID migration must have been run first (creates backup tables)
-- 2. Backup tables (*_backup_pre_uuid) must exist and be intact
-- 3. Mapping tables (uuid_migration_mapping_*) must exist
--
-- WARNING: This rollback will lose any new data created AFTER the UUID migration!
--
-- Author: Claude Code (Senior Backend Developer)
-- Date: 2025-08-09
-- =====================================================================

BEGIN; -- Start transaction - all changes will be atomic

-- =====================================================================
-- PHASE 1: SAFETY CHECKS
-- =====================================================================

DO $$
BEGIN
    -- Check if backup tables exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'users_backup_pre_uuid') THEN
        RAISE EXCEPTION 'ROLLBACK ABORTED: Backup table users_backup_pre_uuid does not exist!';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'agents_backup_pre_uuid') THEN
        RAISE EXCEPTION 'ROLLBACK ABORTED: Backup table agents_backup_pre_uuid does not exist!';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'uuid_migration_mapping_users') THEN
        RAISE EXCEPTION 'ROLLBACK ABORTED: Mapping table uuid_migration_mapping_users does not exist!';
    END IF;
    
    RAISE NOTICE 'Safety checks passed - backup tables exist';
END $$;

-- Log current data counts before rollback
SELECT 'PRE-ROLLBACK COUNTS:' as status;
SELECT 'USERS' as table_name, COUNT(*) as record_count FROM users
UNION ALL
SELECT 'AGENTS' as table_name, COUNT(*) as record_count FROM agents  
UNION ALL
SELECT 'TASKS' as table_name, COUNT(*) as record_count FROM tasks
UNION ALL
SELECT 'CHAT_HISTORY' as table_name, COUNT(*) as record_count FROM chat_history
UNION ALL
SELECT 'AGENT_EXECUTIONS' as table_name, COUNT(*) as record_count FROM agent_executions
UNION ALL
SELECT 'SYSTEM_METRICS' as table_name, COUNT(*) as record_count FROM system_metrics;

-- =====================================================================
-- PHASE 2: CREATE NEW BACKUP OF CURRENT UUID STATE (JUST IN CASE)
-- =====================================================================

-- Backup current UUID state before rolling back (in case we need to re-migrate)
CREATE TABLE users_backup_uuid_state AS SELECT * FROM users;
CREATE TABLE agents_backup_uuid_state AS SELECT * FROM agents;
CREATE TABLE tasks_backup_uuid_state AS SELECT * FROM tasks;
CREATE TABLE chat_history_backup_uuid_state AS SELECT * FROM chat_history;
CREATE TABLE agent_executions_backup_uuid_state AS SELECT * FROM agent_executions;
CREATE TABLE system_metrics_backup_uuid_state AS SELECT * FROM system_metrics;

RAISE NOTICE 'Current UUID state backed up to *_backup_uuid_state tables';

-- =====================================================================
-- PHASE 3: DROP CURRENT UUID SCHEMA
-- =====================================================================

-- Drop all foreign key constraints
ALTER TABLE tasks DROP CONSTRAINT IF EXISTS tasks_user_id_fkey;
ALTER TABLE tasks DROP CONSTRAINT IF EXISTS tasks_agent_id_fkey;
ALTER TABLE chat_history DROP CONSTRAINT IF EXISTS chat_history_user_id_fkey;
ALTER TABLE agent_executions DROP CONSTRAINT IF EXISTS agent_executions_agent_id_fkey;
ALTER TABLE agent_executions DROP CONSTRAINT IF EXISTS agent_executions_task_id_fkey;

-- Drop all indexes
DROP INDEX IF EXISTS users_email_key;
DROP INDEX IF EXISTS users_username_key;
DROP INDEX IF EXISTS idx_users_email;
DROP INDEX IF EXISTS idx_users_username;
DROP INDEX IF EXISTS idx_users_is_admin;
DROP INDEX IF EXISTS agents_name_key;
DROP INDEX IF EXISTS idx_agents_type;
DROP INDEX IF EXISTS idx_tasks_agent_id;
DROP INDEX IF EXISTS idx_tasks_user_id;
DROP INDEX IF EXISTS idx_tasks_status;
DROP INDEX IF EXISTS idx_chat_history_user_id;
DROP INDEX IF EXISTS idx_agent_executions_agent_id;
DROP INDEX IF EXISTS idx_system_metrics_name;
DROP INDEX IF EXISTS idx_system_metrics_recorded_at;

-- Drop all triggers
DROP TRIGGER IF EXISTS update_users_updated_at ON users;
DROP TRIGGER IF EXISTS update_agents_updated_at ON agents;
DROP TRIGGER IF EXISTS update_tasks_updated_at ON tasks;

-- Drop primary key constraints
ALTER TABLE users DROP CONSTRAINT IF EXISTS users_pkey;
ALTER TABLE agents DROP CONSTRAINT IF EXISTS agents_pkey;
ALTER TABLE tasks DROP CONSTRAINT IF EXISTS tasks_pkey;
ALTER TABLE chat_history DROP CONSTRAINT IF EXISTS chat_history_pkey;
ALTER TABLE agent_executions DROP CONSTRAINT IF EXISTS agent_executions_pkey;
ALTER TABLE system_metrics DROP CONSTRAINT IF EXISTS system_metrics_pkey;

-- Drop current tables (we'll restore from backup)
DROP TABLE users;
DROP TABLE agents;
DROP TABLE tasks;
DROP TABLE chat_history;
DROP TABLE agent_executions;
DROP TABLE system_metrics;

RAISE NOTICE 'Current UUID schema dropped';

-- =====================================================================
-- PHASE 4: RESTORE FROM INTEGER BACKUPS
-- =====================================================================

-- Restore original INTEGER-based tables from backups
CREATE TABLE users AS SELECT * FROM users_backup_pre_uuid;
CREATE TABLE agents AS SELECT * FROM agents_backup_pre_uuid;
CREATE TABLE tasks AS SELECT * FROM tasks_backup_pre_uuid;
CREATE TABLE chat_history AS SELECT * FROM chat_history_backup_pre_uuid;
CREATE TABLE agent_executions AS SELECT * FROM agent_executions_backup_pre_uuid;
CREATE TABLE system_metrics AS SELECT * FROM system_metrics_backup_pre_uuid;

RAISE NOTICE 'Original INTEGER tables restored from backups';

-- =====================================================================
-- PHASE 5: RECREATE INTEGER SEQUENCES
-- =====================================================================

-- Recreate sequences with correct current values
CREATE SEQUENCE users_id_seq;
SELECT setval('users_id_seq', COALESCE((SELECT MAX(id) FROM users), 0) + 1, false);

CREATE SEQUENCE agents_id_seq;
SELECT setval('agents_id_seq', COALESCE((SELECT MAX(id) FROM agents), 0) + 1, false);

CREATE SEQUENCE tasks_id_seq;
SELECT setval('tasks_id_seq', COALESCE((SELECT MAX(id) FROM tasks), 0) + 1, false);

CREATE SEQUENCE chat_history_id_seq;
SELECT setval('chat_history_id_seq', COALESCE((SELECT MAX(id) FROM chat_history), 0) + 1, false);

CREATE SEQUENCE agent_executions_id_seq;
SELECT setval('agent_executions_id_seq', COALESCE((SELECT MAX(id) FROM agent_executions), 0) + 1, false);

CREATE SEQUENCE system_metrics_id_seq;
SELECT setval('system_metrics_id_seq', COALESCE((SELECT MAX(id) FROM system_metrics), 0) + 1, false);

-- Set sequence defaults
ALTER TABLE users ALTER COLUMN id SET DEFAULT nextval('users_id_seq'::regclass);
ALTER TABLE agents ALTER COLUMN id SET DEFAULT nextval('agents_id_seq'::regclass);
ALTER TABLE tasks ALTER COLUMN id SET DEFAULT nextval('tasks_id_seq'::regclass);
ALTER TABLE chat_history ALTER COLUMN id SET DEFAULT nextval('chat_history_id_seq'::regclass);
ALTER TABLE agent_executions ALTER COLUMN id SET DEFAULT nextval('agent_executions_id_seq'::regclass);
ALTER TABLE system_metrics ALTER COLUMN id SET DEFAULT nextval('system_metrics_id_seq'::regclass);

RAISE NOTICE 'INTEGER sequences recreated with correct values';

-- =====================================================================
-- PHASE 6: RECREATE PRIMARY KEYS
-- =====================================================================

ALTER TABLE users ADD CONSTRAINT users_pkey PRIMARY KEY (id);
ALTER TABLE agents ADD CONSTRAINT agents_pkey PRIMARY KEY (id);
ALTER TABLE tasks ADD CONSTRAINT tasks_pkey PRIMARY KEY (id);
ALTER TABLE chat_history ADD CONSTRAINT chat_history_pkey PRIMARY KEY (id);
ALTER TABLE agent_executions ADD CONSTRAINT agent_executions_pkey PRIMARY KEY (id);
ALTER TABLE system_metrics ADD CONSTRAINT system_metrics_pkey PRIMARY KEY (id);

RAISE NOTICE 'Primary key constraints recreated';

-- =====================================================================
-- PHASE 7: RECREATE FOREIGN KEYS
-- =====================================================================

ALTER TABLE tasks 
ADD CONSTRAINT tasks_user_id_fkey 
FOREIGN KEY (user_id) REFERENCES users(id);

ALTER TABLE tasks
ADD CONSTRAINT tasks_agent_id_fkey
FOREIGN KEY (agent_id) REFERENCES agents(id);

ALTER TABLE chat_history
ADD CONSTRAINT chat_history_user_id_fkey
FOREIGN KEY (user_id) REFERENCES users(id);

ALTER TABLE agent_executions
ADD CONSTRAINT agent_executions_agent_id_fkey
FOREIGN KEY (agent_id) REFERENCES agents(id);

ALTER TABLE agent_executions
ADD CONSTRAINT agent_executions_task_id_fkey
FOREIGN KEY (task_id) REFERENCES tasks(id);

RAISE NOTICE 'Foreign key constraints recreated';

-- =====================================================================
-- PHASE 8: RECREATE INDEXES
-- =====================================================================

CREATE UNIQUE INDEX users_email_key ON users(email);
CREATE UNIQUE INDEX users_username_key ON users(username);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_is_admin ON users(is_admin) WHERE is_admin = true;

CREATE UNIQUE INDEX agents_name_key ON agents(name);
CREATE INDEX idx_agents_type ON agents(type);

CREATE INDEX idx_tasks_agent_id ON tasks(agent_id);
CREATE INDEX idx_tasks_user_id ON tasks(user_id);
CREATE INDEX idx_tasks_status ON tasks(status);

CREATE INDEX idx_chat_history_user_id ON chat_history(user_id);

CREATE INDEX idx_agent_executions_agent_id ON agent_executions(agent_id);

CREATE INDEX idx_system_metrics_name ON system_metrics(metric_name);
CREATE INDEX idx_system_metrics_recorded_at ON system_metrics(recorded_at);

RAISE NOTICE 'All indexes recreated';

-- =====================================================================
-- PHASE 9: RECREATE TRIGGERS
-- =====================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Recreate update triggers
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_agents_updated_at
    BEFORE UPDATE ON agents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_tasks_updated_at
    BEFORE UPDATE ON tasks
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

RAISE NOTICE 'Triggers recreated';

-- =====================================================================
-- PHASE 10: FINAL VALIDATION
-- =====================================================================

-- Verify rollback success
DO $$
DECLARE
    users_count INTEGER;
    agents_count INTEGER;
    users_backup_count INTEGER;
    agents_backup_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO users_count FROM users;
    SELECT COUNT(*) INTO users_backup_count FROM users_backup_pre_uuid;
    
    SELECT COUNT(*) INTO agents_count FROM agents;
    SELECT COUNT(*) INTO agents_backup_count FROM agents_backup_pre_uuid;
    
    IF users_count != users_backup_count THEN
        RAISE EXCEPTION 'ROLLBACK VALIDATION FAILED: Users count mismatch. Current: %, Expected: %', users_count, users_backup_count;
    END IF;
    
    IF agents_count != agents_backup_count THEN
        RAISE EXCEPTION 'ROLLBACK VALIDATION FAILED: Agents count mismatch. Current: %, Expected: %', agents_count, agents_backup_count;
    END IF;
    
    -- Verify data types are back to INTEGER
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'users' 
        AND column_name = 'id' 
        AND data_type = 'integer'
    ) THEN
        RAISE EXCEPTION 'ROLLBACK VALIDATION FAILED: users.id is not INTEGER type';
    END IF;
    
    RAISE NOTICE 'ROLLBACK VALIDATION SUCCESSFUL: All checks passed';
END $$;

-- Log final counts
SELECT 'ROLLBACK COMPLETED - Final data counts:' as status;
SELECT 'USERS' as table_name, COUNT(*) as record_count FROM users
UNION ALL
SELECT 'AGENTS' as table_name, COUNT(*) as record_count FROM agents  
UNION ALL
SELECT 'TASKS' as table_name, COUNT(*) as record_count FROM tasks
UNION ALL
SELECT 'CHAT_HISTORY' as table_name, COUNT(*) as record_count FROM chat_history
UNION ALL
SELECT 'AGENT_EXECUTIONS' as table_name, COUNT(*) as record_count FROM agent_executions
UNION ALL
SELECT 'SYSTEM_METRICS' as table_name, COUNT(*) as record_count FROM system_metrics;

RAISE NOTICE 'UUID to INTEGER rollback completed successfully at %', NOW();
RAISE NOTICE 'Schema restored to original INTEGER-based structure';
RAISE NOTICE 'UUID state backed up to *_backup_uuid_state tables for potential re-migration';

-- COMMIT the rollback (comment out for testing)
COMMIT;

-- =====================================================================
-- POST-ROLLBACK NOTES:
-- 
-- 1. All tables restored to original INTEGER primary keys
-- 2. All sequences recreated with correct next values  
-- 3. All foreign keys, indexes, and triggers recreated
-- 4. UUID state backed up to *_backup_uuid_state tables
-- 5. Original backup tables (*_backup_pre_uuid) preserved
--
-- WARNING: Any data created AFTER the UUID migration has been LOST!
-- 
-- To clean up backup tables:
-- DROP TABLE *_backup_pre_uuid, *_backup_uuid_state, uuid_migration_mapping_*
-- =====================================================================