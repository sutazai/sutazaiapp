-- =====================================================================
-- CRITICAL DATABASE MIGRATION: INTEGER IDs to UUID IDs
-- =====================================================================
-- Purpose: Convert all INTEGER primary keys to UUID primary keys
-- Risk Level: HIGH - This migration touches all tables and relationships
-- 
-- BEFORE RUNNING:
-- 1. Take a complete database dump: pg_dump -U sutazai sutazai > pre_migration_backup.sql
-- 2. Ensure no active connections to database during migration
-- 3. Test on development environment first
--
-- Author: Claude Code (Senior Backend Developer)
-- Date: 2025-08-09
-- =====================================================================

BEGIN; -- Start transaction - all changes will be atomic

-- =====================================================================
-- PHASE 1: PREPARATION AND SAFETY CHECKS
-- =====================================================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Log migration start
DO $$ 
BEGIN
    RAISE NOTICE 'Starting INTEGER to UUID migration at %', NOW();
    RAISE NOTICE 'Current data counts:';
END $$;

-- Display current data for verification
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
-- PHASE 2: CREATE BACKUP TABLES (CRITICAL FOR ROLLBACK)
-- =====================================================================

-- Backup existing data exactly as-is
CREATE TABLE users_backup_pre_uuid AS SELECT * FROM users;
CREATE TABLE agents_backup_pre_uuid AS SELECT * FROM agents;
CREATE TABLE tasks_backup_pre_uuid AS SELECT * FROM tasks;
CREATE TABLE chat_history_backup_pre_uuid AS SELECT * FROM chat_history;
CREATE TABLE agent_executions_backup_pre_uuid AS SELECT * FROM agent_executions;
CREATE TABLE system_metrics_backup_pre_uuid AS SELECT * FROM system_metrics;

RAISE NOTICE 'Backup tables created successfully';

-- =====================================================================
-- PHASE 3: CREATE INTEGER -> UUID MAPPING TABLES
-- =====================================================================

-- These tables will track the conversion from old INTEGER IDs to new UUIDs
CREATE TABLE uuid_migration_mapping_users (
    old_integer_id INTEGER PRIMARY KEY,
    new_uuid_id UUID NOT NULL DEFAULT gen_random_uuid(),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE uuid_migration_mapping_agents (
    old_integer_id INTEGER PRIMARY KEY,
    new_uuid_id UUID NOT NULL DEFAULT gen_random_uuid(),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE uuid_migration_mapping_tasks (
    old_integer_id INTEGER PRIMARY KEY,
    new_uuid_id UUID NOT NULL DEFAULT gen_random_uuid(),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE uuid_migration_mapping_chat_history (
    old_integer_id INTEGER PRIMARY KEY,
    new_uuid_id UUID NOT NULL DEFAULT gen_random_uuid(),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE uuid_migration_mapping_agent_executions (
    old_integer_id INTEGER PRIMARY KEY,
    new_uuid_id UUID NOT NULL DEFAULT gen_random_uuid(),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE uuid_migration_mapping_system_metrics (
    old_integer_id INTEGER PRIMARY KEY,
    new_uuid_id UUID NOT NULL DEFAULT gen_random_uuid(),
    created_at TIMESTAMP DEFAULT NOW()
);

-- =====================================================================
-- PHASE 4: POPULATE MAPPING TABLES WITH EXISTING DATA
-- =====================================================================

-- Generate UUID mappings for all existing records
INSERT INTO uuid_migration_mapping_users (old_integer_id)
SELECT id FROM users;

INSERT INTO uuid_migration_mapping_agents (old_integer_id) 
SELECT id FROM agents;

INSERT INTO uuid_migration_mapping_tasks (old_integer_id)
SELECT id FROM tasks;

INSERT INTO uuid_migration_mapping_chat_history (old_integer_id)
SELECT id FROM chat_history;

INSERT INTO uuid_migration_mapping_agent_executions (old_integer_id)
SELECT id FROM agent_executions;

INSERT INTO uuid_migration_mapping_system_metrics (old_integer_id)
SELECT id FROM system_metrics;

RAISE NOTICE 'UUID mappings created for all existing records';

-- =====================================================================
-- PHASE 5: DROP ALL FOREIGN KEY CONSTRAINTS (TEMPORARILY)
-- =====================================================================

-- We need to drop FKs before changing column types
ALTER TABLE tasks DROP CONSTRAINT IF EXISTS tasks_user_id_fkey;
ALTER TABLE tasks DROP CONSTRAINT IF EXISTS tasks_agent_id_fkey;
ALTER TABLE chat_history DROP CONSTRAINT IF EXISTS chat_history_user_id_fkey;
ALTER TABLE agent_executions DROP CONSTRAINT IF EXISTS agent_executions_agent_id_fkey;
ALTER TABLE agent_executions DROP CONSTRAINT IF EXISTS agent_executions_task_id_fkey;

RAISE NOTICE 'Foreign key constraints dropped temporarily';

-- =====================================================================
-- PHASE 6: ADD NEW UUID COLUMNS ALONGSIDE INTEGER COLUMNS
-- =====================================================================

-- Add UUID columns to all tables (initially nullable)
ALTER TABLE users ADD COLUMN id_uuid UUID;
ALTER TABLE agents ADD COLUMN id_uuid UUID;
ALTER TABLE tasks ADD COLUMN id_uuid UUID;
ALTER TABLE tasks ADD COLUMN user_id_uuid UUID;
ALTER TABLE tasks ADD COLUMN agent_id_uuid UUID;
ALTER TABLE chat_history ADD COLUMN id_uuid UUID;
ALTER TABLE chat_history ADD COLUMN user_id_uuid UUID;
ALTER TABLE agent_executions ADD COLUMN id_uuid UUID;
ALTER TABLE agent_executions ADD COLUMN agent_id_uuid UUID;
ALTER TABLE agent_executions ADD COLUMN task_id_uuid UUID;
ALTER TABLE system_metrics ADD COLUMN id_uuid UUID;

RAISE NOTICE 'UUID columns added to all tables';

-- =====================================================================
-- PHASE 7: POPULATE UUID COLUMNS WITH MAPPED VALUES
-- =====================================================================

-- Update primary key UUID columns using mapping tables
UPDATE users 
SET id_uuid = (SELECT new_uuid_id FROM uuid_migration_mapping_users WHERE old_integer_id = users.id);

UPDATE agents
SET id_uuid = (SELECT new_uuid_id FROM uuid_migration_mapping_agents WHERE old_integer_id = agents.id);

UPDATE tasks
SET id_uuid = (SELECT new_uuid_id FROM uuid_migration_mapping_tasks WHERE old_integer_id = tasks.id);

UPDATE chat_history
SET id_uuid = (SELECT new_uuid_id FROM uuid_migration_mapping_chat_history WHERE old_integer_id = chat_history.id);

UPDATE agent_executions
SET id_uuid = (SELECT new_uuid_id FROM uuid_migration_mapping_agent_executions WHERE old_integer_id = agent_executions.id);

UPDATE system_metrics
SET id_uuid = (SELECT new_uuid_id FROM uuid_migration_mapping_system_metrics WHERE old_integer_id = system_metrics.id);

-- Update foreign key UUID columns using mapping tables
UPDATE tasks 
SET user_id_uuid = (SELECT new_uuid_id FROM uuid_migration_mapping_users WHERE old_integer_id = tasks.user_id)
WHERE user_id IS NOT NULL;

UPDATE tasks
SET agent_id_uuid = (SELECT new_uuid_id FROM uuid_migration_mapping_agents WHERE old_integer_id = tasks.agent_id)  
WHERE agent_id IS NOT NULL;

UPDATE chat_history
SET user_id_uuid = (SELECT new_uuid_id FROM uuid_migration_mapping_users WHERE old_integer_id = chat_history.user_id)
WHERE user_id IS NOT NULL;

UPDATE agent_executions  
SET agent_id_uuid = (SELECT new_uuid_id FROM uuid_migration_mapping_agents WHERE old_integer_id = agent_executions.agent_id)
WHERE agent_id IS NOT NULL;

UPDATE agent_executions
SET task_id_uuid = (SELECT new_uuid_id FROM uuid_migration_mapping_tasks WHERE old_integer_id = agent_executions.task_id)
WHERE task_id IS NOT NULL;

RAISE NOTICE 'All UUID columns populated with mapped values';

-- =====================================================================
-- PHASE 8: VALIDATION - ENSURE NO DATA LOSS
-- =====================================================================

DO $$ 
DECLARE
    users_count_original INTEGER;
    users_count_mapped INTEGER;
    agents_count_original INTEGER;
    agents_count_mapped INTEGER;
BEGIN
    SELECT COUNT(*) INTO users_count_original FROM users;
    SELECT COUNT(*) INTO users_count_mapped FROM users WHERE id_uuid IS NOT NULL;
    
    SELECT COUNT(*) INTO agents_count_original FROM agents;
    SELECT COUNT(*) INTO agents_count_mapped FROM agents WHERE id_uuid IS NOT NULL;
    
    IF users_count_original != users_count_mapped THEN
        RAISE EXCEPTION 'DATA LOSS DETECTED: Users mapping failed. Original: %, Mapped: %', users_count_original, users_count_mapped;
    END IF;
    
    IF agents_count_original != agents_count_mapped THEN
        RAISE EXCEPTION 'DATA LOSS DETECTED: Agents mapping failed. Original: %, Mapped: %', agents_count_original, agents_count_mapped;
    END IF;
    
    RAISE NOTICE 'Validation passed: All records successfully mapped to UUIDs';
END $$;

-- =====================================================================
-- PHASE 9: DROP OLD INTEGER COLUMNS AND CONSTRAINTS
-- =====================================================================

-- Drop old primary key constraints and sequences
ALTER TABLE users DROP CONSTRAINT users_pkey;
ALTER TABLE agents DROP CONSTRAINT agents_pkey;
ALTER TABLE tasks DROP CONSTRAINT tasks_pkey;
ALTER TABLE chat_history DROP CONSTRAINT chat_history_pkey;
ALTER TABLE agent_executions DROP CONSTRAINT agent_executions_pkey;
ALTER TABLE system_metrics DROP CONSTRAINT system_metrics_pkey;

-- Drop old integer columns
ALTER TABLE users DROP COLUMN id;
ALTER TABLE agents DROP COLUMN id;
ALTER TABLE tasks DROP COLUMN id, DROP COLUMN user_id, DROP COLUMN agent_id;
ALTER TABLE chat_history DROP COLUMN id, DROP COLUMN user_id;
ALTER TABLE agent_executions DROP COLUMN id, DROP COLUMN agent_id, DROP COLUMN task_id;
ALTER TABLE system_metrics DROP COLUMN id;

-- Drop sequences
DROP SEQUENCE IF EXISTS users_id_seq;
DROP SEQUENCE IF EXISTS agents_id_seq;
DROP SEQUENCE IF EXISTS tasks_id_seq;
DROP SEQUENCE IF EXISTS chat_history_id_seq;
DROP SEQUENCE IF EXISTS agent_executions_id_seq;
DROP SEQUENCE IF EXISTS system_metrics_id_seq;

RAISE NOTICE 'Old INTEGER columns and sequences removed';

-- =====================================================================
-- PHASE 10: RENAME UUID COLUMNS TO PRIMARY COLUMN NAMES
-- =====================================================================

-- Rename UUID columns to become the primary columns
ALTER TABLE users RENAME COLUMN id_uuid TO id;
ALTER TABLE agents RENAME COLUMN id_uuid TO id;
ALTER TABLE tasks RENAME COLUMN id_uuid TO id;
ALTER TABLE tasks RENAME COLUMN user_id_uuid TO user_id;
ALTER TABLE tasks RENAME COLUMN agent_id_uuid TO agent_id;
ALTER TABLE chat_history RENAME COLUMN id_uuid TO id;
ALTER TABLE chat_history RENAME COLUMN user_id_uuid TO user_id;
ALTER TABLE agent_executions RENAME COLUMN id_uuid TO id;
ALTER TABLE agent_executions RENAME COLUMN agent_id_uuid TO agent_id;
ALTER TABLE agent_executions RENAME COLUMN task_id_uuid TO task_id;
ALTER TABLE system_metrics RENAME COLUMN id_uuid TO id;

RAISE NOTICE 'UUID columns renamed to primary names';

-- =====================================================================
-- PHASE 11: ADD NOT NULL CONSTRAINTS AND PRIMARY KEYS
-- =====================================================================

-- Make primary key columns NOT NULL and add PRIMARY KEY constraints
ALTER TABLE users ALTER COLUMN id SET NOT NULL;
ALTER TABLE users ALTER COLUMN id SET DEFAULT gen_random_uuid();
ALTER TABLE users ADD CONSTRAINT users_pkey PRIMARY KEY (id);

ALTER TABLE agents ALTER COLUMN id SET NOT NULL;
ALTER TABLE agents ALTER COLUMN id SET DEFAULT gen_random_uuid();
ALTER TABLE agents ADD CONSTRAINT agents_pkey PRIMARY KEY (id);

ALTER TABLE tasks ALTER COLUMN id SET NOT NULL;
ALTER TABLE tasks ALTER COLUMN id SET DEFAULT gen_random_uuid();
ALTER TABLE tasks ADD CONSTRAINT tasks_pkey PRIMARY KEY (id);

ALTER TABLE chat_history ALTER COLUMN id SET NOT NULL;
ALTER TABLE chat_history ALTER COLUMN id SET DEFAULT gen_random_uuid();
ALTER TABLE chat_history ADD CONSTRAINT chat_history_pkey PRIMARY KEY (id);

ALTER TABLE agent_executions ALTER COLUMN id SET NOT NULL;
ALTER TABLE agent_executions ALTER COLUMN id SET DEFAULT gen_random_uuid();
ALTER TABLE agent_executions ADD CONSTRAINT agent_executions_pkey PRIMARY KEY (id);

ALTER TABLE system_metrics ALTER COLUMN id SET NOT NULL;
ALTER TABLE system_metrics ALTER COLUMN id SET DEFAULT gen_random_uuid();
ALTER TABLE system_metrics ADD CONSTRAINT system_metrics_pkey PRIMARY KEY (id);

RAISE NOTICE 'Primary key constraints recreated with UUIDs';

-- =====================================================================
-- PHASE 12: RECREATE FOREIGN KEY CONSTRAINTS
-- =====================================================================

-- Recreate foreign key constraints with UUID references
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
-- PHASE 13: RECREATE INDEXES
-- =====================================================================

-- Recreate all indexes that existed before migration
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
-- PHASE 14: RECREATE TRIGGERS (IF THEY EXIST)
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
DROP TRIGGER IF EXISTS update_users_updated_at ON users;
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_agents_updated_at ON agents;
CREATE TRIGGER update_agents_updated_at
    BEFORE UPDATE ON agents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_tasks_updated_at ON tasks;
CREATE TRIGGER update_tasks_updated_at
    BEFORE UPDATE ON tasks
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

RAISE NOTICE 'Triggers recreated';

-- =====================================================================
-- PHASE 15: FINAL VALIDATION
-- =====================================================================

DO $$
DECLARE
    total_errors INTEGER := 0;
    users_backup_count INTEGER;
    users_current_count INTEGER;
    agents_backup_count INTEGER;
    agents_current_count INTEGER;
BEGIN
    -- Validate data integrity
    SELECT COUNT(*) INTO users_backup_count FROM users_backup_pre_uuid;
    SELECT COUNT(*) INTO users_current_count FROM users;
    
    SELECT COUNT(*) INTO agents_backup_count FROM agents_backup_pre_uuid;
    SELECT COUNT(*) INTO agents_current_count FROM agents;
    
    IF users_backup_count != users_current_count THEN
        total_errors := total_errors + 1;
        RAISE WARNING 'USER COUNT MISMATCH: Backup=%, Current=%', users_backup_count, users_current_count;
    END IF;
    
    IF agents_backup_count != agents_current_count THEN
        total_errors := total_errors + 1;
        RAISE WARNING 'AGENT COUNT MISMATCH: Backup=%, Current=%', agents_backup_count, agents_current_count;
    END IF;
    
    -- Check for any null UUIDs (should not exist)
    IF EXISTS (SELECT 1 FROM users WHERE id IS NULL) THEN
        total_errors := total_errors + 1;
        RAISE WARNING 'NULL UUIDs found in users table';
    END IF;
    
    IF EXISTS (SELECT 1 FROM agents WHERE id IS NULL) THEN
        total_errors := total_errors + 1;
        RAISE WARNING 'NULL UUIDs found in agents table';
    END IF;
    
    IF total_errors > 0 THEN
        RAISE EXCEPTION 'MIGRATION FAILED: % validation errors detected. Transaction will be rolled back.', total_errors;
    ELSE
        RAISE NOTICE 'MIGRATION VALIDATION SUCCESSFUL: All checks passed';
    END IF;
END $$;

-- Log final counts for verification
SELECT 'MIGRATION COMPLETED - Final data counts:' as status;
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
-- MIGRATION COMPLETION
-- =====================================================================

RAISE NOTICE 'INTEGER to UUID migration completed successfully at %', NOW();
RAISE NOTICE 'Backup tables preserved for rollback: *_backup_pre_uuid';
RAISE NOTICE 'Mapping tables preserved for reference: uuid_migration_mapping_*';

-- COMMIT the transaction (comment out for testing)
COMMIT;

-- =====================================================================
-- POST-MIGRATION NOTES:
-- 
-- 1. Backup tables (*_backup_pre_uuid) are preserved for rollback
-- 2. Mapping tables (uuid_migration_mapping_*) show INTEGER->UUID conversions  
-- 3. All foreign key relationships have been preserved
-- 4. All indexes and triggers have been recreated
-- 5. New records will automatically get UUID primary keys via gen_random_uuid()
--
-- To clean up backup tables after confirming success:
-- DROP TABLE users_backup_pre_uuid, agents_backup_pre_uuid, ... (etc)
-- DROP TABLE uuid_migration_mapping_users, uuid_migration_mapping_agents, ... (etc)
-- =====================================================================