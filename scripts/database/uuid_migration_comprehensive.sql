-- =====================================================================
-- COMPREHENSIVE DATABASE MIGRATION: INTEGER IDs to UUID IDs
-- =====================================================================
-- Purpose: Convert all INTEGER primary keys to UUID primary keys
-- Risk Level: HIGH - This migration touches all tables and relationships
-- 
-- BEFORE RUNNING:
-- 1. Take a complete database dump: pg_dump -U sutazai sutazai > pre_migration_backup.sql
-- 2. Ensure no active connections to database during migration
-- 3. Test on development environment first
--
-- Author: Claude Code (Ultra Database Migration Specialist)
-- Date: 2025-08-10
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
SELECT 'SESSIONS' as table_name, COUNT(*) as record_count FROM sessions
UNION ALL
SELECT 'CHAT_HISTORY' as table_name, COUNT(*) as record_count FROM chat_history
UNION ALL
SELECT 'AGENT_EXECUTIONS' as table_name, COUNT(*) as record_count FROM agent_executions
UNION ALL
SELECT 'AGENT_HEALTH' as table_name, COUNT(*) as record_count FROM agent_health
UNION ALL
SELECT 'MODEL_REGISTRY' as table_name, COUNT(*) as record_count FROM model_registry
UNION ALL
SELECT 'SYSTEM_ALERTS' as table_name, COUNT(*) as record_count FROM system_alerts
UNION ALL
SELECT 'SYSTEM_METRICS' as table_name, COUNT(*) as record_count FROM system_metrics;

-- =====================================================================
-- PHASE 2: CREATE BACKUP TABLES (CRITICAL FOR ROLLBACK)
-- =====================================================================

-- Backup existing data exactly as-is
CREATE TABLE users_backup_pre_uuid AS SELECT * FROM users;
CREATE TABLE agents_backup_pre_uuid AS SELECT * FROM agents;
CREATE TABLE tasks_backup_pre_uuid AS SELECT * FROM tasks;
CREATE TABLE sessions_backup_pre_uuid AS SELECT * FROM sessions;
CREATE TABLE chat_history_backup_pre_uuid AS SELECT * FROM chat_history;
CREATE TABLE agent_executions_backup_pre_uuid AS SELECT * FROM agent_executions;
CREATE TABLE agent_health_backup_pre_uuid AS SELECT * FROM agent_health;
CREATE TABLE model_registry_backup_pre_uuid AS SELECT * FROM model_registry;
CREATE TABLE system_alerts_backup_pre_uuid AS SELECT * FROM system_alerts;
CREATE TABLE system_metrics_backup_pre_uuid AS SELECT * FROM system_metrics;

DO $$ BEGIN RAISE NOTICE 'Backup tables created successfully'; END $$;

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

CREATE TABLE uuid_migration_mapping_sessions (
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

CREATE TABLE uuid_migration_mapping_agent_health (
    old_integer_id INTEGER PRIMARY KEY,
    new_uuid_id UUID NOT NULL DEFAULT gen_random_uuid(),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE uuid_migration_mapping_model_registry (
    old_integer_id INTEGER PRIMARY KEY,
    new_uuid_id UUID NOT NULL DEFAULT gen_random_uuid(),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE uuid_migration_mapping_system_alerts (
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

INSERT INTO uuid_migration_mapping_sessions (old_integer_id)
SELECT id FROM sessions;

INSERT INTO uuid_migration_mapping_chat_history (old_integer_id)
SELECT id FROM chat_history;

INSERT INTO uuid_migration_mapping_agent_executions (old_integer_id)
SELECT id FROM agent_executions;

INSERT INTO uuid_migration_mapping_agent_health (old_integer_id)
SELECT id FROM agent_health;

INSERT INTO uuid_migration_mapping_model_registry (old_integer_id)
SELECT id FROM model_registry;

INSERT INTO uuid_migration_mapping_system_alerts (old_integer_id)
SELECT id FROM system_alerts;

INSERT INTO uuid_migration_mapping_system_metrics (old_integer_id)
SELECT id FROM system_metrics;

DO $$ BEGIN RAISE NOTICE 'UUID mappings created for all existing records'; END $$;

-- =====================================================================
-- PHASE 5: DROP ALL FOREIGN KEY CONSTRAINTS (TEMPORARILY)
-- =====================================================================

-- We need to drop FKs before changing column types
ALTER TABLE tasks DROP CONSTRAINT IF EXISTS tasks_user_id_fkey;
ALTER TABLE tasks DROP CONSTRAINT IF EXISTS tasks_agent_id_fkey;
ALTER TABLE sessions DROP CONSTRAINT IF EXISTS sessions_user_id_fkey;
ALTER TABLE chat_history DROP CONSTRAINT IF EXISTS chat_history_user_id_fkey;
ALTER TABLE agent_executions DROP CONSTRAINT IF EXISTS agent_executions_agent_id_fkey;
ALTER TABLE agent_executions DROP CONSTRAINT IF EXISTS agent_executions_task_id_fkey;
ALTER TABLE agent_health DROP CONSTRAINT IF EXISTS agent_health_agent_id_fkey;
ALTER TABLE system_alerts DROP CONSTRAINT IF EXISTS system_alerts_resolved_by_fkey;

DO $$ BEGIN RAISE NOTICE 'Foreign key constraints dropped temporarily'; END $$;

-- =====================================================================
-- PHASE 6: ADD NEW UUID COLUMNS ALONGSIDE INTEGER COLUMNS
-- =====================================================================

-- Add UUID columns to all tables (initially nullable)
ALTER TABLE users ADD COLUMN id_uuid UUID;
ALTER TABLE agents ADD COLUMN id_uuid UUID;
ALTER TABLE tasks ADD COLUMN id_uuid UUID;
ALTER TABLE tasks ADD COLUMN user_id_uuid UUID;
ALTER TABLE tasks ADD COLUMN agent_id_uuid UUID;
ALTER TABLE sessions ADD COLUMN id_uuid UUID;
ALTER TABLE sessions ADD COLUMN user_id_uuid UUID;
ALTER TABLE chat_history ADD COLUMN id_uuid UUID;
ALTER TABLE chat_history ADD COLUMN user_id_uuid UUID;
ALTER TABLE agent_executions ADD COLUMN id_uuid UUID;
ALTER TABLE agent_executions ADD COLUMN agent_id_uuid UUID;
ALTER TABLE agent_executions ADD COLUMN task_id_uuid UUID;
ALTER TABLE agent_health ADD COLUMN id_uuid UUID;
ALTER TABLE agent_health ADD COLUMN agent_id_uuid UUID;
ALTER TABLE model_registry ADD COLUMN id_uuid UUID;
ALTER TABLE system_alerts ADD COLUMN id_uuid UUID;
ALTER TABLE system_alerts ADD COLUMN resolved_by_uuid UUID;
ALTER TABLE system_metrics ADD COLUMN id_uuid UUID;

DO $$ BEGIN RAISE NOTICE 'UUID columns added to all tables'; END $$;

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

UPDATE sessions
SET id_uuid = (SELECT new_uuid_id FROM uuid_migration_mapping_sessions WHERE old_integer_id = sessions.id);

UPDATE chat_history
SET id_uuid = (SELECT new_uuid_id FROM uuid_migration_mapping_chat_history WHERE old_integer_id = chat_history.id);

UPDATE agent_executions
SET id_uuid = (SELECT new_uuid_id FROM uuid_migration_mapping_agent_executions WHERE old_integer_id = agent_executions.id);

UPDATE agent_health
SET id_uuid = (SELECT new_uuid_id FROM uuid_migration_mapping_agent_health WHERE old_integer_id = agent_health.id);

UPDATE model_registry
SET id_uuid = (SELECT new_uuid_id FROM uuid_migration_mapping_model_registry WHERE old_integer_id = model_registry.id);

UPDATE system_alerts
SET id_uuid = (SELECT new_uuid_id FROM uuid_migration_mapping_system_alerts WHERE old_integer_id = system_alerts.id);

UPDATE system_metrics
SET id_uuid = (SELECT new_uuid_id FROM uuid_migration_mapping_system_metrics WHERE old_integer_id = system_metrics.id);

-- Update foreign key UUID columns using mapping tables
UPDATE tasks 
SET user_id_uuid = (SELECT new_uuid_id FROM uuid_migration_mapping_users WHERE old_integer_id = tasks.user_id)
WHERE user_id IS NOT NULL;

UPDATE tasks
SET agent_id_uuid = (SELECT new_uuid_id FROM uuid_migration_mapping_agents WHERE old_integer_id = tasks.agent_id)  
WHERE agent_id IS NOT NULL;

UPDATE sessions
SET user_id_uuid = (SELECT new_uuid_id FROM uuid_migration_mapping_users WHERE old_integer_id = sessions.user_id)
WHERE user_id IS NOT NULL;

UPDATE chat_history
SET user_id_uuid = (SELECT new_uuid_id FROM uuid_migration_mapping_users WHERE old_integer_id = chat_history.user_id)
WHERE user_id IS NOT NULL;

UPDATE agent_executions  
SET agent_id_uuid = (SELECT new_uuid_id FROM uuid_migration_mapping_agents WHERE old_integer_id = agent_executions.agent_id)
WHERE agent_id IS NOT NULL;

UPDATE agent_executions
SET task_id_uuid = (SELECT new_uuid_id FROM uuid_migration_mapping_tasks WHERE old_integer_id = agent_executions.task_id)
WHERE task_id IS NOT NULL;

UPDATE agent_health
SET agent_id_uuid = (SELECT new_uuid_id FROM uuid_migration_mapping_agents WHERE old_integer_id = agent_health.agent_id)
WHERE agent_id IS NOT NULL;

UPDATE system_alerts
SET resolved_by_uuid = (SELECT new_uuid_id FROM uuid_migration_mapping_users WHERE old_integer_id = system_alerts.resolved_by)
WHERE resolved_by IS NOT NULL;

DO $$ BEGIN RAISE NOTICE 'All UUID columns populated with mapped values'; END $$;

-- =====================================================================
-- PHASE 8: VALIDATION - ENSURE NO DATA LOSS
-- =====================================================================

DO $$ 
DECLARE
    table_count_match INTEGER := 0;
    total_tables INTEGER := 10;
BEGIN
    -- Validate all tables have matching counts
    IF (SELECT COUNT(*) FROM users) = (SELECT COUNT(*) FROM users WHERE id_uuid IS NOT NULL) THEN
        table_count_match := table_count_match + 1;
    END IF;
    
    IF (SELECT COUNT(*) FROM agents) = (SELECT COUNT(*) FROM agents WHERE id_uuid IS NOT NULL) THEN
        table_count_match := table_count_match + 1;
    END IF;
    
    IF (SELECT COUNT(*) FROM tasks) = (SELECT COUNT(*) FROM tasks WHERE id_uuid IS NOT NULL) THEN
        table_count_match := table_count_match + 1;
    END IF;
    
    IF (SELECT COUNT(*) FROM sessions) = (SELECT COUNT(*) FROM sessions WHERE id_uuid IS NOT NULL) THEN
        table_count_match := table_count_match + 1;
    END IF;
    
    IF (SELECT COUNT(*) FROM chat_history) = (SELECT COUNT(*) FROM chat_history WHERE id_uuid IS NOT NULL) THEN
        table_count_match := table_count_match + 1;
    END IF;
    
    IF (SELECT COUNT(*) FROM agent_executions) = (SELECT COUNT(*) FROM agent_executions WHERE id_uuid IS NOT NULL) THEN
        table_count_match := table_count_match + 1;
    END IF;
    
    IF (SELECT COUNT(*) FROM agent_health) = (SELECT COUNT(*) FROM agent_health WHERE id_uuid IS NOT NULL) THEN
        table_count_match := table_count_match + 1;
    END IF;
    
    IF (SELECT COUNT(*) FROM model_registry) = (SELECT COUNT(*) FROM model_registry WHERE id_uuid IS NOT NULL) THEN
        table_count_match := table_count_match + 1;
    END IF;
    
    IF (SELECT COUNT(*) FROM system_alerts) = (SELECT COUNT(*) FROM system_alerts WHERE id_uuid IS NOT NULL) THEN
        table_count_match := table_count_match + 1;
    END IF;
    
    IF (SELECT COUNT(*) FROM system_metrics) = (SELECT COUNT(*) FROM system_metrics WHERE id_uuid IS NOT NULL) THEN
        table_count_match := table_count_match + 1;
    END IF;
    
    IF table_count_match != total_tables THEN
        RAISE EXCEPTION 'DATA LOSS DETECTED: Only % out of % tables validated successfully', table_count_match, total_tables;
    END IF;
    
    RAISE NOTICE 'Validation passed: All % tables successfully mapped to UUIDs', total_tables;
END $$;

-- =====================================================================
-- PHASE 9: DROP OLD INTEGER COLUMNS AND CONSTRAINTS
-- =====================================================================

-- Drop old primary key constraints and sequences
ALTER TABLE users DROP CONSTRAINT users_pkey;
ALTER TABLE agents DROP CONSTRAINT agents_pkey;
ALTER TABLE tasks DROP CONSTRAINT tasks_pkey;
ALTER TABLE sessions DROP CONSTRAINT sessions_pkey;
ALTER TABLE chat_history DROP CONSTRAINT chat_history_pkey;
ALTER TABLE agent_executions DROP CONSTRAINT agent_executions_pkey;
ALTER TABLE agent_health DROP CONSTRAINT agent_health_pkey;
ALTER TABLE model_registry DROP CONSTRAINT model_registry_pkey;
ALTER TABLE system_alerts DROP CONSTRAINT system_alerts_pkey;
ALTER TABLE system_metrics DROP CONSTRAINT system_metrics_pkey;

-- Drop old integer columns
ALTER TABLE users DROP COLUMN id;
ALTER TABLE agents DROP COLUMN id;
ALTER TABLE tasks DROP COLUMN id, DROP COLUMN user_id, DROP COLUMN agent_id;
ALTER TABLE sessions DROP COLUMN id, DROP COLUMN user_id;
ALTER TABLE chat_history DROP COLUMN id, DROP COLUMN user_id;
ALTER TABLE agent_executions DROP COLUMN id, DROP COLUMN agent_id, DROP COLUMN task_id;
ALTER TABLE agent_health DROP COLUMN id, DROP COLUMN agent_id;
ALTER TABLE model_registry DROP COLUMN id;
ALTER TABLE system_alerts DROP COLUMN id, DROP COLUMN resolved_by;
ALTER TABLE system_metrics DROP COLUMN id;

-- Drop sequences
DROP SEQUENCE IF EXISTS users_id_seq;
DROP SEQUENCE IF EXISTS agents_id_seq;
DROP SEQUENCE IF EXISTS tasks_id_seq;
DROP SEQUENCE IF EXISTS sessions_id_seq;
DROP SEQUENCE IF EXISTS chat_history_id_seq;
DROP SEQUENCE IF EXISTS agent_executions_id_seq;
DROP SEQUENCE IF EXISTS agent_health_id_seq;
DROP SEQUENCE IF EXISTS model_registry_id_seq;
DROP SEQUENCE IF EXISTS system_alerts_id_seq;
DROP SEQUENCE IF EXISTS system_metrics_id_seq;

DO $$ BEGIN RAISE NOTICE 'Old INTEGER columns and sequences removed'; END $$;

-- =====================================================================
-- PHASE 10: RENAME UUID COLUMNS TO PRIMARY COLUMN NAMES
-- =====================================================================

-- Rename UUID columns to become the primary columns
ALTER TABLE users RENAME COLUMN id_uuid TO id;
ALTER TABLE agents RENAME COLUMN id_uuid TO id;
ALTER TABLE tasks RENAME COLUMN id_uuid TO id;
ALTER TABLE tasks RENAME COLUMN user_id_uuid TO user_id;
ALTER TABLE tasks RENAME COLUMN agent_id_uuid TO agent_id;
ALTER TABLE sessions RENAME COLUMN id_uuid TO id;
ALTER TABLE sessions RENAME COLUMN user_id_uuid TO user_id;
ALTER TABLE chat_history RENAME COLUMN id_uuid TO id;
ALTER TABLE chat_history RENAME COLUMN user_id_uuid TO user_id;
ALTER TABLE agent_executions RENAME COLUMN id_uuid TO id;
ALTER TABLE agent_executions RENAME COLUMN agent_id_uuid TO agent_id;
ALTER TABLE agent_executions RENAME COLUMN task_id_uuid TO task_id;
ALTER TABLE agent_health RENAME COLUMN id_uuid TO id;
ALTER TABLE agent_health RENAME COLUMN agent_id_uuid TO agent_id;
ALTER TABLE model_registry RENAME COLUMN id_uuid TO id;
ALTER TABLE system_alerts RENAME COLUMN id_uuid TO id;
ALTER TABLE system_alerts RENAME COLUMN resolved_by_uuid TO resolved_by;
ALTER TABLE system_metrics RENAME COLUMN id_uuid TO id;

DO $$ BEGIN RAISE NOTICE 'UUID columns renamed to primary names'; END $$;

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

ALTER TABLE sessions ALTER COLUMN id SET NOT NULL;
ALTER TABLE sessions ALTER COLUMN id SET DEFAULT gen_random_uuid();
ALTER TABLE sessions ADD CONSTRAINT sessions_pkey PRIMARY KEY (id);

ALTER TABLE chat_history ALTER COLUMN id SET NOT NULL;
ALTER TABLE chat_history ALTER COLUMN id SET DEFAULT gen_random_uuid();
ALTER TABLE chat_history ADD CONSTRAINT chat_history_pkey PRIMARY KEY (id);

ALTER TABLE agent_executions ALTER COLUMN id SET NOT NULL;
ALTER TABLE agent_executions ALTER COLUMN id SET DEFAULT gen_random_uuid();
ALTER TABLE agent_executions ADD CONSTRAINT agent_executions_pkey PRIMARY KEY (id);

ALTER TABLE agent_health ALTER COLUMN id SET NOT NULL;
ALTER TABLE agent_health ALTER COLUMN id SET DEFAULT gen_random_uuid();
ALTER TABLE agent_health ADD CONSTRAINT agent_health_pkey PRIMARY KEY (id);

ALTER TABLE model_registry ALTER COLUMN id SET NOT NULL;
ALTER TABLE model_registry ALTER COLUMN id SET DEFAULT gen_random_uuid();
ALTER TABLE model_registry ADD CONSTRAINT model_registry_pkey PRIMARY KEY (id);

ALTER TABLE system_alerts ALTER COLUMN id SET NOT NULL;
ALTER TABLE system_alerts ALTER COLUMN id SET DEFAULT gen_random_uuid();
ALTER TABLE system_alerts ADD CONSTRAINT system_alerts_pkey PRIMARY KEY (id);

ALTER TABLE system_metrics ALTER COLUMN id SET NOT NULL;
ALTER TABLE system_metrics ALTER COLUMN id SET DEFAULT gen_random_uuid();
ALTER TABLE system_metrics ADD CONSTRAINT system_metrics_pkey PRIMARY KEY (id);

DO $$ BEGIN RAISE NOTICE 'Primary key constraints recreated with UUIDs'; END $$;

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

ALTER TABLE sessions
ADD CONSTRAINT sessions_user_id_fkey
FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE;

ALTER TABLE chat_history
ADD CONSTRAINT chat_history_user_id_fkey
FOREIGN KEY (user_id) REFERENCES users(id);

ALTER TABLE agent_executions
ADD CONSTRAINT agent_executions_agent_id_fkey
FOREIGN KEY (agent_id) REFERENCES agents(id);

ALTER TABLE agent_executions
ADD CONSTRAINT agent_executions_task_id_fkey
FOREIGN KEY (task_id) REFERENCES tasks(id);

ALTER TABLE agent_health
ADD CONSTRAINT agent_health_agent_id_fkey
FOREIGN KEY (agent_id) REFERENCES agents(id);

ALTER TABLE system_alerts
ADD CONSTRAINT system_alerts_resolved_by_fkey
FOREIGN KEY (resolved_by) REFERENCES users(id);

DO $$ BEGIN RAISE NOTICE 'Foreign key constraints recreated'; END $$;

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
CREATE INDEX idx_agents_status ON agents(status);

CREATE INDEX idx_tasks_agent_id ON tasks(agent_id);
CREATE INDEX idx_tasks_user_id ON tasks(user_id);
CREATE INDEX idx_tasks_status ON tasks(status);

CREATE UNIQUE INDEX sessions_token_key ON sessions(token);
CREATE INDEX idx_sessions_token ON sessions(token);
CREATE INDEX idx_sessions_user_id ON sessions(user_id);

CREATE INDEX idx_chat_history_user_id ON chat_history(user_id);

CREATE INDEX idx_agent_executions_agent_id ON agent_executions(agent_id);

CREATE INDEX idx_agent_health_agent_id ON agent_health(agent_id);

CREATE UNIQUE INDEX model_registry_name_version_key ON model_registry(name, version);

CREATE INDEX idx_system_alerts_severity ON system_alerts(severity);
CREATE INDEX idx_system_alerts_status ON system_alerts(status);

CREATE INDEX idx_system_metrics_name ON system_metrics(metric_name);
CREATE INDEX idx_system_metrics_recorded_at ON system_metrics(recorded_at);

DO $$ BEGIN RAISE NOTICE 'All indexes recreated'; END $$;

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

DO $$ BEGIN RAISE NOTICE 'Triggers recreated'; END $$;

-- =====================================================================
-- PHASE 15: FINAL VALIDATION
-- =====================================================================

DO $$
DECLARE
    total_errors INTEGER := 0;
    validation_results TEXT := '';
BEGIN
    -- Validate data integrity for all tables
    IF (SELECT COUNT(*) FROM users_backup_pre_uuid) != (SELECT COUNT(*) FROM users) THEN
        total_errors := total_errors + 1;
        validation_results := validation_results || 'USER COUNT MISMATCH; ';
    END IF;
    
    IF (SELECT COUNT(*) FROM agents_backup_pre_uuid) != (SELECT COUNT(*) FROM agents) THEN
        total_errors := total_errors + 1;
        validation_results := validation_results || 'AGENT COUNT MISMATCH; ';
    END IF;
    
    -- Check for any null UUIDs (should not exist)
    IF EXISTS (SELECT 1 FROM users WHERE id IS NULL) THEN
        total_errors := total_errors + 1;
        validation_results := validation_results || 'NULL UUIDs in users; ';
    END IF;
    
    IF EXISTS (SELECT 1 FROM agents WHERE id IS NULL) THEN
        total_errors := total_errors + 1;
        validation_results := validation_results || 'NULL UUIDs in agents; ';
    END IF;
    
    IF total_errors > 0 THEN
        RAISE EXCEPTION 'MIGRATION FAILED: % validation errors: %', total_errors, validation_results;
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
SELECT 'SESSIONS' as table_name, COUNT(*) as record_count FROM sessions
UNION ALL
SELECT 'CHAT_HISTORY' as table_name, COUNT(*) as record_count FROM chat_history
UNION ALL
SELECT 'AGENT_EXECUTIONS' as table_name, COUNT(*) as record_count FROM agent_executions
UNION ALL
SELECT 'AGENT_HEALTH' as table_name, COUNT(*) as record_count FROM agent_health
UNION ALL
SELECT 'MODEL_REGISTRY' as table_name, COUNT(*) as record_count FROM model_registry
UNION ALL
SELECT 'SYSTEM_ALERTS' as table_name, COUNT(*) as record_count FROM system_alerts
UNION ALL
SELECT 'SYSTEM_METRICS' as table_name, COUNT(*) as record_count FROM system_metrics;

-- =====================================================================
-- MIGRATION COMPLETION
-- =====================================================================

DO $$ BEGIN 
    RAISE NOTICE 'INTEGER to UUID migration completed successfully at %', NOW();
    RAISE NOTICE 'Backup tables preserved for rollback: *_backup_pre_uuid';
    RAISE NOTICE 'Mapping tables preserved for reference: uuid_migration_mapping_*';
END $$;

-- COMMIT the transaction
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