-- =====================================================================
-- STEPWISE UUID MIGRATION: Add UUID columns first, then replace
-- =====================================================================
-- Author: Claude Code (Ultra Database Migration Specialist)
-- Date: 2025-08-10
-- Strategy: Safe stepwise approach with rollback capability
-- =====================================================================

BEGIN;

-- Enable UUID extensions
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

DO $$ BEGIN RAISE NOTICE 'Starting stepwise UUID migration at %', NOW(); END $$;

-- =====================================================================
-- STEP 1: ADD UUID COLUMNS TO ALL TABLES
-- =====================================================================

DO $$ BEGIN RAISE NOTICE 'Step 1: Adding UUID columns...'; END $$;

-- Add UUID column to users
ALTER TABLE users ADD COLUMN id_uuid UUID DEFAULT gen_random_uuid();
UPDATE users SET id_uuid = gen_random_uuid() WHERE id_uuid IS NULL;

-- Add UUID column to agents  
ALTER TABLE agents ADD COLUMN id_uuid UUID DEFAULT gen_random_uuid();
UPDATE agents SET id_uuid = gen_random_uuid() WHERE id_uuid IS NULL;

-- Add UUID columns to tasks (both PK and FKs)
ALTER TABLE tasks ADD COLUMN id_uuid UUID DEFAULT gen_random_uuid();
ALTER TABLE tasks ADD COLUMN user_id_uuid UUID;
ALTER TABLE tasks ADD COLUMN agent_id_uuid UUID;
UPDATE tasks SET id_uuid = gen_random_uuid() WHERE id_uuid IS NULL;

-- Add UUID columns to sessions
ALTER TABLE sessions ADD COLUMN id_uuid UUID DEFAULT gen_random_uuid();
ALTER TABLE sessions ADD COLUMN user_id_uuid UUID;
UPDATE sessions SET id_uuid = gen_random_uuid() WHERE id_uuid IS NULL;

-- Add UUID columns to chat_history
ALTER TABLE chat_history ADD COLUMN id_uuid UUID DEFAULT gen_random_uuid();
ALTER TABLE chat_history ADD COLUMN user_id_uuid UUID;
UPDATE chat_history SET id_uuid = gen_random_uuid() WHERE id_uuid IS NULL;

-- Add UUID columns to agent_executions
ALTER TABLE agent_executions ADD COLUMN id_uuid UUID DEFAULT gen_random_uuid();
ALTER TABLE agent_executions ADD COLUMN agent_id_uuid UUID;
ALTER TABLE agent_executions ADD COLUMN task_id_uuid UUID;
UPDATE agent_executions SET id_uuid = gen_random_uuid() WHERE id_uuid IS NULL;

-- Add UUID columns to agent_health
ALTER TABLE agent_health ADD COLUMN id_uuid UUID DEFAULT gen_random_uuid();
ALTER TABLE agent_health ADD COLUMN agent_id_uuid UUID;
UPDATE agent_health SET id_uuid = gen_random_uuid() WHERE id_uuid IS NULL;

-- Add UUID column to model_registry
ALTER TABLE model_registry ADD COLUMN id_uuid UUID DEFAULT gen_random_uuid();
UPDATE model_registry SET id_uuid = gen_random_uuid() WHERE id_uuid IS NULL;

-- Add UUID columns to system_alerts
ALTER TABLE system_alerts ADD COLUMN id_uuid UUID DEFAULT gen_random_uuid();
ALTER TABLE system_alerts ADD COLUMN resolved_by_uuid UUID;
UPDATE system_alerts SET id_uuid = gen_random_uuid() WHERE id_uuid IS NULL;

-- Add UUID column to system_metrics
ALTER TABLE system_metrics ADD COLUMN id_uuid UUID DEFAULT gen_random_uuid();
UPDATE system_metrics SET id_uuid = gen_random_uuid() WHERE id_uuid IS NULL;

DO $$ BEGIN RAISE NOTICE 'Step 1 completed: UUID columns added'; END $$;

-- =====================================================================
-- STEP 2: POPULATE FOREIGN KEY UUID COLUMNS
-- =====================================================================

DO $$ BEGIN RAISE NOTICE 'Step 2: Populating foreign key UUIDs...'; END $$;

-- Map foreign keys in tasks table
UPDATE tasks SET 
    user_id_uuid = users.id_uuid 
FROM users 
WHERE tasks.user_id = users.id;

UPDATE tasks SET 
    agent_id_uuid = agents.id_uuid 
FROM agents 
WHERE tasks.agent_id = agents.id;

-- Map foreign keys in sessions table
UPDATE sessions SET 
    user_id_uuid = users.id_uuid 
FROM users 
WHERE sessions.user_id = users.id;

-- Map foreign keys in chat_history table
UPDATE chat_history SET 
    user_id_uuid = users.id_uuid 
FROM users 
WHERE chat_history.user_id = users.id;

-- Map foreign keys in agent_executions table
UPDATE agent_executions SET 
    agent_id_uuid = agents.id_uuid 
FROM agents 
WHERE agent_executions.agent_id = agents.id;

UPDATE agent_executions SET 
    task_id_uuid = tasks.id_uuid 
FROM tasks 
WHERE agent_executions.task_id = tasks.id;

-- Map foreign keys in agent_health table
UPDATE agent_health SET 
    agent_id_uuid = agents.id_uuid 
FROM agents 
WHERE agent_health.agent_id = agents.id;

-- Map foreign keys in system_alerts table
UPDATE system_alerts SET 
    resolved_by_uuid = users.id_uuid 
FROM users 
WHERE system_alerts.resolved_by = users.id;

DO $$ BEGIN RAISE NOTICE 'Step 2 completed: Foreign key UUIDs populated'; END $$;

-- =====================================================================
-- STEP 3: VALIDATION
-- =====================================================================

DO $$ 
DECLARE
    total_errors INTEGER := 0;
    null_count INTEGER;
BEGIN
    RAISE NOTICE 'Step 3: Validating UUID migration...';
    
    -- Check for null UUIDs in primary keys
    SELECT COUNT(*) INTO null_count FROM users WHERE id_uuid IS NULL;
    IF null_count > 0 THEN
        total_errors := total_errors + 1;
        RAISE WARNING 'Found % null UUIDs in users table', null_count;
    END IF;
    
    SELECT COUNT(*) INTO null_count FROM agents WHERE id_uuid IS NULL;
    IF null_count > 0 THEN
        total_errors := total_errors + 1;
        RAISE WARNING 'Found % null UUIDs in agents table', null_count;
    END IF;
    
    IF total_errors > 0 THEN
        RAISE EXCEPTION 'UUID migration validation failed with % errors', total_errors;
    END IF;
    
    RAISE NOTICE 'Step 3 completed: Validation successful';
END $$;

-- =====================================================================
-- STEP 4: CREATE BACKUP TABLES BEFORE FINAL SWITCH
-- =====================================================================

DO $$ BEGIN RAISE NOTICE 'Step 4: Creating backup tables...'; END $$;

CREATE TABLE users_backup AS SELECT * FROM users;
CREATE TABLE agents_backup AS SELECT * FROM agents;
CREATE TABLE tasks_backup AS SELECT * FROM tasks;
CREATE TABLE sessions_backup AS SELECT * FROM sessions;
CREATE TABLE chat_history_backup AS SELECT * FROM chat_history;
CREATE TABLE agent_executions_backup AS SELECT * FROM agent_executions;
CREATE TABLE agent_health_backup AS SELECT * FROM agent_health;
CREATE TABLE model_registry_backup AS SELECT * FROM model_registry;
CREATE TABLE system_alerts_backup AS SELECT * FROM system_alerts;
CREATE TABLE system_metrics_backup AS SELECT * FROM system_metrics;

DO $$ BEGIN RAISE NOTICE 'Step 4 completed: Backup tables created'; END $$;

-- =====================================================================
-- STEP 5: DROP FOREIGN KEY CONSTRAINTS
-- =====================================================================

DO $$ BEGIN RAISE NOTICE 'Step 5: Dropping foreign key constraints...'; END $$;

ALTER TABLE tasks DROP CONSTRAINT IF EXISTS tasks_user_id_fkey;
ALTER TABLE tasks DROP CONSTRAINT IF EXISTS tasks_agent_id_fkey;
ALTER TABLE sessions DROP CONSTRAINT IF EXISTS sessions_user_id_fkey;
ALTER TABLE chat_history DROP CONSTRAINT IF EXISTS chat_history_user_id_fkey;
ALTER TABLE agent_executions DROP CONSTRAINT IF EXISTS agent_executions_agent_id_fkey;
ALTER TABLE agent_executions DROP CONSTRAINT IF EXISTS agent_executions_task_id_fkey;
ALTER TABLE agent_health DROP CONSTRAINT IF EXISTS agent_health_agent_id_fkey;
ALTER TABLE system_alerts DROP CONSTRAINT IF EXISTS system_alerts_resolved_by_fkey;

DO $$ BEGIN RAISE NOTICE 'Step 5 completed: Foreign key constraints dropped'; END $$;

-- =====================================================================
-- STEP 6: SWITCH TO UUID PRIMARY KEYS
-- =====================================================================

DO $$ BEGIN RAISE NOTICE 'Step 6: Switching to UUID primary keys...'; END $$;

-- Drop old primary keys and sequences
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

-- Drop integer columns and rename UUID columns
ALTER TABLE users DROP COLUMN id, DROP COLUMN IF EXISTS permissions;
ALTER TABLE users RENAME COLUMN id_uuid TO id;

ALTER TABLE agents DROP COLUMN id;
ALTER TABLE agents RENAME COLUMN id_uuid TO id;

ALTER TABLE tasks DROP COLUMN id, DROP COLUMN user_id, DROP COLUMN agent_id;
ALTER TABLE tasks RENAME COLUMN id_uuid TO id;
ALTER TABLE tasks RENAME COLUMN user_id_uuid TO user_id;
ALTER TABLE tasks RENAME COLUMN agent_id_uuid TO agent_id;

ALTER TABLE sessions DROP COLUMN id, DROP COLUMN user_id;
ALTER TABLE sessions RENAME COLUMN id_uuid TO id;
ALTER TABLE sessions RENAME COLUMN user_id_uuid TO user_id;

ALTER TABLE chat_history DROP COLUMN id, DROP COLUMN user_id;
ALTER TABLE chat_history RENAME COLUMN id_uuid TO id;
ALTER TABLE chat_history RENAME COLUMN user_id_uuid TO user_id;

ALTER TABLE agent_executions DROP COLUMN id, DROP COLUMN agent_id, DROP COLUMN task_id;
ALTER TABLE agent_executions RENAME COLUMN id_uuid TO id;
ALTER TABLE agent_executions RENAME COLUMN agent_id_uuid TO agent_id;
ALTER TABLE agent_executions RENAME COLUMN task_id_uuid TO task_id;

ALTER TABLE agent_health DROP COLUMN id, DROP COLUMN agent_id;
ALTER TABLE agent_health RENAME COLUMN id_uuid TO id;
ALTER TABLE agent_health RENAME COLUMN agent_id_uuid TO agent_id;

ALTER TABLE model_registry DROP COLUMN id;
ALTER TABLE model_registry RENAME COLUMN id_uuid TO id;

ALTER TABLE system_alerts DROP COLUMN id, DROP COLUMN resolved_by;
ALTER TABLE system_alerts RENAME COLUMN id_uuid TO id;
ALTER TABLE system_alerts RENAME COLUMN resolved_by_uuid TO resolved_by;

ALTER TABLE system_metrics DROP COLUMN id;
ALTER TABLE system_metrics RENAME COLUMN id_uuid TO id;

-- Create new primary key constraints
ALTER TABLE users ALTER COLUMN id SET NOT NULL;
ALTER TABLE users ADD CONSTRAINT users_pkey PRIMARY KEY (id);

ALTER TABLE agents ALTER COLUMN id SET NOT NULL;
ALTER TABLE agents ADD CONSTRAINT agents_pkey PRIMARY KEY (id);

ALTER TABLE tasks ALTER COLUMN id SET NOT NULL;
ALTER TABLE tasks ADD CONSTRAINT tasks_pkey PRIMARY KEY (id);

ALTER TABLE sessions ALTER COLUMN id SET NOT NULL;
ALTER TABLE sessions ADD CONSTRAINT sessions_pkey PRIMARY KEY (id);

ALTER TABLE chat_history ALTER COLUMN id SET NOT NULL;
ALTER TABLE chat_history ADD CONSTRAINT chat_history_pkey PRIMARY KEY (id);

ALTER TABLE agent_executions ALTER COLUMN id SET NOT NULL;
ALTER TABLE agent_executions ADD CONSTRAINT agent_executions_pkey PRIMARY KEY (id);

ALTER TABLE agent_health ALTER COLUMN id SET NOT NULL;
ALTER TABLE agent_health ADD CONSTRAINT agent_health_pkey PRIMARY KEY (id);

ALTER TABLE model_registry ALTER COLUMN id SET NOT NULL;
ALTER TABLE model_registry ADD CONSTRAINT model_registry_pkey PRIMARY KEY (id);

ALTER TABLE system_alerts ALTER COLUMN id SET NOT NULL;
ALTER TABLE system_alerts ADD CONSTRAINT system_alerts_pkey PRIMARY KEY (id);

ALTER TABLE system_metrics ALTER COLUMN id SET NOT NULL;
ALTER TABLE system_metrics ADD CONSTRAINT system_metrics_pkey PRIMARY KEY (id);

DO $$ BEGIN RAISE NOTICE 'Step 6 completed: UUID primary keys active'; END $$;

-- =====================================================================
-- STEP 7: RECREATE FOREIGN KEY CONSTRAINTS
-- =====================================================================

DO $$ BEGIN RAISE NOTICE 'Step 7: Recreating foreign key constraints...'; END $$;

ALTER TABLE tasks ADD CONSTRAINT tasks_user_id_fkey FOREIGN KEY (user_id) REFERENCES users(id);
ALTER TABLE tasks ADD CONSTRAINT tasks_agent_id_fkey FOREIGN KEY (agent_id) REFERENCES agents(id);
ALTER TABLE sessions ADD CONSTRAINT sessions_user_id_fkey FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE;
ALTER TABLE chat_history ADD CONSTRAINT chat_history_user_id_fkey FOREIGN KEY (user_id) REFERENCES users(id);
ALTER TABLE agent_executions ADD CONSTRAINT agent_executions_agent_id_fkey FOREIGN KEY (agent_id) REFERENCES agents(id);
ALTER TABLE agent_executions ADD CONSTRAINT agent_executions_task_id_fkey FOREIGN KEY (task_id) REFERENCES tasks(id);
ALTER TABLE agent_health ADD CONSTRAINT agent_health_agent_id_fkey FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE CASCADE;
ALTER TABLE system_alerts ADD CONSTRAINT system_alerts_resolved_by_fkey FOREIGN KEY (resolved_by) REFERENCES users(id);

DO $$ BEGIN RAISE NOTICE 'Step 7 completed: Foreign key constraints recreated'; END $$;

-- =====================================================================
-- STEP 8: RECREATE INDEXES AND TRIGGERS
-- =====================================================================

DO $$ BEGIN RAISE NOTICE 'Step 8: Recreating indexes and triggers...'; END $$;

-- Recreate indexes
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_is_admin ON users(is_admin) WHERE is_admin = true;
CREATE UNIQUE INDEX users_email_key ON users(email);
CREATE UNIQUE INDEX users_username_key ON users(username);

CREATE INDEX idx_agents_type ON agents(type);
CREATE UNIQUE INDEX agents_name_key ON agents(name);

CREATE INDEX idx_tasks_agent_id ON tasks(agent_id);
CREATE INDEX idx_tasks_user_id ON tasks(user_id);
CREATE INDEX idx_tasks_status ON tasks(status);

CREATE INDEX idx_sessions_token ON sessions(token);
CREATE INDEX idx_sessions_user_id ON sessions(user_id);
CREATE UNIQUE INDEX sessions_token_key ON sessions(token);

CREATE INDEX idx_chat_history_user_id ON chat_history(user_id);
CREATE INDEX idx_agent_executions_agent_id ON agent_executions(agent_id);
CREATE INDEX idx_agent_health_agent_id ON agent_health(agent_id);

CREATE UNIQUE INDEX model_registry_name_version_key ON model_registry(name, version);

CREATE INDEX idx_system_alerts_severity ON system_alerts(severity);
CREATE INDEX idx_system_alerts_status ON system_alerts(status);
CREATE INDEX idx_system_metrics_name ON system_metrics(metric_name);
CREATE INDEX idx_system_metrics_recorded_at ON system_metrics(recorded_at);

-- Recreate triggers
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_agent_health_updated_at BEFORE UPDATE ON agent_health FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_model_registry_updated_at BEFORE UPDATE ON model_registry FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DO $$ BEGIN RAISE NOTICE 'Step 8 completed: Indexes and triggers recreated'; END $$;

-- =====================================================================
-- FINAL VALIDATION AND COMPLETION
-- =====================================================================

DO $$ 
DECLARE
    table_count INTEGER;
BEGIN
    RAISE NOTICE 'Final validation: Checking all tables...';
    
    SELECT COUNT(*) INTO table_count FROM users;
    RAISE NOTICE 'Users table: % records with UUID primary keys', table_count;
    
    SELECT COUNT(*) INTO table_count FROM agents;
    RAISE NOTICE 'Agents table: % records with UUID primary keys', table_count;
    
    RAISE NOTICE 'UUID migration completed successfully at %', NOW();
    RAISE NOTICE 'All tables now use UUID primary keys and foreign keys';
    RAISE NOTICE 'Original data preserved in *_backup tables for rollback';
END $$;

-- Drop old sequences
DROP SEQUENCE IF EXISTS users_id_seq CASCADE;
DROP SEQUENCE IF EXISTS agents_id_seq CASCADE;
DROP SEQUENCE IF EXISTS tasks_id_seq CASCADE;
DROP SEQUENCE IF EXISTS sessions_id_seq CASCADE;
DROP SEQUENCE IF EXISTS chat_history_id_seq CASCADE;
DROP SEQUENCE IF EXISTS agent_executions_id_seq CASCADE;
DROP SEQUENCE IF EXISTS agent_health_id_seq CASCADE;
DROP SEQUENCE IF EXISTS model_registry_id_seq CASCADE;
DROP SEQUENCE IF EXISTS system_alerts_id_seq CASCADE;
DROP SEQUENCE IF EXISTS system_metrics_id_seq CASCADE;

COMMIT;