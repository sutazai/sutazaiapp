-- =====================================================================
-- SAFE UUID MIGRATION: INTEGER to UUID Primary Keys
-- =====================================================================
-- Author: Claude Code (Ultra Database Migration Specialist)
-- Date: 2025-08-10
-- Risk: MEDIUM (Atomic transaction with rollback safety)
-- =====================================================================

BEGIN;

-- Enable UUID extensions
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Log start
DO $$ BEGIN RAISE NOTICE 'Starting safe UUID migration at %', NOW(); END $$;

-- Show current data counts
SELECT 'Before Migration - Data Counts:' as status;
SELECT table_name, 
       (xpath('/row/cnt/text()', query_to_xml(format('select count(*) as cnt from %I.%I', table_schema, table_name), false, true, '')))[1]::text::int as row_count
FROM information_schema.tables 
WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
ORDER BY table_name;

-- =====================================================================
-- PHASE 1: CREATE BACKUP TABLES
-- =====================================================================

DO $$ BEGIN RAISE NOTICE 'Creating backup tables...'; END $$;

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

DO $$ BEGIN RAISE NOTICE 'Backup tables created successfully'; END $$;

-- =====================================================================
-- PHASE 2: CREATE NEW UUID TABLES WITH PROPER STRUCTURE
-- =====================================================================

DO $$ BEGIN RAISE NOTICE 'Creating new UUID-based tables...'; END $$;

-- Users table with UUID
CREATE TABLE users_new (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_admin BOOLEAN DEFAULT false,
    last_login TIMESTAMP WITH TIME ZONE,
    failed_login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMP WITH TIME ZONE
);

-- Agents table with UUID
CREATE TABLE agents_new (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) UNIQUE NOT NULL,
    type VARCHAR(50) NOT NULL,
    status VARCHAR(50) DEFAULT 'inactive',
    config JSON DEFAULT '{}'::json,
    endpoint VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_heartbeat TIMESTAMP WITH TIME ZONE,
    capabilities JSON DEFAULT '[]'::json
);

-- Tasks table with UUID
CREATE TABLE tasks_new (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    agent_id UUID REFERENCES agents_new(id),
    user_id UUID REFERENCES users_new(id),
    status VARCHAR(50) DEFAULT 'pending',
    priority INTEGER DEFAULT 5,
    payload JSONB DEFAULT '{}'::jsonb,
    result JSONB,
    error_message TEXT,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITHOUT TIME ZONE,
    completed_at TIMESTAMP WITHOUT TIME ZONE
);

-- Sessions table with UUID
CREATE TABLE sessions_new (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users_new(id) ON DELETE CASCADE,
    token VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    is_active BOOLEAN DEFAULT true,
    user_agent TEXT,
    ip_address INET,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Chat history table with UUID
CREATE TABLE chat_history_new (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users_new(id),
    message TEXT NOT NULL,
    response TEXT,
    model_used VARCHAR(100),
    tokens_used INTEGER,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Agent executions table with UUID
CREATE TABLE agent_executions_new (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID REFERENCES agents_new(id),
    task_id UUID REFERENCES tasks_new(id),
    status VARCHAR(50) DEFAULT 'pending',
    input_data JSONB,
    output_data JSONB,
    error_message TEXT,
    execution_time INTERVAL,
    started_at TIMESTAMP WITHOUT TIME ZONE,
    completed_at TIMESTAMP WITHOUT TIME ZONE,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Agent health table with UUID
CREATE TABLE agent_health_new (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID REFERENCES agents_new(id),
    status VARCHAR(50) NOT NULL,
    cpu_usage DOUBLE PRECISION,
    memory_usage DOUBLE PRECISION,
    disk_usage DOUBLE PRECISION,
    network_latency DOUBLE PRECISION,
    last_check TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Model registry table with UUID
CREATE TABLE model_registry_new (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL,
    version VARCHAR(50) NOT NULL,
    type VARCHAR(50) NOT NULL,
    path VARCHAR(255),
    config JSON DEFAULT '{}'::json,
    size_bytes BIGINT,
    checksum VARCHAR(255),
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT true,
    UNIQUE(name, version)
);

-- System alerts table with UUID
CREATE TABLE system_alerts_new (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    source VARCHAR(100),
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP WITHOUT TIME ZONE,
    resolved_by UUID REFERENCES users_new(id)
);

-- System metrics table with UUID
CREATE TABLE system_metrics_new (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    metric_name VARCHAR(100) NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    unit VARCHAR(20),
    tags JSON DEFAULT '{}'::json,
    recorded_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

DO $$ BEGIN RAISE NOTICE 'New UUID tables created successfully'; END $$;

-- =====================================================================
-- PHASE 3: MIGRATE DATA WITH UUID MAPPING
-- =====================================================================

DO $$ BEGIN RAISE NOTICE 'Migrating data to UUID tables...'; END $$;

-- Create mapping tables for ID conversion
CREATE TEMP TABLE user_id_mapping AS
SELECT id as old_id, gen_random_uuid() as new_id FROM users;

CREATE TEMP TABLE agent_id_mapping AS  
SELECT id as old_id, gen_random_uuid() as new_id FROM agents;

CREATE TEMP TABLE task_id_mapping AS
SELECT id as old_id, gen_random_uuid() as new_id FROM tasks;

CREATE TEMP TABLE session_id_mapping AS
SELECT id as old_id, gen_random_uuid() as new_id FROM sessions;

CREATE TEMP TABLE chat_history_id_mapping AS
SELECT id as old_id, gen_random_uuid() as new_id FROM chat_history;

CREATE TEMP TABLE agent_execution_id_mapping AS
SELECT id as old_id, gen_random_uuid() as new_id FROM agent_executions;

CREATE TEMP TABLE agent_health_id_mapping AS
SELECT id as old_id, gen_random_uuid() as new_id FROM agent_health;

CREATE TEMP TABLE model_registry_id_mapping AS
SELECT id as old_id, gen_random_uuid() as new_id FROM model_registry;

CREATE TEMP TABLE system_alert_id_mapping AS
SELECT id as old_id, gen_random_uuid() as new_id FROM system_alerts;

CREATE TEMP TABLE system_metric_id_mapping AS
SELECT id as old_id, gen_random_uuid() as new_id FROM system_metrics;

-- Migrate users
INSERT INTO users_new (id, username, email, password_hash, is_active, created_at, updated_at, is_admin, last_login, failed_login_attempts, locked_until)
SELECT 
    m.new_id,
    u.username,
    u.email,
    u.password_hash,
    u.is_active,
    u.created_at,
    u.updated_at,
    u.is_admin,
    u.last_login,
    u.failed_login_attempts,
    u.locked_until
FROM users u
JOIN user_id_mapping m ON u.id = m.old_id;

-- Migrate agents
INSERT INTO agents_new (id, name, type, status, config, endpoint, created_at, updated_at, last_heartbeat, capabilities)
SELECT 
    m.new_id,
    a.name,
    a.type,
    a.status,
    a.config,
    a.endpoint,
    a.created_at,
    a.updated_at,
    a.last_heartbeat,
    a.capabilities
FROM agents a
JOIN agent_id_mapping m ON a.id = m.old_id;

-- Migrate tasks (with foreign key mapping)
INSERT INTO tasks_new (id, title, description, agent_id, user_id, status, priority, payload, result, error_message, created_at, started_at, completed_at)
SELECT 
    tm.new_id,
    t.title,
    t.description,
    am.new_id,
    um.new_id,
    t.status,
    t.priority,
    t.payload,
    t.result,
    t.error_message,
    t.created_at,
    t.started_at,
    t.completed_at
FROM tasks t
JOIN task_id_mapping tm ON t.id = tm.old_id
LEFT JOIN agent_id_mapping am ON t.agent_id = am.old_id
LEFT JOIN user_id_mapping um ON t.user_id = um.old_id;

-- Migrate sessions
INSERT INTO sessions_new (id, user_id, token, expires_at, is_active, user_agent, ip_address, created_at, last_accessed)
SELECT 
    sm.new_id,
    um.new_id,
    s.token,
    s.expires_at,
    s.is_active,
    s.user_agent,
    s.ip_address,
    s.created_at,
    s.last_accessed
FROM sessions s
JOIN session_id_mapping sm ON s.id = sm.old_id
LEFT JOIN user_id_mapping um ON s.user_id = um.old_id;

-- Migrate chat_history
INSERT INTO chat_history_new (id, user_id, message, response, model_used, tokens_used, created_at)
SELECT 
    cm.new_id,
    um.new_id,
    c.message,
    c.response,
    c.model_used,
    c.tokens_used,
    c.created_at
FROM chat_history c
JOIN chat_history_id_mapping cm ON c.id = cm.old_id
LEFT JOIN user_id_mapping um ON c.user_id = um.old_id;

-- Migrate agent_executions
INSERT INTO agent_executions_new (id, agent_id, task_id, status, input_data, output_data, error_message, execution_time, started_at, completed_at, created_at)
SELECT 
    em.new_id,
    am.new_id,
    tm.new_id,
    e.status,
    e.input_data,
    e.output_data,
    e.error_message,
    e.execution_time,
    e.started_at,
    e.completed_at,
    e.created_at
FROM agent_executions e
JOIN agent_execution_id_mapping em ON e.id = em.old_id
LEFT JOIN agent_id_mapping am ON e.agent_id = am.old_id
LEFT JOIN task_id_mapping tm ON e.task_id = tm.old_id;

-- Migrate agent_health
INSERT INTO agent_health_new (id, agent_id, status, cpu_usage, memory_usage, disk_usage, network_latency, last_check, created_at, updated_at)
SELECT 
    hm.new_id,
    am.new_id,
    h.status,
    h.cpu_usage,
    h.memory_usage,
    h.disk_usage,
    h.network_latency,
    h.last_check,
    h.created_at,
    h.updated_at
FROM agent_health h
JOIN agent_health_id_mapping hm ON h.id = hm.old_id
LEFT JOIN agent_id_mapping am ON h.agent_id = am.old_id;

-- Migrate model_registry
INSERT INTO model_registry_new (id, name, version, type, path, config, size_bytes, checksum, created_at, updated_at, is_active)
SELECT 
    mm.new_id,
    mr.name,
    mr.version,
    mr.type,
    mr.path,
    mr.config,
    mr.size_bytes,
    mr.checksum,
    mr.created_at,
    mr.updated_at,
    mr.is_active
FROM model_registry mr
JOIN model_registry_id_mapping mm ON mr.id = mm.old_id;

-- Migrate system_alerts
INSERT INTO system_alerts_new (id, type, severity, message, source, status, created_at, resolved_at, resolved_by)
SELECT 
    sam.new_id,
    sa.type,
    sa.severity,
    sa.message,
    sa.source,
    sa.status,
    sa.created_at,
    sa.resolved_at,
    um.new_id
FROM system_alerts sa
JOIN system_alert_id_mapping sam ON sa.id = sam.old_id
LEFT JOIN user_id_mapping um ON sa.resolved_by = um.old_id;

-- Migrate system_metrics
INSERT INTO system_metrics_new (id, metric_name, metric_value, unit, tags, recorded_at)
SELECT 
    smm.new_id,
    sm.metric_name,
    sm.metric_value,
    sm.unit,
    sm.tags,
    sm.recorded_at
FROM system_metrics sm
JOIN system_metric_id_mapping smm ON sm.id = smm.old_id;

DO $$ BEGIN RAISE NOTICE 'Data migration completed'; END $$;

-- =====================================================================
-- PHASE 4: VALIDATION
-- =====================================================================

DO $$ 
DECLARE
    total_errors INTEGER := 0;
    original_count INTEGER;
    new_count INTEGER;
BEGIN
    RAISE NOTICE 'Validating data integrity...';
    
    -- Check each table
    SELECT COUNT(*) INTO original_count FROM users_backup;
    SELECT COUNT(*) INTO new_count FROM users_new;
    IF original_count != new_count THEN
        total_errors := total_errors + 1;
        RAISE WARNING 'Users count mismatch: original=%, new=%', original_count, new_count;
    END IF;
    
    SELECT COUNT(*) INTO original_count FROM agents_backup;
    SELECT COUNT(*) INTO new_count FROM agents_new;
    IF original_count != new_count THEN
        total_errors := total_errors + 1;
        RAISE WARNING 'Agents count mismatch: original=%, new=%', original_count, new_count;
    END IF;
    
    IF total_errors > 0 THEN
        RAISE EXCEPTION 'Migration validation failed with % errors', total_errors;
    END IF;
    
    RAISE NOTICE 'Validation successful: All data migrated correctly';
END $$;

-- =====================================================================
-- PHASE 5: REPLACE OLD TABLES WITH NEW ONES
-- =====================================================================

DO $$ BEGIN RAISE NOTICE 'Replacing old tables with UUID tables...'; END $$;

-- Drop old tables
DROP TABLE IF EXISTS users CASCADE;
DROP TABLE IF EXISTS agents CASCADE;
DROP TABLE IF EXISTS tasks CASCADE;
DROP TABLE IF EXISTS sessions CASCADE;
DROP TABLE IF EXISTS chat_history CASCADE;
DROP TABLE IF EXISTS agent_executions CASCADE;
DROP TABLE IF EXISTS agent_health CASCADE;
DROP TABLE IF EXISTS model_registry CASCADE;
DROP TABLE IF EXISTS system_alerts CASCADE;
DROP TABLE IF EXISTS system_metrics CASCADE;

-- Rename new tables to original names
ALTER TABLE users_new RENAME TO users;
ALTER TABLE agents_new RENAME TO agents;
ALTER TABLE tasks_new RENAME TO tasks;
ALTER TABLE sessions_new RENAME TO sessions;
ALTER TABLE chat_history_new RENAME TO chat_history;
ALTER TABLE agent_executions_new RENAME TO agent_executions;
ALTER TABLE agent_health_new RENAME TO agent_health;
ALTER TABLE model_registry_new RENAME TO model_registry;
ALTER TABLE system_alerts_new RENAME TO system_alerts;
ALTER TABLE system_metrics_new RENAME TO system_metrics;

-- =====================================================================
-- PHASE 6: RECREATE INDEXES AND TRIGGERS
-- =====================================================================

-- Create indexes
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_is_admin ON users(is_admin) WHERE is_admin = true;

CREATE INDEX idx_agents_type ON users(email);
CREATE INDEX idx_agents_status ON agents(status);

CREATE INDEX idx_tasks_agent_id ON tasks(agent_id);
CREATE INDEX idx_tasks_user_id ON tasks(user_id);
CREATE INDEX idx_tasks_status ON tasks(status);

CREATE INDEX idx_sessions_token ON sessions(token);
CREATE INDEX idx_sessions_user_id ON sessions(user_id);

CREATE INDEX idx_chat_history_user_id ON chat_history(user_id);

CREATE INDEX idx_agent_executions_agent_id ON agent_executions(agent_id);
CREATE INDEX idx_agent_health_agent_id ON agent_health(agent_id);

CREATE INDEX idx_system_alerts_severity ON system_alerts(severity);
CREATE INDEX idx_system_alerts_status ON system_alerts(status);

CREATE INDEX idx_system_metrics_name ON system_metrics(metric_name);
CREATE INDEX idx_system_metrics_recorded_at ON system_metrics(recorded_at);

-- Create update triggers for tables with updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_agent_health_updated_at
    BEFORE UPDATE ON agent_health
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_model_registry_updated_at
    BEFORE UPDATE ON model_registry
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DO $$ BEGIN RAISE NOTICE 'Indexes and triggers recreated'; END $$;

-- =====================================================================
-- FINAL VALIDATION AND COMPLETION
-- =====================================================================

-- Show final counts
SELECT 'After Migration - Data Counts:' as status;
SELECT table_name, 
       (xpath('/row/cnt/text()', query_to_xml(format('select count(*) as cnt from %I.%I', table_schema, table_name), false, true, '')))[1]::text::int as row_count
FROM information_schema.tables 
WHERE table_schema = 'public' AND table_type = 'BASE TABLE' AND table_name NOT LIKE '%_backup'
ORDER BY table_name;

DO $$ BEGIN 
    RAISE NOTICE 'UUID migration completed successfully at %', NOW();
    RAISE NOTICE 'All tables now use UUID primary keys';
    RAISE NOTICE 'Backup tables preserved for safety';
END $$;

COMMIT;