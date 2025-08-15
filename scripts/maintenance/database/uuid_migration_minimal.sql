-- =====================================================================
--   SAFE UUID MIGRATION: INTEGER to UUID Primary Keys
-- =====================================================================
-- Author: Claude Code (Ultra Database Migration Specialist) 
-- Date: 2025-08-10
-- Strategy: Create new UUID tables with exact schema match
-- =====================================================================

BEGIN;

-- Enable UUID extensions
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

DO $$ BEGIN RAISE NOTICE 'Starting   UUID migration at %', NOW(); END $$;

-- =====================================================================
-- BACKUP AND MIGRATE: USERS TABLE
-- =====================================================================

CREATE TABLE users_backup AS SELECT * FROM users;

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

CREATE TEMP TABLE user_id_mapping AS
SELECT id as old_id, gen_random_uuid() as new_id FROM users;

INSERT INTO users_new
SELECT m.new_id, u.username, u.email, u.password_hash, u.is_active, 
       u.created_at, u.updated_at, u.is_admin, u.last_login, 
       u.failed_login_attempts, u.locked_until
FROM users u JOIN user_id_mapping m ON u.id = m.old_id;

-- =====================================================================
-- BACKUP AND MIGRATE: AGENTS TABLE
-- =====================================================================

CREATE TABLE agents_backup AS SELECT * FROM agents;

CREATE TABLE agents_new (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) UNIQUE NOT NULL,
    type VARCHAR(50) NOT NULL,
    description TEXT,
    endpoint VARCHAR(255) NOT NULL,
    port INTEGER,
    is_active BOOLEAN DEFAULT true,
    capabilities JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TEMP TABLE agent_id_mapping AS
SELECT id as old_id, gen_random_uuid() as new_id FROM agents;

INSERT INTO agents_new
SELECT m.new_id, a.name, a.type, a.description, a.endpoint, 
       a.port, a.is_active, a.capabilities, a.created_at
FROM agents a JOIN agent_id_mapping m ON a.id = m.old_id;

-- =====================================================================
-- BACKUP AND MIGRATE: TASKS TABLE (has foreign keys)
-- =====================================================================

CREATE TABLE tasks_backup AS SELECT * FROM tasks;

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

CREATE TEMP TABLE task_id_mapping AS
SELECT id as old_id, gen_random_uuid() as new_id FROM tasks;

INSERT INTO tasks_new
SELECT m.new_id, t.title, t.description, 
       am.new_id, um.new_id, t.status, t.priority,
       t.payload, t.result, t.error_message,
       t.created_at, t.started_at, t.completed_at
FROM tasks t 
JOIN task_id_mapping m ON t.id = m.old_id
LEFT JOIN agent_id_mapping am ON t.agent_id = am.old_id
LEFT JOIN user_id_mapping um ON t.user_id = um.old_id;

-- =====================================================================
-- BACKUP AND MIGRATE: SESSIONS TABLE
-- =====================================================================

CREATE TABLE sessions_backup AS SELECT * FROM sessions;

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

CREATE TEMP TABLE session_id_mapping AS
SELECT id as old_id, gen_random_uuid() as new_id FROM sessions;

INSERT INTO sessions_new
SELECT m.new_id, um.new_id, s.token, s.expires_at, s.is_active,
       s.user_agent, s.ip_address, s.created_at, s.last_accessed
FROM sessions s
JOIN session_id_mapping m ON s.id = m.old_id
LEFT JOIN user_id_mapping um ON s.user_id = um.old_id;

-- =====================================================================
-- BACKUP AND MIGRATE: CHAT_HISTORY TABLE
-- =====================================================================

CREATE TABLE chat_history_backup AS SELECT * FROM chat_history;

CREATE TABLE chat_history_new (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users_new(id),
    message TEXT NOT NULL,
    response TEXT,
    model_used VARCHAR(100),
    tokens_used INTEGER,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TEMP TABLE chat_id_mapping AS
SELECT id as old_id, gen_random_uuid() as new_id FROM chat_history;

INSERT INTO chat_history_new
SELECT m.new_id, um.new_id, c.message, c.response, c.model_used,
       c.tokens_used, c.created_at
FROM chat_history c
JOIN chat_id_mapping m ON c.id = m.old_id
LEFT JOIN user_id_mapping um ON c.user_id = um.old_id;

-- =====================================================================
-- BACKUP AND MIGRATE: AGENT_EXECUTIONS TABLE
-- =====================================================================

CREATE TABLE agent_executions_backup AS SELECT * FROM agent_executions;

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

CREATE TEMP TABLE exec_id_mapping AS
SELECT id as old_id, gen_random_uuid() as new_id FROM agent_executions;

INSERT INTO agent_executions_new
SELECT m.new_id, am.new_id, tm.new_id, e.status, e.input_data,
       e.output_data, e.error_message, e.execution_time,
       e.started_at, e.completed_at, e.created_at
FROM agent_executions e
JOIN exec_id_mapping m ON e.id = m.old_id
LEFT JOIN agent_id_mapping am ON e.agent_id = am.old_id
LEFT JOIN task_id_mapping tm ON e.task_id = tm.old_id;

-- =====================================================================
-- BACKUP AND MIGRATE: AGENT_HEALTH TABLE
-- =====================================================================

CREATE TABLE agent_health_backup AS SELECT * FROM agent_health;

CREATE TABLE agent_health_new (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID REFERENCES agents_new(id) ON DELETE CASCADE,
    status VARCHAR(50) NOT NULL,
    cpu_usage DOUBLE PRECISION,
    memory_usage DOUBLE PRECISION,
    disk_usage DOUBLE PRECISION,
    network_latency DOUBLE PRECISION,
    last_check TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TEMP TABLE health_id_mapping AS
SELECT id as old_id, gen_random_uuid() as new_id FROM agent_health;

INSERT INTO agent_health_new
SELECT m.new_id, am.new_id, h.status, h.cpu_usage, h.memory_usage,
       h.disk_usage, h.network_latency, h.last_check, h.created_at, h.updated_at
FROM agent_health h
JOIN health_id_mapping m ON h.id = m.old_id
LEFT JOIN agent_id_mapping am ON h.agent_id = am.old_id;

-- =====================================================================
-- BACKUP AND MIGRATE: MODEL_REGISTRY TABLE
-- =====================================================================

CREATE TABLE model_registry_backup AS SELECT * FROM model_registry;

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

CREATE TEMP TABLE model_id_mapping AS
SELECT id as old_id, gen_random_uuid() as new_id FROM model_registry;

INSERT INTO model_registry_new
SELECT m.new_id, mr.name, mr.version, mr.type, mr.path, mr.config,
       mr.size_bytes, mr.checksum, mr.created_at, mr.updated_at, mr.is_active
FROM model_registry mr
JOIN model_id_mapping m ON mr.id = m.old_id;

-- =====================================================================
-- BACKUP AND MIGRATE: SYSTEM_ALERTS TABLE
-- =====================================================================

CREATE TABLE system_alerts_backup AS SELECT * FROM system_alerts;

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

CREATE TEMP TABLE alert_id_mapping AS
SELECT id as old_id, gen_random_uuid() as new_id FROM system_alerts;

INSERT INTO system_alerts_new
SELECT m.new_id, sa.type, sa.severity, sa.message, sa.source, sa.status,
       sa.created_at, sa.resolved_at, um.new_id
FROM system_alerts sa
JOIN alert_id_mapping m ON sa.id = m.old_id
LEFT JOIN user_id_mapping um ON sa.resolved_by = um.old_id;

-- =====================================================================
-- BACKUP AND MIGRATE: SYSTEM_METRICS TABLE
-- =====================================================================

CREATE TABLE system_metrics_backup AS SELECT * FROM system_metrics;

CREATE TABLE system_metrics_new (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    metric_name VARCHAR(100) NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    unit VARCHAR(20),
    tags JSON DEFAULT '{}'::json,
    recorded_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TEMP TABLE metric_id_mapping AS
SELECT id as old_id, gen_random_uuid() as new_id FROM system_metrics;

INSERT INTO system_metrics_new
SELECT m.new_id, sm.metric_name, sm.metric_value, sm.unit, sm.tags, sm.recorded_at
FROM system_metrics sm
JOIN metric_id_mapping m ON sm.id = m.old_id;

-- =====================================================================
-- VALIDATION
-- =====================================================================

DO $$ 
DECLARE
    total_errors INTEGER := 0;
BEGIN
    IF (SELECT COUNT(*) FROM users_backup) != (SELECT COUNT(*) FROM users_new) THEN
        total_errors := total_errors + 1;
        RAISE WARNING 'Users count mismatch';
    END IF;
    
    IF (SELECT COUNT(*) FROM agents_backup) != (SELECT COUNT(*) FROM agents_new) THEN
        total_errors := total_errors + 1;
        RAISE WARNING 'Agents count mismatch';
    END IF;
    
    IF total_errors > 0 THEN
        RAISE EXCEPTION 'Migration validation failed with % errors', total_errors;
    END IF;
    
    RAISE NOTICE 'Migration validation successful';
END $$;

-- =====================================================================
-- REPLACE OLD TABLES
-- =====================================================================

DO $$ BEGIN RAISE NOTICE 'Replacing old tables with UUID versions...'; END $$;

-- Drop old tables (CASCADE handles foreign keys)
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

-- Rename new tables
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
-- RECREATE INDEXES AND TRIGGERS
-- =====================================================================

-- Indexes
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_is_admin ON users(is_admin) WHERE is_admin = true;
CREATE INDEX idx_agents_type ON agents(type);
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

-- Triggers for updated_at columns
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_agent_health_updated_at
    BEFORE UPDATE ON agent_health FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_model_registry_updated_at
    BEFORE UPDATE ON model_registry FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Note: agents table doesn't have updated_at in the actual schema

DO $$ BEGIN 
    RAISE NOTICE 'UUID migration completed successfully at %', NOW();
    RAISE NOTICE 'All tables now use UUID primary keys';
    RAISE NOTICE 'Original data preserved in *_backup tables';
END $$;

COMMIT;