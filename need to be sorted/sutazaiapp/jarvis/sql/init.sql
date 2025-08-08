-- Hygiene Monitoring Database Initialization
-- This script creates the necessary tables and indexes for the monitoring system

-- Create database (this may already exist from the POSTGRES_DB env var)
-- CREATE DATABASE IF NOT EXISTS hygiene_monitoring;

-- Connect to the database
\c hygiene_monitoring;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- System metrics table
CREATE TABLE IF NOT EXISTS system_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    cpu_usage REAL,
    memory_percentage REAL,
    memory_used_gb REAL,
    memory_total_gb REAL,
    disk_percentage REAL,
    disk_used_gb REAL,
    disk_total_gb REAL,
    network_status TEXT DEFAULT 'HEALTHY',
    load_avg REAL[] DEFAULT ARRAY[0,0,0],
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for system_metrics
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON system_metrics(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_metrics_created_at ON system_metrics(created_at DESC);

-- Violations table
CREATE TABLE IF NOT EXISTS violations (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    rule_id TEXT NOT NULL,
    rule_name TEXT,
    file_path TEXT,
    severity TEXT NOT NULL,
    description TEXT,
    pattern_matched TEXT,
    status TEXT DEFAULT 'ACTIVE',
    resolved_at TIMESTAMPTZ NULL,
    resolver_id TEXT NULL,
    resolution_notes TEXT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for violations
CREATE INDEX IF NOT EXISTS idx_violations_timestamp ON violations(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_violations_rule_id ON violations(rule_id);
CREATE INDEX IF NOT EXISTS idx_violations_severity ON violations(severity);
CREATE INDEX IF NOT EXISTS idx_violations_status ON violations(status);
CREATE INDEX IF NOT EXISTS idx_violations_file_path ON violations(file_path);

-- Agent health table
CREATE TABLE IF NOT EXISTS agent_health (
    id SERIAL PRIMARY KEY,
    agent_id TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'ACTIVE',
    last_heartbeat TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tasks_completed INTEGER DEFAULT 0,
    tasks_failed INTEGER DEFAULT 0,
    cpu_usage REAL DEFAULT 0.0,
    memory_usage REAL DEFAULT 0.0,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for agent_health
CREATE INDEX IF NOT EXISTS idx_agent_health_agent_id ON agent_health(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_health_status ON agent_health(status);
CREATE INDEX IF NOT EXISTS idx_agent_health_last_heartbeat ON agent_health(last_heartbeat DESC);

-- Actions/events table for audit trail
CREATE TABLE IF NOT EXISTS actions (
    id SERIAL PRIMARY KEY,
    action_id UUID DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    action_type TEXT NOT NULL, -- SCAN, ENFORCE, RESOLVE, etc.
    agent_id TEXT,
    rule_id TEXT,
    file_path TEXT,
    status TEXT NOT NULL, -- STARTED, COMPLETED, FAILED
    details JSONB DEFAULT '{}'::jsonb,
    duration_ms INTEGER,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for actions
CREATE INDEX IF NOT EXISTS idx_actions_timestamp ON actions(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_actions_action_type ON actions(action_type);
CREATE INDEX IF NOT EXISTS idx_actions_agent_id ON actions(agent_id);
CREATE INDEX IF NOT EXISTS idx_actions_status ON actions(status);

-- Rule configurations table
CREATE TABLE IF NOT EXISTS rule_configurations (
    id SERIAL PRIMARY KEY,
    rule_id TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    category TEXT,
    severity TEXT NOT NULL,
    enabled BOOLEAN DEFAULT TRUE,
    configuration JSONB DEFAULT '{}'::jsonb,
    dependencies TEXT[] DEFAULT ARRAY[]::TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for rule_configurations
CREATE INDEX IF NOT EXISTS idx_rule_configurations_rule_id ON rule_configurations(rule_id);
CREATE INDEX IF NOT EXISTS idx_rule_configurations_enabled ON rule_configurations(enabled);
CREATE INDEX IF NOT EXISTS idx_rule_configurations_severity ON rule_configurations(severity);

-- Views for common queries
CREATE OR REPLACE VIEW recent_violations AS
SELECT 
    v.*,
    rc.name as rule_display_name,
    rc.category
FROM violations v
LEFT JOIN rule_configurations rc ON v.rule_id = rc.rule_id
WHERE v.timestamp > NOW() - INTERVAL '24 hours'
ORDER BY v.timestamp DESC;

CREATE OR REPLACE VIEW violation_summary AS
SELECT 
    rule_id,
    rule_name,
    severity,
    COUNT(*) as violation_count,
    MAX(timestamp) as last_violation,
    COUNT(*) FILTER (WHERE status = 'ACTIVE') as active_count,
    COUNT(*) FILTER (WHERE status = 'RESOLVED') as resolved_count
FROM violations
WHERE timestamp > NOW() - INTERVAL '7 days'
GROUP BY rule_id, rule_name, severity
ORDER BY violation_count DESC;

CREATE OR REPLACE VIEW system_health_summary AS
SELECT 
    DATE_TRUNC('hour', timestamp) as hour,
    AVG(cpu_usage) as avg_cpu,
    AVG(memory_percentage) as avg_memory,
    AVG(disk_percentage) as avg_disk,
    MAX(cpu_usage) as max_cpu,
    MAX(memory_percentage) as max_memory,
    COUNT(*) as measurement_count
FROM system_metrics
WHERE timestamp > NOW() - INTERVAL '24 hours'
GROUP BY DATE_TRUNC('hour', timestamp)
ORDER BY hour DESC;

-- Function to update updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_violations_updated_at
    BEFORE UPDATE ON violations
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_agent_health_updated_at
    BEFORE UPDATE ON agent_health
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_rule_configurations_updated_at
    BEFORE UPDATE ON rule_configurations
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Insert default rule configurations
INSERT INTO rule_configurations (rule_id, name, description, category, severity, enabled) VALUES
('no_fantasy_elements', 'No Fantasy Elements', 'Only real, production-ready implementations allowed', 'Code Quality', 'CRITICAL', true),
('no_breaking_changes', 'No Breaking Changes', 'Every change must respect existing functionality', 'Functionality', 'CRITICAL', true),
('analyze_everything', 'Analyze Everything', 'Systematic review before proceeding', 'Process', 'HIGH', true),
('reuse_before_creating', 'Reuse Before Creating', 'Check for existing solutions first', 'Efficiency', 'MEDIUM', true),
('professional_standards', 'Professional Standards', 'Maintain professional code quality', 'Quality', 'HIGH', true),
('centralized_documentation', 'Centralized Documentation', 'Organized and consistent documentation', 'Documentation', 'HIGH', true),
('script_organization', 'Script Organization', 'Centralized and documented scripts', 'Scripts', 'MEDIUM', true),
('python_script_standards', 'Python Script Standards', 'Structured and purposeful Python code', 'Scripts', 'MEDIUM', true),
('no_version_duplication', 'No Version Duplication', 'Single source of truth for code', 'Structure', 'HIGH', true),
('functionality_first_cleanup', 'Functionality First Cleanup', 'Verify before removing code', 'Process', 'CRITICAL', true),
('clean_docker_structure', 'Clean Docker Structure', 'Modular and predictable containers', 'Infrastructure', 'MEDIUM', true),
('single_deployment_script', 'Single Deployment Script', 'One canonical deployment process', 'Deployment', 'HIGH', true),
('no_garbage_no_rot', 'No Garbage, No Rot', 'Zero tolerance for abandoned code', 'Quality', 'HIGH', true),
('correct_ai_agent', 'Correct AI Agent Usage', 'Route tasks to specialized agents', 'Process', 'MEDIUM', true),
('clean_documentation', 'Clean Documentation', 'Documentation as critical as code', 'Documentation', 'MEDIUM', true),
('local_llm_ollama', 'Local LLM via Ollama', 'Standardized LLM framework usage', 'Security', 'LOW', true)
ON CONFLICT (rule_id) DO NOTHING;

-- Create sample data for development/testing
DO $$
BEGIN
    -- Insert sample system metrics (last hour)
    FOR i IN 1..60 LOOP
        INSERT INTO system_metrics (
            timestamp,
            cpu_usage,
            memory_percentage,
            disk_percentage,
            network_status
        ) VALUES (
            NOW() - (i || ' minutes')::interval,
            RANDOM() * 100,
            20 + RANDOM() * 60,
            40 + RANDOM() * 20,
            CASE WHEN RANDOM() > 0.9 THEN 'DEGRADED' ELSE 'HEALTHY' END
        );
    END LOOP;

    -- Insert sample agent health data
    INSERT INTO agent_health (agent_id, name, status, tasks_completed, tasks_failed, cpu_usage, memory_usage) VALUES
    ('hygiene-scanner', 'Hygiene Scanner', 'ACTIVE', 150, 2, 5.2, 45.8),
    ('rule-enforcer', 'Rule Enforcer', 'ACTIVE', 89, 1, 3.1, 32.4),
    ('cleanup-agent', 'Cleanup Agent', 'IDLE', 45, 0, 1.2, 28.9)
    ON CONFLICT (agent_id) DO UPDATE SET
        name = EXCLUDED.name,
        status = EXCLUDED.status,
        tasks_completed = EXCLUDED.tasks_completed,
        tasks_failed = EXCLUDED.tasks_failed,
        cpu_usage = EXCLUDED.cpu_usage,
        memory_usage = EXCLUDED.memory_usage,
        last_heartbeat = NOW();

END $$;

-- Create a user for the application (if not exists)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'hygiene_app') THEN
        CREATE ROLE hygiene_app WITH LOGIN PASSWORD 'app_secure_password_2024';
    END IF;
END $$;

-- Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO hygiene_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO hygiene_app;

-- Success message
\echo 'Database initialization completed successfully!'
\echo 'Tables created: system_metrics, violations, agent_health, actions, rule_configurations'
\echo 'Views created: recent_violations, violation_summary, system_health_summary'
\echo 'Sample data inserted for development/testing'