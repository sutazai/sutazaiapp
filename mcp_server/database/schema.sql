-- SutazAI MCP Server Database Schema
-- Creates tables for MCP server integration with existing SutazAI system

-- Agents management table
CREATE TABLE IF NOT EXISTS agents (
    agent_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_name VARCHAR(255) UNIQUE NOT NULL,
    agent_type VARCHAR(100) NOT NULL,
    status VARCHAR(50) DEFAULT 'inactive',
    capabilities JSONB DEFAULT '[]',
    config JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    last_seen TIMESTAMP,
    health_status VARCHAR(50) DEFAULT 'unknown',
    resource_usage JSONB DEFAULT '{}',
    
    -- Indexes
    UNIQUE(agent_name),
    INDEX idx_agents_type (agent_type),
    INDEX idx_agents_status (status),
    INDEX idx_agents_last_seen (last_seen)
);

-- Agent tasks management
CREATE TABLE IF NOT EXISTS agent_tasks (
    task_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_name VARCHAR(255) NOT NULL,
    task_description TEXT NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    priority VARCHAR(20) DEFAULT 'normal',
    context JSONB DEFAULT '{}',
    result JSONB,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    estimated_duration_seconds INTEGER,
    actual_duration_seconds INTEGER,
    
    -- Constraints
    FOREIGN KEY (agent_name) REFERENCES agents(agent_name) ON DELETE CASCADE,
    CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    CHECK (priority IN ('low', 'normal', 'high', 'urgent')),
    
    -- Indexes
    INDEX idx_agent_tasks_agent (agent_name),
    INDEX idx_agent_tasks_status (status),
    INDEX idx_agent_tasks_created (created_at),
    INDEX idx_agent_tasks_priority (priority)
);

-- Knowledge base documents
CREATE TABLE IF NOT EXISTS knowledge_documents (
    document_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(500),
    content_preview TEXT,
    full_content TEXT,
    document_type VARCHAR(100),
    source_path VARCHAR(1000),
    embedding_model VARCHAR(255),
    collection_name VARCHAR(255),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    processed_at TIMESTAMP,
    embedding_status VARCHAR(50) DEFAULT 'pending',
    
    -- Constraints
    CHECK (embedding_status IN ('pending', 'processing', 'completed', 'failed')),
    
    -- Indexes
    INDEX idx_knowledge_collection (collection_name),
    INDEX idx_knowledge_type (document_type),
    INDEX idx_knowledge_status (embedding_status),
    INDEX idx_knowledge_created (created_at)
);

-- Agent workspaces
CREATE TABLE IF NOT EXISTS agent_workspaces (
    workspace_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_name VARCHAR(255) NOT NULL,
    workspace_path VARCHAR(1000) NOT NULL,
    size_mb DECIMAL(10,2) DEFAULT 0,
    file_count INTEGER DEFAULT 0,
    last_modified TIMESTAMP DEFAULT NOW(),
    backup_status VARCHAR(50) DEFAULT 'none',
    backup_location VARCHAR(1000),
    retention_days INTEGER DEFAULT 30,
    metadata JSONB DEFAULT '{}',
    
    -- Constraints
    FOREIGN KEY (agent_name) REFERENCES agents(agent_name) ON DELETE CASCADE,
    CHECK (backup_status IN ('none', 'pending', 'in_progress', 'completed', 'failed')),
    UNIQUE(agent_name, workspace_path),
    
    -- Indexes
    INDEX idx_workspaces_agent (agent_name),
    INDEX idx_workspaces_modified (last_modified),
    INDEX idx_workspaces_backup (backup_status)
);

-- Multi-agent orchestration sessions
CREATE TABLE IF NOT EXISTS orchestration_sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_description TEXT NOT NULL,
    agents JSONB NOT NULL,
    strategy VARCHAR(50) DEFAULT 'collaborative',
    status VARCHAR(50) DEFAULT 'active',
    progress DECIMAL(5,2) DEFAULT 0.0,
    result JSONB,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    estimated_duration_seconds INTEGER,
    metadata JSONB DEFAULT '{}',
    
    -- Constraints
    CHECK (status IN ('pending', 'active', 'paused', 'completed', 'failed', 'cancelled')),
    CHECK (strategy IN ('sequential', 'parallel', 'hierarchical', 'collaborative')),
    CHECK (progress >= 0.0 AND progress <= 100.0),
    
    -- Indexes
    INDEX idx_orchestration_status (status),
    INDEX idx_orchestration_created (created_at),
    INDEX idx_orchestration_strategy (strategy)
);

-- System metrics history
CREATE TABLE IF NOT EXISTS system_metrics (
    metric_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    metric_type VARCHAR(100) NOT NULL,
    metric_data JSONB NOT NULL,
    timestamp TIMESTAMP DEFAULT NOW(),
    source VARCHAR(100) DEFAULT 'mcp_server',
    
    -- Indexes
    INDEX idx_metrics_type (metric_type),
    INDEX idx_metrics_timestamp (timestamp),
    INDEX idx_metrics_source (source)
);

-- MCP server sessions/connections
CREATE TABLE IF NOT EXISTS mcp_sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_name VARCHAR(255),
    client_version VARCHAR(100),
    connected_at TIMESTAMP DEFAULT NOW(),
    last_activity TIMESTAMP DEFAULT NOW(),
    status VARCHAR(50) DEFAULT 'active',
    request_count INTEGER DEFAULT 0,
    error_count INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}',
    
    -- Constraints
    CHECK (status IN ('active', 'idle', 'disconnected')),
    
    -- Indexes
    INDEX idx_mcp_sessions_status (status),
    INDEX idx_mcp_sessions_activity (last_activity)
);

-- Model management
CREATE TABLE IF NOT EXISTS model_registry (
    model_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(255) UNIQUE NOT NULL,
    model_type VARCHAR(100) NOT NULL,
    size_gb DECIMAL(8,2),
    status VARCHAR(50) DEFAULT 'available',
    ollama_status VARCHAR(50),
    usage_count INTEGER DEFAULT 0,
    last_used TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',
    
    -- Constraints
    CHECK (status IN ('available', 'downloading', 'loading', 'active', 'error', 'removed')),
    
    -- Indexes
    INDEX idx_models_name (model_name),
    INDEX idx_models_type (model_type),
    INDEX idx_models_status (status),
    INDEX idx_models_usage (usage_count DESC)
);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add updated_at triggers
CREATE TRIGGER update_agents_updated_at 
    BEFORE UPDATE ON agents 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_knowledge_documents_updated_at 
    BEFORE UPDATE ON knowledge_documents 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create partitioned table for high-volume metrics
CREATE TABLE IF NOT EXISTS system_metrics_partitioned (
    metric_id UUID DEFAULT gen_random_uuid(),
    metric_type VARCHAR(100) NOT NULL,
    metric_data JSONB NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    source VARCHAR(100) DEFAULT 'mcp_server'
) PARTITION BY RANGE (timestamp);

-- Create partitions for current and next month
CREATE TABLE IF NOT EXISTS system_metrics_y2024m12 PARTITION OF system_metrics_partitioned
    FOR VALUES FROM ('2024-12-01') TO ('2025-01-01');
    
CREATE TABLE IF NOT EXISTS system_metrics_y2025m01 PARTITION OF system_metrics_partitioned
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

-- Views for commonly accessed data
CREATE OR REPLACE VIEW active_agents AS
SELECT 
    agent_name,
    agent_type,
    status,
    capabilities,
    last_seen,
    EXTRACT(EPOCH FROM (NOW() - last_seen))/60 AS minutes_since_last_seen
FROM agents 
WHERE status IN ('active', 'idle') 
ORDER BY last_seen DESC;

CREATE OR REPLACE VIEW agent_task_summary AS
SELECT 
    agent_name,
    COUNT(*) AS total_tasks,
    COUNT(*) FILTER (WHERE status = 'completed') AS completed_tasks,
    COUNT(*) FILTER (WHERE status = 'failed') AS failed_tasks,
    COUNT(*) FILTER (WHERE status = 'running') AS running_tasks,
    AVG(actual_duration_seconds) FILTER (WHERE status = 'completed') AS avg_duration_seconds,
    MAX(created_at) AS last_task_created
FROM agent_tasks 
GROUP BY agent_name;

CREATE OR REPLACE VIEW recent_orchestrations AS
SELECT 
    session_id,
    task_description,
    jsonb_array_length(agents) AS agent_count,
    strategy,
    status,
    progress,
    created_at,
    EXTRACT(EPOCH FROM (NOW() - created_at))/60 AS minutes_ago
FROM orchestration_sessions 
WHERE created_at > NOW() - INTERVAL '7 days'
ORDER BY created_at DESC;

-- Insert some initial data
INSERT INTO agents (agent_name, agent_type, status, capabilities) VALUES
('mcp-coordinator', 'system', 'active', '["coordination", "monitoring", "logging"]'),
('system-monitor', 'system', 'active', '["metrics", "health_checks", "alerts"]')
ON CONFLICT (agent_name) DO NOTHING;

-- Create functions for common operations
CREATE OR REPLACE FUNCTION get_agent_health(agent_name_param VARCHAR)
RETURNS JSONB AS $$
DECLARE
    result JSONB;
BEGIN
    SELECT jsonb_build_object(
        'agent_name', agent_name,
        'status', status,
        'last_seen', last_seen,
        'uptime_minutes', EXTRACT(EPOCH FROM (NOW() - created_at))/60,
        'task_count', (
            SELECT COUNT(*) FROM agent_tasks 
            WHERE agent_tasks.agent_name = agents.agent_name
        ),
        'recent_tasks', (
            SELECT COUNT(*) FROM agent_tasks 
            WHERE agent_tasks.agent_name = agents.agent_name 
            AND created_at > NOW() - INTERVAL '1 hour'
        )
    ) INTO result
    FROM agents 
    WHERE agent_name = agent_name_param;
    
    RETURN COALESCE(result, '{"error": "Agent not found"}'::jsonb);
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION update_agent_health(
    agent_name_param VARCHAR,
    health_data JSONB
) RETURNS BOOLEAN AS $$
BEGIN
    UPDATE agents 
    SET 
        last_seen = NOW(),
        health_status = COALESCE(health_data->>'status', health_status),
        resource_usage = COALESCE(health_data->'resources', resource_usage)
    WHERE agent_name = agent_name_param;
    
    RETURN FOUND;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions (adjust as needed for your user)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO sutazai;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO sutazai;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO sutazai;

-- Add comments for documentation
COMMENT ON TABLE agents IS 'Registry of all AI agents in the SutazAI system';
COMMENT ON TABLE agent_tasks IS 'Task queue and execution history for AI agents';
COMMENT ON TABLE knowledge_documents IS 'Document storage for vector knowledge base';
COMMENT ON TABLE agent_workspaces IS 'Workspace management for agent file persistence';
COMMENT ON TABLE orchestration_sessions IS 'Multi-agent coordination and orchestration sessions';
COMMENT ON TABLE system_metrics IS 'System performance and health metrics';
COMMENT ON TABLE mcp_sessions IS 'MCP server connection and session tracking';
COMMENT ON TABLE model_registry IS 'Registry of available AI models and their status';

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_agent_tasks_composite ON agent_tasks (agent_name, status, created_at);
CREATE INDEX IF NOT EXISTS idx_knowledge_documents_composite ON knowledge_documents (collection_name, embedding_status, created_at);
CREATE INDEX IF NOT EXISTS idx_system_metrics_composite ON system_metrics (metric_type, timestamp DESC);

-- Analytics views
CREATE OR REPLACE VIEW system_overview AS
SELECT 
    (SELECT COUNT(*) FROM agents WHERE status = 'active') AS active_agents,
    (SELECT COUNT(*) FROM agent_tasks WHERE status = 'running') AS running_tasks,
    (SELECT COUNT(*) FROM orchestration_sessions WHERE status = 'active') AS active_orchestrations,
    (SELECT COUNT(*) FROM knowledge_documents WHERE embedding_status = 'completed') AS processed_documents,
    (SELECT COUNT(*) FROM model_registry WHERE status = 'active') AS active_models,
    NOW() AS generated_at;

COMMENT ON VIEW system_overview IS 'High-level system status overview for MCP dashboard'; 