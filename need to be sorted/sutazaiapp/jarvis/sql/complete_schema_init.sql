-- SutazAI Complete Database Schema - Missing Tables and Enhancements
-- This script adds missing tables and improvements to the existing schema

-- Create sessions table (missing from current schema)
CREATE TABLE IF NOT EXISTS sessions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    token VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    is_active BOOLEAN DEFAULT true,
    user_agent TEXT,
    ip_address INET,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for sessions table
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(token);
CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at);
CREATE INDEX IF NOT EXISTS idx_sessions_active ON sessions(is_active);

-- Create agent_health table for monitoring
CREATE TABLE IF NOT EXISTS agent_health (
    id SERIAL PRIMARY KEY,
    agent_id INTEGER REFERENCES agents(id) ON DELETE CASCADE,
    status VARCHAR(50) NOT NULL DEFAULT 'unknown',
    last_heartbeat TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    cpu_usage DECIMAL(5,2) DEFAULT 0.00,
    memory_usage DECIMAL(5,2) DEFAULT 0.00,
    disk_usage DECIMAL(5,2) DEFAULT 0.00,
    response_time DECIMAL(8,3) DEFAULT 0.000,
    error_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for agent_health table
CREATE INDEX IF NOT EXISTS idx_agent_health_agent_id ON agent_health(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_health_status ON agent_health(status);
CREATE INDEX IF NOT EXISTS idx_agent_health_heartbeat ON agent_health(last_heartbeat);

-- Create model_registry table for AI model management
CREATE TABLE IF NOT EXISTS model_registry (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(255) UNIQUE NOT NULL,
    model_type VARCHAR(100) NOT NULL,
    size_mb DECIMAL(10,2),
    status VARCHAR(50) DEFAULT 'available',
    ollama_status VARCHAR(50),
    usage_count INTEGER DEFAULT 0,
    last_used TIMESTAMP,
    file_path TEXT,
    parameters JSONB DEFAULT '{}',
    capabilities JSONB DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for model_registry
CREATE INDEX IF NOT EXISTS idx_model_registry_name ON model_registry(model_name);
CREATE INDEX IF NOT EXISTS idx_model_registry_type ON model_registry(model_type);
CREATE INDEX IF NOT EXISTS idx_model_registry_status ON model_registry(status);

-- Create vector_collections table for vector database integration
CREATE TABLE IF NOT EXISTS vector_collections (
    id SERIAL PRIMARY KEY,
    collection_name VARCHAR(255) UNIQUE NOT NULL,
    database_type VARCHAR(50) NOT NULL, -- 'qdrant', 'faiss', 'chromadb'
    dimension INTEGER NOT NULL,
    document_count INTEGER DEFAULT 0,
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create knowledge_documents table for document management
CREATE TABLE IF NOT EXISTS knowledge_documents (
    id SERIAL PRIMARY KEY,
    title VARCHAR(500),
    content_preview TEXT,
    full_content TEXT,
    document_type VARCHAR(100),
    source_path VARCHAR(1000),
    collection_id INTEGER REFERENCES vector_collections(id),
    embedding_status VARCHAR(50) DEFAULT 'pending',
    processed_at TIMESTAMP,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for knowledge_documents
CREATE INDEX IF NOT EXISTS idx_knowledge_documents_collection ON knowledge_documents(collection_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_documents_type ON knowledge_documents(document_type);
CREATE INDEX IF NOT EXISTS idx_knowledge_documents_status ON knowledge_documents(embedding_status);

-- Create orchestration_sessions table for multi-agent coordination
CREATE TABLE IF NOT EXISTS orchestration_sessions (
    id SERIAL PRIMARY KEY,
    session_name VARCHAR(255),
    task_description TEXT NOT NULL,
    agents_involved JSONB NOT NULL DEFAULT '[]',
    strategy VARCHAR(50) DEFAULT 'collaborative',
    status VARCHAR(50) DEFAULT 'pending',
    progress DECIMAL(5,2) DEFAULT 0.00,
    result JSONB,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

-- Create indexes for orchestration_sessions
CREATE INDEX IF NOT EXISTS idx_orchestration_status ON orchestration_sessions(status);
CREATE INDEX IF NOT EXISTS idx_orchestration_created ON orchestration_sessions(created_at);

-- Create api_usage_logs table for monitoring API usage
CREATE TABLE IF NOT EXISTS api_usage_logs (
    id SERIAL PRIMARY KEY,
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL,
    user_id INTEGER REFERENCES users(id),
    agent_id INTEGER REFERENCES agents(id),
    response_code INTEGER,
    response_time DECIMAL(8,3),
    request_size INTEGER,
    response_size INTEGER,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for api_usage_logs
CREATE INDEX IF NOT EXISTS idx_api_usage_endpoint ON api_usage_logs(endpoint);
CREATE INDEX IF NOT EXISTS idx_api_usage_created ON api_usage_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_api_usage_user ON api_usage_logs(user_id);

-- Create system_alerts table for monitoring alerts
CREATE TABLE IF NOT EXISTS system_alerts (
    id SERIAL PRIMARY KEY,
    alert_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL DEFAULT 'medium',
    title VARCHAR(255) NOT NULL,
    description TEXT,
    source VARCHAR(100),
    status VARCHAR(50) DEFAULT 'active',
    resolved_at TIMESTAMP,
    resolved_by INTEGER REFERENCES users(id),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for system_alerts
CREATE INDEX IF NOT EXISTS idx_system_alerts_type ON system_alerts(alert_type);
CREATE INDEX IF NOT EXISTS idx_system_alerts_severity ON system_alerts(severity);
CREATE INDEX IF NOT EXISTS idx_system_alerts_status ON system_alerts(status);
CREATE INDEX IF NOT EXISTS idx_system_alerts_created ON system_alerts(created_at);

-- Create function to update updated_at columns
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at columns
CREATE TRIGGER update_sessions_updated_at
    BEFORE UPDATE ON sessions
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

CREATE TRIGGER update_vector_collections_updated_at
    BEFORE UPDATE ON vector_collections
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_knowledge_documents_updated_at
    BEFORE UPDATE ON knowledge_documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Insert default data

-- Insert default users (admin user for system)
INSERT INTO users (username, email, password_hash, is_active) VALUES 
('admin', 'admin@sutazai.local', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewjyJyM7QK8kL5yC', true),  -- password: admin123
('system', 'system@sutazai.local', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewjyJyM7QK8kL5yC', true)   -- password: admin123
ON CONFLICT (username) DO NOTHING;

-- Update agents table with current running agents
INSERT INTO agents (name, type, description, endpoint, port, is_active, capabilities) VALUES
('ai-agent-orchestrator', 'orchestration', 'Main AI agent orchestrator', 'http://ai-agent-orchestrator:8589', 8589, true, '["orchestration", "coordination", "task_routing"]'),
('multi-agent-coordinator', 'coordination', 'Multi-agent system coordinator', 'http://multi-agent-coordinator:8587', 8587, true, '["coordination", "scheduling", "resource_management"]'),
('resource-arbitration-agent', 'resource', 'Resource allocation and arbitration', 'http://resource-arbitration-agent:8588', 8588, true, '["resource_allocation", "load_balancing"]'),
('task-assignment-coordinator', 'task', 'Task assignment and routing', 'http://task-assignment-coordinator:8551', 8551, true, '["task_assignment", "routing", "scheduling"]'),
('hardware-resource-optimizer', 'optimization', 'Hardware resource optimization', 'http://hardware-resource-optimizer:8002', 8002, true, '["hardware_optimization", "performance_tuning"]'),
('ollama-integration-specialist', 'integration', 'Ollama LLM integration specialist', 'http://ollama-integration-specialist:11015', 11015, true, '["llm_integration", "model_management"]'),
('ai-metrics-exporter', 'monitoring', 'AI system metrics collection', 'http://ai-metrics-exporter:11063', 11063, false, '["metrics", "monitoring", "telemetry"]')
ON CONFLICT (name) DO UPDATE SET
    type = EXCLUDED.type,
    description = EXCLUDED.description,
    endpoint = EXCLUDED.endpoint,
    port = EXCLUDED.port,
    capabilities = EXCLUDED.capabilities;

-- Insert current models from Ollama
INSERT INTO model_registry (model_name, model_type, size_mb, status, ollama_status, parameters) VALUES
('tinyllama', 'llm', 637, 'active', 'loaded', '{"parameters": "1.1B", "quantization": "Q4_0", "context_length": 2048}'),
('gpt-oss', 'llm', 2500, 'available', 'not_loaded', '{"context_length": 4096, "description": "Open source GPT variant"}')
ON CONFLICT (model_name) DO UPDATE SET
    status = EXCLUDED.status,
    ollama_status = EXCLUDED.ollama_status,
    parameters = EXCLUDED.parameters;

-- Insert vector collections for existing databases
INSERT INTO vector_collections (collection_name, database_type, dimension, status) VALUES
('default_qdrant', 'qdrant', 384, 'active'),
('default_faiss', 'faiss', 384, 'active'),
('default_chromadb', 'chromadb', 384, 'degraded')
ON CONFLICT (collection_name) DO UPDATE SET
    status = EXCLUDED.status;

-- Create useful views

-- Agent status overview
CREATE OR REPLACE VIEW agent_status_overview AS
SELECT 
    a.id,
    a.name,
    a.type,
    a.is_active as configured_active,
    ah.status as health_status,
    ah.last_heartbeat,
    ah.cpu_usage,
    ah.memory_usage,
    CASE 
        WHEN ah.last_heartbeat > (CURRENT_TIMESTAMP - INTERVAL '5 minutes') THEN 'HEALTHY'
        WHEN ah.last_heartbeat > (CURRENT_TIMESTAMP - INTERVAL '30 minutes') THEN 'STALE'
        ELSE 'OFFLINE'
    END as connectivity_status
FROM agents a
LEFT JOIN agent_health ah ON a.id = ah.agent_id;

-- System health dashboard view
CREATE OR REPLACE VIEW system_health_dashboard AS
SELECT 
    (SELECT COUNT(*) FROM agents WHERE is_active = true) as total_agents,
    (SELECT COUNT(*) FROM agent_health WHERE status = 'healthy' AND last_heartbeat > (CURRENT_TIMESTAMP - INTERVAL '5 minutes')) as healthy_agents,
    (SELECT COUNT(*) FROM tasks WHERE status = 'pending') as pending_tasks,
    (SELECT COUNT(*) FROM tasks WHERE status = 'running') as running_tasks,
    (SELECT COUNT(*) FROM orchestration_sessions WHERE status = 'active') as active_orchestrations,
    (SELECT COUNT(*) FROM system_alerts WHERE status = 'active') as active_alerts,
    (SELECT COUNT(*) FROM model_registry WHERE status = 'active') as active_models,
    CURRENT_TIMESTAMP as snapshot_time;

-- Recent activity view
CREATE OR REPLACE VIEW recent_activity AS
SELECT 
    'task' as activity_type,
    t.title as description,
    t.status,
    t.created_at,
    u.username as initiated_by,
    a.name as agent_involved
FROM tasks t
LEFT JOIN users u ON t.user_id = u.id
LEFT JOIN agents a ON t.agent_id = a.id
WHERE t.created_at > (CURRENT_TIMESTAMP - INTERVAL '24 hours')

UNION ALL

SELECT 
    'orchestration' as activity_type,
    os.task_description as description,
    os.status,
    os.created_at,
    'system' as initiated_by,
    NULL as agent_involved
FROM orchestration_sessions os
WHERE os.created_at > (CURRENT_TIMESTAMP - INTERVAL '24 hours')

ORDER BY created_at DESC
LIMIT 50;

-- Performance metrics view
CREATE OR REPLACE VIEW performance_metrics AS
SELECT 
    'agent_response_time' as metric_name,
    AVG(ah.response_time) as avg_value,
    MAX(ah.response_time) as max_value,
    MIN(ah.response_time) as min_value,
    COUNT(*) as sample_count
FROM agent_health ah 
WHERE ah.last_heartbeat > (CURRENT_TIMESTAMP - INTERVAL '1 hour')
GROUP BY metric_name

UNION ALL

SELECT 
    'api_response_time' as metric_name,
    AVG(aul.response_time) as avg_value,
    MAX(aul.response_time) as max_value,
    MIN(aul.response_time) as min_value,
    COUNT(*) as sample_count
FROM api_usage_logs aul 
WHERE aul.created_at > (CURRENT_TIMESTAMP - INTERVAL '1 hour')
GROUP BY metric_name;

-- Grant permissions to existing users
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO sutazai;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO sutazai;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO sutazai;

-- Verification queries
\echo '=== Database Schema Verification ==='
\echo 'Tables created:'
SELECT schemaname, tablename 
FROM pg_tables 
WHERE schemaname = 'public' 
ORDER BY tablename;

\echo ''
\echo 'Indexes created:'
SELECT schemaname, tablename, indexname 
FROM pg_indexes 
WHERE schemaname = 'public' 
ORDER BY tablename, indexname;

\echo ''
\echo 'Views created:'
SELECT schemaname, viewname 
FROM pg_views 
WHERE schemaname = 'public' 
ORDER BY viewname;

\echo ''
\echo 'Data verification:'
SELECT 
    'agents' as table_name, 
    COUNT(*) as record_count 
FROM agents

UNION ALL

SELECT 
    'users' as table_name, 
    COUNT(*) as record_count 
FROM users

UNION ALL

SELECT 
    'model_registry' as table_name, 
    COUNT(*) as record_count 
FROM model_registry

ORDER BY table_name;

\echo ''
\echo '=== Database initialization complete! ==='