-- PostgreSQL Unified Memory Database Schema
-- Version: 1.0
-- Date: 2025-08-20

-- Drop existing tables if they exist (for clean setup)
DROP TABLE IF EXISTS performance_metrics CASCADE;
DROP TABLE IF EXISTS active_sessions CASCADE;
DROP TABLE IF EXISTS migration_metadata CASCADE;
DROP TABLE IF EXISTS unified_memory CASCADE;

-- Main memory storage table
CREATE TABLE unified_memory (
    id BIGSERIAL PRIMARY KEY,
    key TEXT NOT NULL,
    value JSONB NOT NULL,
    namespace VARCHAR(255) NOT NULL DEFAULT 'default',
    data_type VARCHAR(50) NOT NULL DEFAULT 'json',
    metadata JSONB,
    source_db VARCHAR(255),  -- Track original database
    source_path TEXT,        -- Original file path
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    accessed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    
    -- Access tracking
    access_count INTEGER DEFAULT 0,
    ttl INTEGER,
    
    -- Constraint
    UNIQUE(key, namespace)
);

-- Indexes for performance
CREATE INDEX idx_unified_memory_namespace ON unified_memory(namespace);
CREATE INDEX idx_unified_memory_expires ON unified_memory(expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX idx_unified_memory_accessed ON unified_memory(accessed_at);
CREATE INDEX idx_unified_memory_source ON unified_memory(source_db);
CREATE INDEX idx_unified_memory_data_type ON unified_memory(data_type);
CREATE INDEX idx_unified_memory_value_gin ON unified_memory USING gin(value);
CREATE INDEX idx_unified_memory_created ON unified_memory(created_at DESC);
CREATE INDEX idx_unified_memory_updated ON unified_memory(updated_at DESC);

-- Metadata tracking table
CREATE TABLE migration_metadata (
    id SERIAL PRIMARY KEY,
    source_file TEXT NOT NULL,
    records_migrated INTEGER NOT NULL,
    migration_started TIMESTAMP WITH TIME ZONE NOT NULL,
    migration_completed TIMESTAMP WITH TIME ZONE,
    status VARCHAR(50) NOT NULL,
    error_message TEXT,
    checksum VARCHAR(64)
);

-- Session tracking (for active sessions)
CREATE TABLE active_sessions (
    session_id VARCHAR(255) PRIMARY KEY,
    agent_type VARCHAR(100),
    topology VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB
);

-- Performance metrics table
CREATE TABLE performance_metrics (
    id BIGSERIAL PRIMARY KEY,
    metric_name VARCHAR(255) NOT NULL,
    metric_value NUMERIC,
    metric_data JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    namespace VARCHAR(255) DEFAULT 'default'
);

-- Indexes for performance metrics
CREATE INDEX idx_metrics_name ON performance_metrics(metric_name);
CREATE INDEX idx_metrics_timestamp ON performance_metrics(timestamp DESC);
CREATE INDEX idx_metrics_namespace ON performance_metrics(namespace);

-- Create update trigger for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    NEW.access_count = OLD.access_count + 1;
    NEW.accessed_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_unified_memory_updated_at BEFORE UPDATE
    ON unified_memory FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create function to clean expired entries
CREATE OR REPLACE FUNCTION clean_expired_entries()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM unified_memory 
    WHERE expires_at IS NOT NULL AND expires_at < NOW();
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions (adjust as needed)
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO sutazai;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO sutazai;