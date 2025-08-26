# JARVIS Multi-Agent System Database Schema

## Overview
Complete database schema design for the JARVIS multi-agent AI system, supporting PostgreSQL (primary), Redis (cache), Neo4j (graph), ChromaDB (vectors), and Qdrant (semantic search).

## PostgreSQL Schema (Port 10000)

### Core Tables

```sql
-- Agent Registry
CREATE TABLE agents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) UNIQUE NOT NULL,
    type VARCHAR(100) NOT NULL, -- 'letta', 'autogpt', 'crewai', etc.
    status VARCHAR(50) DEFAULT 'inactive', -- active, inactive, error, updating
    capabilities JSONB NOT NULL DEFAULT '[]',
    configuration JSONB NOT NULL DEFAULT '{}',
    resources JSONB DEFAULT '{}', -- cpu, memory limits
    port INTEGER UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_heartbeat TIMESTAMP,
    INDEX idx_agent_status (status),
    INDEX idx_agent_type (type)
);

-- Task Management
CREATE TABLE tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    type VARCHAR(100) NOT NULL, -- 'voice', 'chat', 'autonomous', 'scheduled'
    status VARCHAR(50) DEFAULT 'pending', -- pending, running, completed, failed, cancelled
    priority INTEGER DEFAULT 5, -- 1-10
    agent_id UUID REFERENCES agents(id),
    parent_task_id UUID REFERENCES tasks(id),
    input_data JSONB,
    output_data JSONB,
    error_details JSONB,
    execution_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    INDEX idx_task_status (status),
    INDEX idx_task_agent (agent_id),
    INDEX idx_task_priority (priority DESC)
);

-- Conversation History
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255),
    message_type VARCHAR(50) NOT NULL, -- 'voice', 'text', 'system'
    role VARCHAR(50) NOT NULL, -- 'user', 'assistant', 'system'
    content TEXT NOT NULL,
    audio_file_path TEXT,
    agent_id UUID REFERENCES agents(id),
    task_id UUID REFERENCES tasks(id),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_conv_session (session_id),
    INDEX idx_conv_user (user_id),
    INDEX idx_conv_created (created_at DESC)
);

-- Model Registry
CREATE TABLE models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) UNIQUE NOT NULL,
    type VARCHAR(100) NOT NULL, -- 'llm', 'embedding', 'whisper', 'tts'
    provider VARCHAR(100) NOT NULL, -- 'ollama', 'local', 'transformers'
    size_mb INTEGER,
    parameters JSONB DEFAULT '{}',
    performance_metrics JSONB DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'available', -- available, downloading, error
    path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_used TIMESTAMP,
    usage_count INTEGER DEFAULT 0,
    INDEX idx_model_type (type),
    INDEX idx_model_status (status)
);

-- Agent Collaboration
CREATE TABLE agent_collaborations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID REFERENCES tasks(id),
    coordinator_agent_id UUID REFERENCES agents(id),
    participant_agent_ids UUID[] NOT NULL,
    collaboration_type VARCHAR(100), -- 'sequential', 'parallel', 'hierarchical'
    workflow JSONB NOT NULL,
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    INDEX idx_collab_task (task_id),
    INDEX idx_collab_coordinator (coordinator_agent_id)
);

-- System Events
CREATE TABLE system_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type VARCHAR(100) NOT NULL, -- 'agent_start', 'task_complete', 'error', 'warning'
    severity VARCHAR(50) DEFAULT 'info', -- debug, info, warning, error, critical
    source VARCHAR(255) NOT NULL,
    agent_id UUID REFERENCES agents(id),
    task_id UUID REFERENCES tasks(id),
    message TEXT NOT NULL,
    details JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_event_type (event_type),
    INDEX idx_event_severity (severity),
    INDEX idx_event_created (created_at DESC)
);

-- Performance Metrics
CREATE TABLE performance_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    metric_type VARCHAR(100) NOT NULL, -- 'latency', 'throughput', 'accuracy', 'resource'
    source VARCHAR(255) NOT NULL,
    agent_id UUID REFERENCES agents(id),
    value DECIMAL(10, 4) NOT NULL,
    unit VARCHAR(50),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_metric_type (metric_type),
    INDEX idx_metric_agent (agent_id),
    INDEX idx_metric_created (created_at DESC)
);

-- Self-Improvement Logs
CREATE TABLE improvement_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    improvement_type VARCHAR(100) NOT NULL, -- 'code_generation', 'optimization', 'bug_fix'
    target VARCHAR(255) NOT NULL,
    description TEXT,
    before_state JSONB,
    after_state JSONB,
    performance_gain DECIMAL(5, 2),
    agent_id UUID REFERENCES agents(id),
    approved BOOLEAN DEFAULT FALSE,
    applied BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    applied_at TIMESTAMP,
    INDEX idx_improve_type (improvement_type),
    INDEX idx_improve_applied (applied)
);

-- Memory Store (for Letta/MemGPT)
CREATE TABLE agent_memory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID REFERENCES agents(id) NOT NULL,
    memory_type VARCHAR(100) NOT NULL, -- 'core', 'recall', 'archival', 'working'
    key VARCHAR(255) NOT NULL,
    value TEXT NOT NULL,
    embedding VECTOR(384), -- For semantic search
    importance DECIMAL(3, 2) DEFAULT 0.5, -- 0-1
    access_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP,
    expires_at TIMESTAMP,
    INDEX idx_memory_agent (agent_id),
    INDEX idx_memory_type (memory_type),
    INDEX idx_memory_key (key),
    INDEX idx_memory_importance (importance DESC),
    UNIQUE(agent_id, memory_type, key)
);
```

## Redis Schema (Port 10001)

### Key Patterns

```redis
# Session Management
session:{session_id} = {
    "user_id": "string",
    "agent_id": "uuid",
    "context": {},
    "created_at": "timestamp",
    "last_activity": "timestamp"
}
TTL: 3600 seconds

# Agent Status Cache
agent:status:{agent_id} = {
    "status": "active|inactive|error",
    "last_heartbeat": "timestamp",
    "current_task": "task_id",
    "resource_usage": {}
}
TTL: 30 seconds

# Task Queue
task:queue:high = LIST of task_ids (priority 8-10)
task:queue:medium = LIST of task_ids (priority 4-7)
task:queue:low = LIST of task_ids (priority 1-3)

# Real-time Metrics
metrics:realtime:{metric_type}:{source} = {
    "value": number,
    "timestamp": "timestamp"
}
TTL: 300 seconds

# Model Cache
model:cache:{model_name} = {
    "loaded": boolean,
    "last_used": "timestamp",
    "memory_usage_mb": number
}
TTL: 600 seconds

# Voice Processing Buffer
voice:buffer:{session_id} = BINARY audio data
TTL: 60 seconds

# Agent Communication
agent:comm:{from_agent}:{to_agent} = LIST of messages
TTL: 300 seconds

# Rate Limiting
rate:limit:{client_ip}:{endpoint} = counter
TTL: 60 seconds

# Distributed Locks
lock:agent:{agent_id} = "locked_by_task_id"
TTL: 30 seconds
```

## Neo4j Schema (Port 10002)

### Node Types

```cypher
// Agent Node
CREATE (a:Agent {
    id: 'uuid',
    name: 'string',
    type: 'string',
    capabilities: ['list'],
    created_at: datetime()
})

// Task Node  
CREATE (t:Task {
    id: 'uuid',
    name: 'string',
    type: 'string',
    status: 'string',
    created_at: datetime()
})

// User Node
CREATE (u:User {
    id: 'string',
    name: 'string',
    preferences: {}
})

// Model Node
CREATE (m:Model {
    id: 'uuid',
    name: 'string',
    type: 'string',
    provider: 'string'
})

// Capability Node
CREATE (c:Capability {
    name: 'string',
    category: 'string',
    description: 'string'
})

// Knowledge Node
CREATE (k:Knowledge {
    id: 'uuid',
    topic: 'string',
    content: 'string',
    source: 'string',
    created_at: datetime()
})
```

### Relationships

```cypher
// Task Relationships
(t:Task)-[:ASSIGNED_TO]->(a:Agent)
(t:Task)-[:REQUESTED_BY]->(u:User)
(t:Task)-[:DEPENDS_ON]->(t2:Task)
(t:Task)-[:PRODUCES]->(k:Knowledge)

// Agent Relationships
(a:Agent)-[:HAS_CAPABILITY]->(c:Capability)
(a:Agent)-[:USES_MODEL]->(m:Model)
(a:Agent)-[:COLLABORATES_WITH]->(a2:Agent)
(a:Agent)-[:LEARNED_FROM]->(t:Task)

// User Relationships
(u:User)-[:PREFERS]->(a:Agent)
(u:User)-[:INTERACTED_WITH]->(a:Agent)

// Knowledge Graph
(k:Knowledge)-[:RELATED_TO]->(k2:Knowledge)
(k:Knowledge)-[:GENERATED_BY]->(a:Agent)
(k:Knowledge)-[:USED_IN]->(t:Task)
```

### Example Queries

```cypher
// Find best agent for task
MATCH (c:Capability {name: $capability})
MATCH (a:Agent)-[:HAS_CAPABILITY]->(c)
WHERE a.status = 'active'
RETURN a
ORDER BY a.performance_score DESC
LIMIT 1

// Get collaboration history
MATCH (a1:Agent)-[:COLLABORATES_WITH]->(a2:Agent)
WHERE a1.id = $agent_id
RETURN a2, count(*) as collaboration_count
ORDER BY collaboration_count DESC

// Knowledge graph traversal
MATCH path = (k1:Knowledge)-[:RELATED_TO*1..3]->(k2:Knowledge)
WHERE k1.topic = $topic
RETURN path
```

## ChromaDB Collections (Port 10100)

```python
# Collection Schemas

conversations_collection = {
    "name": "conversations",
    "metadata": {
        "description": "User conversation embeddings",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
    },
    "documents": [
        {
            "id": "conv_id",
            "document": "conversation_text",
            "metadata": {
                "session_id": "string",
                "user_id": "string",
                "agent_id": "uuid",
                "timestamp": "datetime",
                "sentiment": "positive|negative|neutral"
            }
        }
    ]
}

knowledge_collection = {
    "name": "knowledge_base",
    "metadata": {
        "description": "Agent knowledge and documentation",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
    },
    "documents": [
        {
            "id": "knowledge_id",
            "document": "knowledge_content",
            "metadata": {
                "source": "string",
                "agent_id": "uuid",
                "category": "string",
                "confidence": 0.95,
                "created_at": "datetime"
            }
        }
    ]
}

code_collection = {
    "name": "code_snippets",
    "metadata": {
        "description": "Generated code and improvements",
        "embedding_model": "codbert-base"
    },
    "documents": [
        {
            "id": "code_id",
            "document": "code_content",
            "metadata": {
                "language": "python|javascript|yaml",
                "purpose": "string",
                "agent_id": "uuid",
                "performance_score": 0.85,
                "tested": true
            }
        }
    ]
}
```

## Qdrant Collections (Port 10101)

```python
# Collection Configuration

from qdrant_client.models import VectorParams, Distance

agent_capabilities = {
    "collection_name": "agent_capabilities",
    "vectors_config": VectorParams(
        size=384,
        distance=Distance.COSINE
    ),
    "points": [
        {
            "id": "capability_id",
            "vector": [0.1, 0.2, ...],  # 384-dim embedding
            "payload": {
                "agent_id": "uuid",
                "capability": "string",
                "description": "string",
                "success_rate": 0.92,
                "avg_execution_time": 1500
            }
        }
    ]
}

task_patterns = {
    "collection_name": "task_patterns",
    "vectors_config": VectorParams(
        size=384,
        distance=Distance.COSINE
    ),
    "points": [
        {
            "id": "pattern_id",
            "vector": [0.1, 0.2, ...],
            "payload": {
                "pattern_type": "string",
                "task_sequence": ["step1", "step2"],
                "success_rate": 0.88,
                "optimal_agents": ["agent1", "agent2"]
            }
        }
    ]
}

user_preferences = {
    "collection_name": "user_preferences",
    "vectors_config": VectorParams(
        size=384,
        distance=Distance.EUCLID
    ),
    "points": [
        {
            "id": "preference_id",
            "vector": [0.1, 0.2, ...],
            "payload": {
                "user_id": "string",
                "preference_type": "agent|task|interface",
                "value": {},
                "confidence": 0.75
            }
        }
    ]
}
```

## Database Migrations

```sql
-- Migration: 001_initial_schema.sql
BEGIN;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create all tables
-- [Tables from above]

-- Create indexes
CREATE INDEX idx_conversations_text_search ON conversations 
USING gin(to_tsvector('english', content));

-- Create triggers for updated_at
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_agents_updated_at 
BEFORE UPDATE ON agents
FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_agent_memory_updated_at
BEFORE UPDATE ON agent_memory  
FOR EACH ROW EXECUTE FUNCTION update_updated_at();

COMMIT;

-- Migration: 002_add_partitioning.sql
BEGIN;

-- Partition large tables by date
CREATE TABLE conversations_2025_01 PARTITION OF conversations
FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

CREATE TABLE system_events_2025_01 PARTITION OF system_events
FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

CREATE TABLE performance_metrics_2025_01 PARTITION OF performance_metrics
FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

COMMIT;
```

## Data Retention Policies

```yaml
retention_policies:
  conversations:
    default: 90_days
    voice_recordings: 30_days
    
  system_events:
    debug: 7_days
    info: 30_days
    warning: 90_days
    error: 180_days
    critical: 365_days
    
  performance_metrics:
    realtime: 24_hours
    hourly: 7_days
    daily: 90_days
    
  agent_memory:
    working: 1_hour
    recall: 7_days
    core: permanent
    archival: permanent
    
  improvement_logs:
    unapplied: 30_days
    applied: permanent
```

## Backup Strategy

```bash
#!/bin/bash
# backup_databases.sh

# PostgreSQL backup
pg_dump -h localhost -p 10000 -U postgres jarvis_db > /backup/postgres_$(date +%Y%m%d).sql

# Redis backup
redis-cli -p 10001 BGSAVE
cp /data/redis/dump.rdb /backup/redis_$(date +%Y%m%d).rdb

# Neo4j backup
neo4j-admin backup --database=neo4j --backup-dir=/backup/neo4j_$(date +%Y%m%d)

# ChromaDB backup (file-based)
tar -czf /backup/chromadb_$(date +%Y%m%d).tar.gz /data/chromadb

# Qdrant backup
curl -X POST 'http://localhost:10101/collections/agent_capabilities/snapshots'
```

## Performance Optimization

### PostgreSQL
```sql
-- Optimize queries with proper indexes
CREATE INDEX CONCURRENTLY idx_tasks_status_priority 
ON tasks(status, priority DESC) 
WHERE status IN ('pending', 'running');

-- Vacuum and analyze regularly
VACUUM ANALYZE agents;
VACUUM ANALYZE tasks;
VACUUM ANALYZE conversations;

-- Connection pooling configuration
max_connections = 200
shared_buffers = 256MB
effective_cache_size = 1GB
```

### Redis
```redis
# Configure for performance
maxmemory 512mb
maxmemory-policy allkeys-lru
save ""  # Disable persistence for cache-only data
```

### Neo4j
```cypher
// Create indexes for performance
CREATE INDEX agent_id_index FOR (a:Agent) ON (a.id);
CREATE INDEX task_id_index FOR (t:Task) ON (t.id);
CREATE INDEX knowledge_topic_index FOR (k:Knowledge) ON (k.topic);
```

## Security Considerations

```sql
-- Row-level security for multi-tenancy
ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;

CREATE POLICY user_conversations ON conversations
    FOR ALL
    USING (user_id = current_setting('app.current_user'));

-- Encryption at rest
CREATE EXTENSION pgcrypto;

ALTER TABLE agent_memory 
    ALTER COLUMN value TYPE TEXT USING pgp_sym_encrypt(value, 'encryption_key');
```

## Monitoring Queries

```sql
-- Active agent summary
SELECT 
    type,
    COUNT(*) as count,
    COUNT(CASE WHEN status = 'active' THEN 1 END) as active,
    AVG(EXTRACT(EPOCH FROM (NOW() - last_heartbeat))) as avg_heartbeat_age
FROM agents
GROUP BY type;

-- Task performance
SELECT 
    agent_id,
    COUNT(*) as total_tasks,
    AVG(execution_time_ms) as avg_time,
    COUNT(CASE WHEN status = 'completed' THEN 1 END) * 100.0 / COUNT(*) as success_rate
FROM tasks
WHERE created_at > NOW() - INTERVAL '24 hours'
GROUP BY agent_id;

-- Memory usage by agent
SELECT 
    agent_id,
    memory_type,
    COUNT(*) as entries,
    pg_size_pretty(SUM(LENGTH(value)::BIGINT)) as total_size
FROM agent_memory
GROUP BY agent_id, memory_type
ORDER BY SUM(LENGTH(value)) DESC;
```