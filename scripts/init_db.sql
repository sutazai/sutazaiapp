-- Create user and additional databases
CREATE USER sutazai WITH PASSWORD 'sutazai_secure_2024' CREATEDB SUPERUSER;
ALTER USER sutazai CREATEDB;

CREATE DATABASE vector_store OWNER sutazai;
CREATE DATABASE agent_memory OWNER sutazai;
CREATE DATABASE langflow OWNER sutazai;

-- Create extensions
\c sutazai;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- MCP server bootstrap schema (idempotent)
-- Agents registry
CREATE TABLE IF NOT EXISTS agents (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_name      TEXT NOT NULL UNIQUE,
    agent_type      TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'inactive',
    capabilities    JSONB NOT NULL DEFAULT '[]'::jsonb,
    config          JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ
);

-- Agent tasks
CREATE TABLE IF NOT EXISTS agent_tasks (
    task_id         BIGSERIAL PRIMARY KEY,
    agent_name      TEXT NOT NULL,
    task_description TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    priority        TEXT NOT NULL DEFAULT 'normal',
    context         JSONB NOT NULL DEFAULT '{}'::jsonb,
    result          JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at    TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_agent_tasks_created_at ON agent_tasks (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_tasks_agent_name ON agent_tasks (agent_name);

-- Knowledge documents (embeddings metadata)
CREATE TABLE IF NOT EXISTS knowledge_documents (
    document_id     UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title           TEXT NOT NULL,
    content_preview TEXT,
    embedding_model TEXT,
    collection_name TEXT,
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_knowledge_docs_created_at ON knowledge_documents (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_knowledge_docs_collection ON knowledge_documents (collection_name);

-- Agent workspaces
CREATE TABLE IF NOT EXISTS agent_workspaces (
    workspace_id    UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_name      TEXT NOT NULL,
    workspace_path  TEXT NOT NULL,
    size_mb         NUMERIC,
    last_modified   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    backup_status   TEXT
);
CREATE INDEX IF NOT EXISTS idx_agent_workspaces_last_modified ON agent_workspaces (last_modified DESC);

\c vector_store;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

\c agent_memory;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

\c langflow;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
