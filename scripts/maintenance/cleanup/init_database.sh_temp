#!/bin/bash

# Ultra-thinking Database Initialization Script
# Created by System Architect, Backend Architect, and DevOps Manager
# Purpose: Initialize PostgreSQL with proper users, database, and schema

set -e


# Signal handlers for graceful shutdown
cleanup_and_exit() {
    local exit_code="${1:-0}"
    echo "Script interrupted, cleaning up..." >&2
    # Clean up any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    exit "$exit_code"
}

trap 'cleanup_and_exit 130' INT
trap 'cleanup_and_exit 143' TERM
trap 'cleanup_and_exit 1' ERR

echo "========================================="
echo "ULTRA-THINKING DATABASE INITIALIZATION"
echo "========================================="

# Check if PostgreSQL container is running
if ! docker ps | grep -q sutazai-postgres; then
    echo "ERROR: PostgreSQL container is not running"
    exit 1
fi

echo "Step 1: Creating postgres superuser role..."
docker exec sutazai-postgres su - postgres -c "createuser -s postgres" 2>/dev/null || echo "  - postgres user may already exist"

echo "Step 2: Creating application user 'sutazaiuser'..."
docker exec sutazai-postgres su - postgres -c "createuser sutazaiuser" 2>/dev/null || echo "  - sutazaiuser may already exist"

echo "Step 3: Creating database 'sutazai'..."
docker exec sutazai-postgres su - postgres -c "createdb sutazai -O sutazaiuser" 2>/dev/null || echo "  - database may already exist"

echo "Step 4: Setting password for sutazaiuser..."
docker exec sutazai-postgres su - postgres -c "psql -c \"ALTER USER sutazaiuser WITH PASSWORD 'sutazai2024';\"" || echo "  - password update failed"

echo "Step 5: Granting privileges..."
docker exec sutazai-postgres su - postgres -c "psql -d sutazai -c \"GRANT ALL PRIVILEGES ON DATABASE sutazai TO sutazaiuser;\"" || echo "  - privileges may already be granted"

echo "Step 6: Creating schema..."
docker exec sutazai-postgres su - postgres -c "psql -d sutazai -c \"
-- Create schema if not exists
CREATE SCHEMA IF NOT EXISTS public;

-- Create tables with UUID primary keys (following codebase standards)
CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT true,
    is_superuser BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    token VARCHAR(500) UNIQUE NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Agents table
CREATE TABLE IF NOT EXISTS agents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) UNIQUE NOT NULL,
    type VARCHAR(100) NOT NULL,
    status VARCHAR(50) DEFAULT 'inactive',
    config JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tasks table
CREATE TABLE IF NOT EXISTS tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID REFERENCES agents(id) ON DELETE SET NULL,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    type VARCHAR(100) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    input_data JSONB,
    output_data JSONB,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

-- Conversations table
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(500),
    model VARCHAR(100) DEFAULT 'tinyllama',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Messages table
CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(token);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);

-- Grant permissions
GRANT ALL ON ALL TABLES IN SCHEMA public TO sutazaiuser;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO sutazaiuser;
\""

echo "Step 7: Verifying database setup..."
docker exec sutazai-postgres su - postgres -c "psql -d sutazai -c '\dt'" || echo "  - verification failed"

echo "========================================="
echo "DATABASE INITIALIZATION COMPLETE"
echo "========================================="
echo ""
echo "Connection details:"
echo "  Host: localhost"
echo "  Port: 10000"
echo "  Database: sutazai"
echo "  User: sutazaiuser"
echo "  Password: sutazai2024"
echo ""
echo "Test connection:"
echo "  docker exec sutazai-postgres psql -U sutazaiuser -d sutazai -c '\dt'"