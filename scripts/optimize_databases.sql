-- PostgreSQL Database Optimization Script
-- Adds production-ready indexes for high-traffic queries
-- Safe to run multiple times (uses CREATE INDEX IF NOT EXISTS)

-- Connect to jarvis_ai database
\c jarvis_ai;

-- ============================================================
-- User and Authentication Indexes
-- ============================================================

-- Email is already indexed in the model (unique index)
-- Username is already indexed in the model (unique index)

-- Add index for login queries (last_login for session management)
CREATE INDEX IF NOT EXISTS idx_users_last_login ON users(last_login DESC);

-- Add index for active users filtering
CREATE INDEX IF NOT EXISTS idx_users_is_active ON users(is_active) WHERE is_active = true;

-- Add composite index for email + active status (common auth query)
CREATE INDEX IF NOT EXISTS idx_users_email_active ON users(email, is_active);

-- Add index for account security queries
CREATE INDEX IF NOT EXISTS idx_users_locked ON users(account_locked_until) 
    WHERE account_locked_until IS NOT NULL AND account_locked_until > NOW();

-- ============================================================
-- Additional Tables (if they exist)
-- These will create indexes only if the tables are present
-- ============================================================

-- Conversation/Chat history indexes (common pattern)
DO $$
BEGIN
    IF EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'conversations') THEN
        CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);
        CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_conversations_user_created ON conversations(user_id, created_at DESC);
        RAISE NOTICE 'Created indexes on conversations table';
    END IF;
END $$;

-- Message history indexes
DO $$
BEGIN
    IF EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'messages') THEN
        CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
        CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_messages_conversation_created ON messages(conversation_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_messages_user_id ON messages(user_id) WHERE user_id IS NOT NULL;
        RAISE NOTICE 'Created indexes on messages table';
    END IF;
END $$;

-- Session tracking indexes (if using database sessions)
DO $$
BEGIN
    IF EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'sessions') THEN
        CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
        CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(token);
        CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at) WHERE expires_at > NOW();
        RAISE NOTICE 'Created indexes on sessions table';
    END IF;
END $$;

-- Agent task tracking indexes
DO $$
BEGIN
    IF EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'agent_tasks') THEN
        CREATE INDEX IF NOT EXISTS idx_agent_tasks_user_id ON agent_tasks(user_id);
        CREATE INDEX IF NOT EXISTS idx_agent_tasks_status ON agent_tasks(status);
        CREATE INDEX IF NOT EXISTS idx_agent_tasks_created_at ON agent_tasks(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_agent_tasks_agent_id ON agent_tasks(agent_id);
        RAISE NOTICE 'Created indexes on agent_tasks table';
    END IF;
END $$;

-- File uploads/documents indexes
DO $$
BEGIN
    IF EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'documents') THEN
        CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id);
        CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(document_type);
        RAISE NOTICE 'Created indexes on documents table';
    END IF;
END $$;

-- ============================================================
-- Performance Optimization: VACUUM and ANALYZE
-- ============================================================

-- Update table statistics for query planner
ANALYZE users;

-- Vacuum to reclaim space and update statistics
VACUUM ANALYZE;

-- ============================================================
-- Summary Report
-- ============================================================

SELECT 
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE schemaname = 'public'
ORDER BY tablename, indexname;

\echo 'Database optimization complete!'
\echo 'Indexes created on existing tables'
\echo 'Run EXPLAIN ANALYZE on slow queries to verify index usage'
