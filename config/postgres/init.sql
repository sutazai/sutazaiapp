-- Unified System Database Initialization
-- Creates the necessary database structure for persistent learning

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enable vector extension (if available)
-- CREATE EXTENSION IF NOT EXISTS vector;

-- Create user if not exists
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'sutazai') THEN
        CREATE USER sutazai WITH PASSWORD 'sutazai123';
    END IF;
END
$$;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE sutazai TO sutazai;
GRANT ALL ON SCHEMA public TO sutazai;

-- Set default privileges
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO sutazai;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO sutazai;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON FUNCTIONS TO sutazai;