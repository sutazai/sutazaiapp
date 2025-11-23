-- Create Kong database
-- This script is executed during PostgreSQL container initialization

CREATE DATABASE kong OWNER jarvis;
\c kong;

-- Grant necessary permissions
GRANT ALL PRIVILEGES ON DATABASE kong TO jarvis;

\c jarvis_ai;

-- Ensure proper schema is set up
CREATE SCHEMA IF NOT EXISTS public;
GRANT ALL ON SCHEMA public TO jarvis;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO jarvis;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO jarvis;

-- Log successful initialization
DO $$ 
BEGIN 
    RAISE NOTICE 'SutazAI PostgreSQL initialized successfully at %', NOW();
END $$;
