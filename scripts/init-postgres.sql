-- Create additional databases
CREATE DATABASE vector_store;
CREATE DATABASE agent_memory;

-- Create extensions
\c sutazai;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

\c vector_store;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

\c agent_memory;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
