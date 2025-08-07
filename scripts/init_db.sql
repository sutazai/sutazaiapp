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

\c vector_store;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

\c agent_memory;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

\c langflow;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
