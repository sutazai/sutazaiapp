-- PostgreSQL Security Hardening Configuration
-- Rule 11 Compliance: Database security standards

-- Create security-focused roles and privileges
DO $$
BEGIN
    -- Create application-specific roles
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'sutazai_app') THEN
        CREATE ROLE sutazai_app WITH LOGIN PASSWORD '${POSTGRES_APP_PASSWORD}';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'sutazai_readonly') THEN
        CREATE ROLE sutazai_readonly WITH LOGIN PASSWORD '${POSTGRES_READONLY_PASSWORD}';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'sutazai_monitor') THEN
        CREATE ROLE sutazai_monitor WITH LOGIN PASSWORD '${POSTGRES_MONITOR_PASSWORD}';
    END IF;
END
$$;

-- Grant appropriate permissions
GRANT CONNECT ON DATABASE sutazai TO sutazai_app;
GRANT CONNECT ON DATABASE sutazai TO sutazai_readonly;
GRANT CONNECT ON DATABASE sutazai TO sutazai_monitor;

-- Create schemas for better security isolation
CREATE SCHEMA IF NOT EXISTS app_data AUTHORIZATION sutazai_app;
CREATE SCHEMA IF NOT EXISTS app_config AUTHORIZATION sutazai_app;
CREATE SCHEMA IF NOT EXISTS app_logs AUTHORIZATION sutazai_app;

-- Set up row-level security
ALTER DATABASE sutazai SET row_security = on;

-- Grant privileges
GRANT USAGE ON SCHEMA app_data TO sutazai_app;
GRANT USAGE ON SCHEMA app_config TO sutazai_app;
GRANT USAGE ON SCHEMA app_logs TO sutazai_app;

GRANT SELECT ON ALL TABLES IN SCHEMA app_data TO sutazai_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA app_config TO sutazai_readonly;

-- Monitoring privileges
GRANT pg_monitor TO sutazai_monitor;
GRANT SELECT ON pg_stat_database TO sutazai_monitor;
GRANT SELECT ON pg_stat_user_tables TO sutazai_monitor;

-- Security settings
ALTER SYSTEM SET log_statement = 'all';
ALTER SYSTEM SET log_min_duration_statement = 1000;
ALTER SYSTEM SET log_connections = on;
ALTER SYSTEM SET log_disconnections = on;
ALTER SYSTEM SET log_lock_waits = on;
ALTER SYSTEM SET log_checkpoints = on;
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';

-- Performance and security tuning
ALTER SYSTEM SET max_connections = 100;
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET work_mem = '4MB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;

-- SSL configuration
ALTER SYSTEM SET ssl = on;
ALTER SYSTEM SET ssl_cert_file = 'server.crt';
ALTER SYSTEM SET ssl_key_file = 'server.key';
ALTER SYSTEM SET ssl_prefer_server_ciphers = on;
ALTER SYSTEM SET ssl_ciphers = 'ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-SHA384:ECDHE-RSA-AES128-SHA256';

-- Reload configuration
SELECT pg_reload_conf();