# Database Consolidation Plan

## Current State Analysis

### Databases Found (As of 2025-08-20)

#### Memory Databases (12 SQLite files)
1. **Main memory.db**: `/opt/sutazaiapp/.swarm/memory.db`
   - Size: 16M
   - Entries: 21,453
   - Primary namespace: performance-metrics (3933 entries)

2. **Docker dind memory**: `/opt/sutazaiapp/docker/dind/.swarm/memory.db`
   - Size: 1.1M
   - Entries: 2,098

3. **Backend memory**: `/opt/sutazaiapp/backend/.swarm/memory.db`
   - Size: 796K
   - Entries: 1,439

4. **Other memory.db files**: 9 additional files with 2,032 total entries

#### Extended Memory Database
- Location: `/opt/sutazaiapp/data/mcp/extended-memory/extended_memory.db`
- Size: 112K
- Entries: 165
- Schema: Different from memory.db (uses `memory_store` table)

#### Application Databases
- N8N: 452K (workflow automation data)
- Flowise: 376K (chatflow data)

### Total Data Statistics
- **Total memory entries**: 27,589 across all databases
- **Total database files**: 16
- **Total storage**: ~20MB

## Unified Database Design

### Target Architecture
- **Primary Database**: PostgreSQL (port 10000)
- **Cache Layer**: Redis (port 10001)
- **Unified Schema**: Single consolidated schema supporting all use cases

### Proposed PostgreSQL Schema

```sql
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
    
    -- Indexing
    UNIQUE(key, namespace)
);

-- Indexes for performance
CREATE INDEX idx_unified_memory_namespace ON unified_memory(namespace);
CREATE INDEX idx_unified_memory_expires ON unified_memory(expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX idx_unified_memory_accessed ON unified_memory(accessed_at);
CREATE INDEX idx_unified_memory_source ON unified_memory(source_db);
CREATE INDEX idx_unified_memory_data_type ON unified_memory(data_type);
CREATE INDEX idx_unified_memory_value_gin ON unified_memory USING gin(value);

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
    namespace VARCHAR(255) DEFAULT 'default',
    
    -- Indexing
    INDEX idx_metrics_name (metric_name),
    INDEX idx_metrics_timestamp (timestamp DESC)
);
```

## Migration Strategy

### Phase 1: Backup and Preparation
1. Create full backups of all SQLite databases
2. Set up PostgreSQL database with unified schema
3. Create migration tracking infrastructure

### Phase 2: Data Migration
1. Migrate main memory.db (highest priority - 21,453 entries)
2. Migrate extended memory database (165 entries)
3. Migrate Docker-related memory databases (3,537 entries)
4. Migrate remaining memory databases (2,032 entries)

### Phase 3: Validation
1. Verify record counts match
2. Spot-check data integrity
3. Test application connectivity

### Phase 4: Cutover
1. Update application configurations
2. Switch to PostgreSQL connections
3. Monitor for issues

### Phase 5: Cleanup
1. Archive old SQLite databases
2. Remove database files after verification period
3. Document final state

## Risk Mitigation

### Data Integrity
- Full backups before migration
- Checksums for data validation
- Transaction-based migration (rollback capability)

### Zero Downtime Strategy
- Parallel run period (write to both)
- Gradual cutover by service
- Rollback plan ready

### Performance Considerations
- PostgreSQL connection pooling
- Redis caching for hot data
- Proper indexing from day one

## Implementation Timeline

1. **Day 1**: Backup, schema creation, migration scripts
2. **Day 2**: Test migration on dev environment
3. **Day 3**: Production migration and validation
4. **Day 4**: Application cutover
5. **Day 5**: Monitoring and cleanup

## Success Criteria

- ✅ All 27,589 records migrated successfully
- ✅ Zero data loss verified
- ✅ Application performance maintained or improved
- ✅ All services connected to unified database
- ✅ Old databases safely archived