#!/bin/bash
# Ultra-Safe Backup Script - MUST RUN FIRST
set -euo pipefail


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

BACKUP_DIR="/opt/sutazaiapp/backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP_DIR

echo "=== Creating Ultra-Safe Backup ==="
echo "Backup directory: $BACKUP_DIR"

# 1. Stop writes to databases for consistency
echo "Preparing databases for backup..."
docker exec sutazai-postgres psql -U sutazai -c "CHECKPOINT;" 2>/dev/null || true
docker exec sutazai-redis redis-cli BGSAVE 2>/dev/null || true
sleep 2

# 2. Database dumps
echo "Backing up PostgreSQL..."
docker exec sutazai-postgres pg_dumpall -U sutazai > $BACKUP_DIR/postgres_full.sql 2>/dev/null || echo "Warning: PostgreSQL backup failed"

echo "Backing up Redis..."
docker exec sutazai-redis redis-cli --rdb /data/dump.rdb 2>/dev/null || true
docker cp sutazai-redis:/data/dump.rdb $BACKUP_DIR/redis.rdb 2>/dev/null || echo "Warning: Redis backup failed"

echo "Backing up Neo4j..."
docker exec sutazai-neo4j neo4j-admin database dump --to-path=/backup neo4j 2>/dev/null || true
docker cp sutazai-neo4j:/backup $BACKUP_DIR/neo4j_backup 2>/dev/null || echo "Warning: Neo4j backup failed"

# 3. Configuration backup
echo "Backing up configurations..."
tar -czf $BACKUP_DIR/configs.tar.gz \
    docker-compose.yml \
    docker-compose.*.yml \
    .env* \
    config/ \
    2>/dev/null || true

# 4. Create restore script
cat > $BACKUP_DIR/restore.sh <<'EOF'
#!/bin/bash
# Restore script for this backup
set -euo pipefail

BACKUP_DIR="$(dirname "$0")"
cd /opt/sutazaiapp

echo "=== Restoring from backup: $BACKUP_DIR ==="

# Stop current system
docker-compose down

# Restore configurations
tar -xzf $BACKUP_DIR/configs.tar.gz

# Start databases
docker-compose up -d postgres redis neo4j
sleep 10

# Restore database data
if [ -f "$BACKUP_DIR/postgres_full.sql" ]; then
    echo "Restoring PostgreSQL..."
    docker exec -i sutazai-postgres psql -U sutazai < $BACKUP_DIR/postgres_full.sql
fi

if [ -f "$BACKUP_DIR/redis.rdb" ]; then
    echo "Restoring Redis..."
    docker cp $BACKUP_DIR/redis.rdb sutazai-redis:/data/dump.rdb
    docker exec sutazai-redis redis-cli SHUTDOWN NOSAVE
    docker-compose restart redis
fi

if [ -d "$BACKUP_DIR/neo4j_backup" ]; then
    echo "Restoring Neo4j..."
    docker cp $BACKUP_DIR/neo4j_backup sutazai-neo4j:/backup
    docker exec sutazai-neo4j neo4j-admin database load --from-path=/backup neo4j --overwrite-destination=true
fi

# Bring full system back up
docker-compose up -d

echo "Restore complete. Waiting for services to stabilize..."
sleep 30

# Validate
curl -s http://localhost:10010/health && echo "Backend: OK" || echo "Backend: FAILED"
curl -s http://localhost:10011/ > /dev/null && echo "Frontend: OK" || echo "Frontend: FAILED"
EOF

chmod +x $BACKUP_DIR/restore.sh

# 5. Capture current state
docker ps --format "{{.Names}},{{.Status}}" > $BACKUP_DIR/container_state.csv
docker stats --no-stream --format "{{.Name}},{{.MemUsage}},{{.CPUPerc}}" > $BACKUP_DIR/resource_state.csv

echo "=== Backup Complete ==="
echo "Location: $BACKUP_DIR"
echo "Size: $(du -sh $BACKUP_DIR | cut -f1)"
echo "Restore command: $BACKUP_DIR/restore.sh"
echo ""
echo "IMPORTANT: Test restore script before proceeding with cleanup!"