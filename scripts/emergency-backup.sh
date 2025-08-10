#!/bin/bash
# Emergency System Backup Script
# Created by: Ultra System Architect
# Date: August 10, 2025
# Purpose: Create comprehensive backup before system changes

set -e

echo "========================================="
echo "EMERGENCY BACKUP SYSTEM - SutazAI v76"
echo "========================================="

BACKUP_DIR="/opt/sutazaiapp/backups/emergency_$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP_DIR

echo "[1/5] Creating backup directory: $BACKUP_DIR"

# Database backups
echo "[2/5] Backing up PostgreSQL database..."
docker exec sutazai-postgres pg_dumpall -U sutazai > $BACKUP_DIR/postgres.sql 2>/dev/null || echo "PostgreSQL backup failed"

echo "[3/5] Backing up Redis database..."
docker exec sutazai-redis redis-cli BGSAVE >/dev/null 2>&1 || echo "Redis BGSAVE initiated"
sleep 3
docker cp sutazai-redis:/data/dump.rdb $BACKUP_DIR/redis.rdb 2>/dev/null || echo "Redis backup failed"

echo "[4/5] Backing up Neo4j database..."
docker exec sutazai-neo4j mkdir -p /backup 2>/dev/null || true
docker exec sutazai-neo4j neo4j-admin database dump --to-path=/backup neo4j 2>/dev/null || echo "Neo4j backup failed"
docker cp sutazai-neo4j:/backup $BACKUP_DIR/neo4j_backup 2>/dev/null || echo "Neo4j backup copy failed"

# Configuration backup
echo "[5/5] Backing up configuration files..."
cp docker-compose.yml $BACKUP_DIR/ 2>/dev/null || echo "docker-compose.yml backup failed"
cp .env $BACKUP_DIR/.env.backup 2>/dev/null || echo ".env backup failed"
tar -czf $BACKUP_DIR/configs.tar.gz config/ 2>/dev/null || echo "Config directory backup failed"

# Create backup manifest
cat > $BACKUP_DIR/manifest.txt <<EOF
Backup Created: $(date)
System Version: v76
Containers Running: $(docker ps -q | wc -l)
Git Branch: $(git branch --show-current)
Last Commit: $(git log -1 --oneline)
EOF

echo "========================================="
echo "BACKUP COMPLETE"
echo "Location: $BACKUP_DIR"
echo "========================================="
ls -lah $BACKUP_DIR/

# Create restore script
cat > $BACKUP_DIR/restore.sh <<'RESTORE'
#!/bin/bash
# Restore script for this backup

echo "Restoring from backup..."

# Stop services
docker compose down

# Restore databases
docker compose up -d postgres redis neo4j
sleep 10

# Restore PostgreSQL
docker exec -i sutazai-postgres psql -U sutazai < postgres.sql

# Restore Redis
docker cp redis.rdb sutazai-redis:/data/dump.rdb
docker exec sutazai-redis redis-cli SHUTDOWN NOSAVE
docker compose restart redis

# Restore configuration
cp docker-compose.yml ../../
cp .env.backup ../../.env
tar -xzf configs.tar.gz -C ../../

# Restart all services
docker compose up -d

echo "Restore complete!"
RESTORE

chmod +x $BACKUP_DIR/restore.sh

echo "Restore script created: $BACKUP_DIR/restore.sh"