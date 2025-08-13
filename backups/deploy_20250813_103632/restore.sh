#!/bin/bash
# Restore script for backup
set -e

BACKUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$BACKUP_DIR")")"

echo "Restoring from backup: $BACKUP_DIR"

# Restore database
echo "Restoring PostgreSQL..."
gunzip -c "$BACKUP_DIR/postgres_backup.sql.gz" | \
    docker exec -i sutazai-postgres psql -U sutazai sutazai

# Restore Redis
echo "Restoring Redis..."
docker cp "$BACKUP_DIR/redis_backup.rdb" sutazai-redis:/data/dump.rdb
docker exec sutazai-redis redis-cli SHUTDOWN NOSAVE
docker restart sutazai-redis

echo "Restore complete!"
