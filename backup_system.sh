#!/bin/bash

BACKUP_DIR="data/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="sutazai_backup_${TIMESTAMP}"

echo "Creating system backup: $BACKUP_NAME"

# Create backup directory
mkdir -p "$BACKUP_DIR/$BACKUP_NAME"

# Backup databases
echo "Backing up PostgreSQL..."
docker exec sutazai-postgres pg_dump -U sutazai sutazai > "$BACKUP_DIR/$BACKUP_NAME/postgres.sql"

echo "Backing up Redis..."
docker exec sutazai-redis redis-cli BGSAVE
docker cp sutazai-redis:/data/dump.rdb "$BACKUP_DIR/$BACKUP_NAME/"

echo "Backing up Qdrant..."
docker cp sutazai-qdrant:/qdrant/storage "$BACKUP_DIR/$BACKUP_NAME/qdrant"

echo "Backing up ChromaDB..."
docker cp sutazai-chromadb:/chroma/chroma "$BACKUP_DIR/$BACKUP_NAME/chromadb"

echo "Backing up models..."
docker cp sutazai-ollama:/root/.ollama "$BACKUP_DIR/$BACKUP_NAME/ollama"

# Compress backup
echo "Compressing backup..."
tar -czf "$BACKUP_DIR/${BACKUP_NAME}.tar.gz" -C "$BACKUP_DIR" "$BACKUP_NAME"
rm -rf "$BACKUP_DIR/$BACKUP_NAME"

echo "Backup completed: ${BACKUP_NAME}.tar.gz"

# Cleanup old backups (keep last 7 days)
find "$BACKUP_DIR" -name "sutazai_backup_*.tar.gz" -mtime +7 -delete
