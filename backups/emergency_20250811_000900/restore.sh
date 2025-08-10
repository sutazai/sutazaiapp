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
