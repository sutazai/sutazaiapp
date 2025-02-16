#!/bin/bash
# Comprehensive backup system

# Create backup directory
BACKUP_DIR="/var/backups/sutazai"
mkdir -p $BACKUP_DIR

# Create backup script
cat > /usr/local/bin/backup_sutazai <<EOF
#!/bin/bash
TIMESTAMP=\$(date +%Y%m%d%H%M%S)
BACKUP_FILE="$BACKUP_DIR/backup_\$TIMESTAMP.tar.gz"

# Backup databases
docker exec sutazai-postgres pg_dumpall -U postgres > /tmp/postgres.sql
docker exec sutazai-redis redis-cli save && cp /var/lib/redis/dump.rdb /tmp/redis.rdb

# Backup configurations
tar -czf \$BACKUP_FILE /tmp/postgres.sql /tmp/redis.rdb /etc/sutazai

# Cleanup
rm -f /tmp/postgres.sql /tmp/redis.rdb

echo "Backup completed: \$BACKUP_FILE"
EOF

# Make executable
chmod +x /usr/local/bin/backup_sutazai

# Schedule daily backups
(crontab -l 2>/dev/null; echo "0 2 * * * /usr/local/bin/backup_sutazai") | crontab -

echo "Backup system configured successfully!" 