#!/bin/bash
# Automatically optimize database tables
docker exec sutazai-postgres psql -U postgres -c "VACUUM ANALYZE;"
echo "Database optimization completed!" 