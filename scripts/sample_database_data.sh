#!/bin/bash

echo "=== Database Data Sampling Report ==="
echo "Date: $(date)"
echo ""

# Sample from main memory.db
echo "Sample from main memory.db (/opt/sutazaiapp/.swarm/memory.db):"
echo "=============================================================="
sqlite3 /opt/sutazaiapp/.swarm/memory.db <<EOF
.mode column
.headers on
SELECT key, namespace, length(value) as value_size, metadata, created_at, access_count
FROM memory_entries
ORDER BY access_count DESC
LIMIT 10;
EOF

echo ""
echo "Namespace distribution:"
sqlite3 /opt/sutazaiapp/.swarm/memory.db <<EOF
SELECT namespace, COUNT(*) as count
FROM memory_entries
GROUP BY namespace
ORDER BY count DESC;
EOF

echo ""
echo "Extended Memory Database Sample:"
echo "================================"
sqlite3 /opt/sutazaiapp/data/mcp/extended-memory/extended_memory.db <<EOF
.mode column
.headers on
SELECT key, type, length(value) as value_size, access_count
FROM memory_store
ORDER BY access_count DESC
LIMIT 10;
EOF

echo ""
echo "Type distribution in extended memory:"
sqlite3 /opt/sutazaiapp/data/mcp/extended-memory/extended_memory.db <<EOF
SELECT type, COUNT(*) as count
FROM memory_store
GROUP BY type
ORDER BY count DESC;
EOF