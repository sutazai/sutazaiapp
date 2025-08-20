#!/bin/bash

echo "=== Database Analysis Report ==="
echo "Date: $(date)"
echo ""

# Find all memory.db files
echo "Memory Databases Found:"
echo "======================="

for db in $(find /opt/sutazaiapp -name "memory.db" 2>/dev/null); do
    count=$(sqlite3 "$db" "SELECT COUNT(*) FROM memory_entries;" 2>/dev/null || echo "0")
    size=$(du -h "$db" | cut -f1)
    echo "$count entries, $size: $db"
done | sort -rn

echo ""
echo "Extended Memory Database:"
echo "========================"
db="/opt/sutazaiapp/data/mcp/extended-memory/extended_memory.db"
if [ -f "$db" ]; then
    count=$(sqlite3 "$db" "SELECT COUNT(*) FROM memory_store;" 2>/dev/null || echo "0")
    size=$(du -h "$db" | cut -f1)
    echo "$count entries, $size: $db"
fi

echo ""
echo "Other Databases:"
echo "==============="
for db in /opt/sutazaiapp/data/n8n/database.sqlite /opt/sutazaiapp/data/flowise/database.sqlite; do
    if [ -f "$db" ]; then
        size=$(du -h "$db" | cut -f1)
        echo "$size: $db"
    fi
done