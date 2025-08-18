#!/bin/bash
set -e

# Check if service is responding
curl -f http://localhost:4000/health > /dev/null 2>&1

# Check memory usage
MEMORY_MB=$(cat /proc/meminfo | grep MemAvailable | awk '{print int($2/1024)}')
if [ $MEMORY_MB -lt 100 ]; then
    echo "Low memory warning: ${MEMORY_MB}MB available"
    exit 1
fi

# Check process count
PROCESS_COUNT=$(ps aux | grep -v grep | grep -c node || true)
if [ $PROCESS_COUNT -gt 5 ]; then
    echo "High process count warning: ${PROCESS_COUNT} processes"
    exit 1
fi

echo "Health check passed"