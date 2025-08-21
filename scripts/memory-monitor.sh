#!/bin/bash

MEMORY_USAGE=$(free | grep '^Mem:' | awk '{printf "%.1f", ($3/$2) * 100.0}')
SWAP_USAGE=$(free | grep '^Swap:' | awk '{printf "%.1f", ($3/$2) * 100.0}')

if (( $(echo "$MEMORY_USAGE > 90" | bc -l) )); then
    echo "CRITICAL: Memory usage at ${MEMORY_USAGE}% - triggering cleanup"
    /opt/sutazaiapp/scripts/memory-optimization.sh --emergency
fi

if (( $(echo "$SWAP_USAGE > 50" | bc -l) )); then
    echo "WARNING: Swap usage at ${SWAP_USAGE}%"
fi

# Log memory usage
echo "$(date '+%Y-%m-%d %H:%M:%S') - Memory: ${MEMORY_USAGE}%, Swap: ${SWAP_USAGE}%" >> /var/log/memory-usage.log
