#!/bin/bash
# MCP Process Monitor

while true; do
    # Count MCP processes
    PROCESS_COUNT=$(ps aux | grep -E "mcp|npx" | grep -v grep | wc -l)
    
    # If too many processes, clean up
    if [ $PROCESS_COUNT -gt 50 ]; then
        echo "[$(date)] Warning: $PROCESS_COUNT MCP processes detected, cleaning up..."
        
        # Kill old npx processes
        for pid in $(ps aux | grep "npx" | grep -v grep | awk '{print $2}' | tail -n +20); do
            kill -9 $pid 2>/dev/null
        done
    fi
    
    # Sleep for 5 minutes
    sleep 300
done
