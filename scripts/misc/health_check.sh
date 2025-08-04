#!/bin/bash
# Health check script for service adapter

# Check if the adapter is responding
response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:${ADAPTER_PORT:-8080}/health)

if [ "$response" = "200" ]; then
    exit 0
else
    exit 1
fi