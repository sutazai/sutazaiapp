#!/bin/bash

# Strict error handling
set -euo pipefail

# Auto-registration script for Consul services
# Can be added to container startup or cron


# Signal handlers for graceful shutdown
cleanup_and_exit() {
    local exit_code="${1:-0}"
    echo "Script interrupted, cleaning up..." >&2
    # Clean up any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    exit "$exit_code"
}

trap 'cleanup_and_exit 130' INT
trap 'cleanup_and_exit 143' TERM
trap 'cleanup_and_exit 1' ERR

if curl -s http://localhost:10006/v1/agent/self > /dev/null 2>&1; then
    python3 -c "
import json, requests
with open('/opt/sutazaiapp/config/consul-services-config.json', 'r') as f:
    config = json.load(f)
for service in config['services']:
    requests.put('http://localhost:10006/v1/agent/service/register', json=service)
print('Consul services registered')
"
fi
