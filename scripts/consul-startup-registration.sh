#!/bin/bash
# Auto-registration script for Consul services
# Can be added to container startup or cron

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
