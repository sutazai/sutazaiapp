#!/bin/bash

# Strict error handling
set -euo pipefail


# Final Consul Registration Script - Using Container DNS Names
# This version uses Docker container names which are resolvable within the Docker network


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

echo "======================================"
echo "Consul Service Registration - Final"
echo "======================================"

# Clean up all previous registrations
echo "Cleaning up previous registrations..."
for service_id in $(curl -s http://localhost:10006/v1/agent/services | python3 -c "import json,sys; [print(k) for k in json.load(sys.stdin).keys()]"); do
    curl -s -X PUT http://localhost:10006/v1/agent/service/deregister/${service_id}
done
echo "✅ Cleanup complete"

# Load the configuration and register each service
echo ""
echo "Registering services from config..."
python3 <<'EOF'
import json
import requests
import time

# Load configuration
with open('/opt/sutazaiapp/config/consul-services-config.json', 'r') as f:
    config = json.load(f)

consul_url = "http://localhost:10006"
registered = []
failed = []

for service in config['services']:
    try:
        # Register the service
        response = requests.put(
            f"{consul_url}/v1/agent/service/register",
            json=service
        )
        
        if response.status_code == 200:
            registered.append(service['name'])
            print(f"✅ Registered: {service['name']} ({service['address']}:{service['port']})")
        else:
            failed.append(service['name'])
            print(f"❌ Failed: {service['name']} - Status {response.status_code}")
    except Exception as e:
        failed.append(service['name'])
        print(f"❌ Error registering {service['name']}: {str(e)}")

print("\n" + "="*40)
print(f"Registration Summary:")
print(f"  Successful: {len(registered)}")
print(f"  Failed: {len(failed)}")

# Wait for health checks
print("\nWaiting for health checks to run...")
time.sleep(5)

# Check health status
response = requests.get(f"{consul_url}/v1/health/state/any")
if response.status_code == 200:
    checks = response.json()
    services_health = {}
    
    for check in checks:
        service_name = check.get('ServiceName', '')
        if service_name and service_name != 'consul':
            status = check.get('Status', 'unknown')
            if service_name not in services_health or status == 'critical':
                services_health[service_name] = status
    
    print("\nService Health Status:")
    print("-" * 30)
    
    # Sort by status (passing first)
    for service, status in sorted(services_health.items(), key=lambda x: (x[1] != 'passing', x[0])):
        icon = "✅" if status == "passing" else "⚠️" if status == "warning" else "❌"
        print(f"  {icon} {service}: {status.upper()}")

print("\n🌐 Consul UI: http://localhost:10006/ui/dc1/services")
EOF

# Save the registration as a startup script
cat > /opt/sutazaiapp/scripts/consul-startup-registration.sh <<'STARTUP'
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
STARTUP

chmod +x /opt/sutazaiapp/scripts/consul-startup-registration.sh

echo ""
echo "✅ Registration complete!"
echo "✅ Startup script created at: /opt/sutazaiapp/scripts/consul-startup-registration.sh"