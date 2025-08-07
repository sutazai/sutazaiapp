#!/bin/bash

set -e

echo "Starting SutazAI RBAC Policy Engine..."

# Wait for dependencies
echo "Waiting for database..."
while ! nc -z postgres 5432; do
  sleep 1
done
echo "Database is ready!"

echo "Waiting for Redis..."
while ! nc -z redis 6379; do
  sleep 1
done
echo "Redis is ready!"

echo "Waiting for Keycloak..."
while ! curl -f http://keycloak:8080/health/ready; do
  sleep 5
done
echo "Keycloak is ready!"

# Initialize policy file if it doesn't exist
if [ ! -f "/app/policies/policy.csv" ]; then
    echo "Creating initial policy file..."
    mkdir -p /app/policies
    cat > /app/policies/policy.csv << 'EOF'
p, role:admin, *, *
p, role:agent, api:agents, read
p, role:agent, api:agents, write
p, role:agent, ollama:*, read
p, role:developer, api:*, read
g, agent-orchestrator, role:admin
g, ai-system-validator, role:system
EOF
    echo "Initial policy file created!"
fi

# Start the application
echo "Starting RBAC Policy Engine..."
exec python main.py