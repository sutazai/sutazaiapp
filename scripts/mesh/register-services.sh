#!/bin/bash

# Register all SutazAI services with Consul

CONSUL_URL="http://localhost:10006"

# Function to register a service
register_service() {
    local name=$1
    local port=$2
    local tags=$3
    
    echo "Registering $name on port $port..."
    
    curl -X PUT "$CONSUL_URL/v1/agent/service/register" \
        -H "Content-Type: application/json" \
        -d @- <<EOF
{
    "ID": "${name}",
    "Name": "${name}",
    "Tags": ${tags},
    "Port": ${port},
    "Check": {
        "HTTP": "http://localhost:${port}/health",
        "Interval": "30s",
        "Timeout": "5s"
    }
}
EOF
    
    if [ $? -eq 0 ]; then
        echo "✓ ${name} registered"
    else
        echo "✗ Failed to register ${name}"
    fi
}

# Register core services
register_service "postgres" 10000 '["database","core"]'
register_service "redis" 10001 '["cache","core"]'
register_service "neo4j" 10002 '["graph","database"]'
register_service "backend-api" 10010 '["api","backend"]'
register_service "chromadb" 10100 '["vectordb","storage"]'
register_service "qdrant" 10101 '["vectordb","storage"]'

# Register mesh services
register_service "kong" 10005 '["gateway","mesh"]'
register_service "consul" 10006 '["discovery","mesh"]'

echo ""
echo "Services registration complete. Checking status..."
echo ""

# List registered services
curl -s "$CONSUL_URL/v1/agent/services" | python3 -m json.tool | grep '"Service"' | cut -d'"' -f4 | sort | uniq

echo ""
echo "Total services registered: $(curl -s "$CONSUL_URL/v1/agent/services" | python3 -m json.tool | grep -c '"Service"')"