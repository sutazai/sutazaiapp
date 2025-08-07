#!/bin/bash
# ARCHIVED COPY — moved from tests/integration/test-service-mesh.sh
# See docs/decisions/2025-08-07-remove-service-mesh.md

#!/bin/bash
# Purpose: Test service mesh connectivity and load balancing
# Usage: ./test-service-mesh.sh
# Requires: Kong, Consul, and services running

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=== Service Mesh Testing Suite ==="
echo "Testing connectivity, service discovery, and load balancing..."
echo ""

# Function to print status
print_status() {
    local status=$1
    local message=$2
    if [ "$status" = "success" ]; then
        echo -e "${GREEN}✓${NC} $message"
    elif [ "$status" = "warning" ]; then
        echo -e "${YELLOW}⚠${NC} $message"
    else
        echo -e "${RED}✗${NC} $message"
    fi
}

# Function to test HTTP endpoint
test_endpoint() {
    local name=$1
    local url=$2
    local expected_code=${3:-200}
    
    response=$(curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null || echo "000")
    
    if [ "$response" = "$expected_code" ]; then
        print_status "success" "$name: HTTP $response"
        return 0
    else
        print_status "error" "$name: HTTP $response (expected $expected_code)"
        return 1
    fi
}

# Function to test TCP connectivity
test_tcp() {
    local name=$1
    local host=$2
    local port=$3
    
    if timeout 2 bash -c "echo > /dev/tcp/$host/$port" 2>/dev/null; then
        print_status "success" "$name: TCP port $port is open"
        return 0
    else
        print_status "error" "$name: TCP port $port is closed"
        return 1
    fi
}

