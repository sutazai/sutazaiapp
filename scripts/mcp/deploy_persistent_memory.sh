#!/bin/bash
set -e

# Extended Memory MCP Service - Persistence Deployment Script
# This script safely migrates from in-memory to SQLite-based persistence

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SERVICE_DIR="$PROJECT_ROOT/docker/mcp-services/extended-memory-persistent"
DATA_DIR="/opt/sutazaiapp/data/mcp/extended-memory"
CONTAINER_NAME="mcp-extended-memory"
NEW_CONTAINER_NAME="mcp-extended-memory-persistent"

echo "================================================================"
echo "Extended Memory MCP Service - Persistence Deployment"
echo "================================================================"
echo "Project Root: $PROJECT_ROOT"
echo "Service Dir: $SERVICE_DIR"
echo "Data Dir: $DATA_DIR"
echo ""

# Function to check if container exists
container_exists() {
    docker ps -a --format '{{.Names}}' | grep -q "^$1$"
}

# Function to check if container is running
container_running() {
    docker ps --format '{{.Names}}' | grep -q "^$1$"
}

# Function to backup current data
backup_current_data() {
    echo "📦 Backing up current in-memory data..."
    
    if container_running "$CONTAINER_NAME"; then
        # Try to extract data from running container
        echo "  Attempting to extract data from running container..."
        
        # Check if the service has an endpoint to get all data
        if curl -s http://localhost:3009/list > /dev/null 2>&1; then
            # Save current data
            mkdir -p "$PROJECT_ROOT/backups/extended-memory"
            BACKUP_FILE="$PROJECT_ROOT/backups/extended-memory/backup_$(date +%Y%m%d_%H%M%S).json"
            
            echo "  Fetching all keys..."
            KEYS=$(curl -s http://localhost:3009/list | jq -r '.keys[]' 2>/dev/null || echo "")
            
            if [ ! -z "$KEYS" ]; then
                echo "{" > "$BACKUP_FILE"
                FIRST=true
                
                while IFS= read -r key; do
                    if [ "$FIRST" = true ]; then
                        FIRST=false
                    else
                        echo "," >> "$BACKUP_FILE"
                    fi
                    
                    VALUE=$(curl -s "http://localhost:3009/retrieve/$key" | jq '.value' 2>/dev/null || echo "null")
                    echo "  \"$key\": $VALUE" >> "$BACKUP_FILE"
                done <<< "$KEYS"
                
                echo "}" >> "$BACKUP_FILE"
                echo "  ✓ Data backed up to: $BACKUP_FILE"
            else
                echo "  ⚠ No data found in current service"
            fi
        else
            echo "  ⚠ Cannot connect to current service"
        fi
    else
        echo "  ℹ Current container not running, skipping backup"
    fi
}

# Step 1: Create data directory
echo "1️⃣ Creating data directory..."
sudo mkdir -p "$DATA_DIR"
sudo chown -R $(id -u):$(id -g) "$DATA_DIR"
echo "   ✓ Data directory created: $DATA_DIR"

# Step 2: Backup current data if available
backup_current_data

# Step 3: Build new image
echo ""
echo "2️⃣ Building new persistent image..."
cd "$SERVICE_DIR"
docker build -t sutazai-mcp-extended-memory:2.0.0 .
echo "   ✓ Image built successfully"

# Step 4: Stop current container if running
if container_running "$CONTAINER_NAME"; then
    echo ""
    echo "3️⃣ Stopping current in-memory container..."
    docker stop "$CONTAINER_NAME"
    echo "   ✓ Container stopped"
    
    # Rename old container for safety
    docker rename "$CONTAINER_NAME" "${CONTAINER_NAME}-old-$(date +%Y%m%d_%H%M%S)"
    echo "   ✓ Old container renamed for safety"
fi

# Step 5: Start new persistent container
echo ""
echo "4️⃣ Starting new persistent container..."
docker run -d \
    --name "$NEW_CONTAINER_NAME" \
    --network sutazai-network \
    -p 3009:3009 \
    -v "$DATA_DIR:/var/lib/mcp" \
    -v extended-memory-logs:/var/log/mcp \
    -e SERVICE_PORT=3009 \
    -e SQLITE_PATH=/var/lib/mcp/extended_memory.db \
    -e ENABLE_CACHE=true \
    --restart unless-stopped \
    sutazai-mcp-extended-memory:2.0.0

echo "   ✓ Container started"

# Step 6: Wait for service to be ready
echo ""
echo "5️⃣ Waiting for service to be ready..."
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:3009/health > /dev/null 2>&1; then
        echo "   ✓ Service is ready!"
        break
    fi
    
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "   Waiting... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "   ❌ Service failed to start"
    exit 1
fi

# Step 7: Verify persistence is enabled
echo ""
echo "6️⃣ Verifying persistence configuration..."
HEALTH_RESPONSE=$(curl -s http://localhost:3009/health)

if echo "$HEALTH_RESPONSE" | jq -e '.persistence.enabled == true' > /dev/null 2>&1; then
    echo "   ✓ Persistence is enabled"
    echo "   Database: $(echo "$HEALTH_RESPONSE" | jq -r '.persistence.path')"
    echo "   Type: $(echo "$HEALTH_RESPONSE" | jq -r '.persistence.type')"
else
    echo "   ❌ Persistence verification failed"
    exit 1
fi

# Step 8: Run basic tests
echo ""
echo "7️⃣ Running basic functionality tests..."

# Test store
TEST_KEY="deploy_test_$(date +%s)"
TEST_VALUE="Persistence deployment successful!"

echo "   Testing store operation..."
STORE_RESPONSE=$(curl -s -X POST http://localhost:3009/store \
    -H "Content-Type: application/json" \
    -d "{\"key\": \"$TEST_KEY\", \"value\": \"$TEST_VALUE\"}")

if echo "$STORE_RESPONSE" | jq -e '.status == "stored"' > /dev/null 2>&1; then
    echo "   ✓ Store operation successful"
else
    echo "   ❌ Store operation failed"
    exit 1
fi

# Test retrieve
echo "   Testing retrieve operation..."
RETRIEVE_RESPONSE=$(curl -s "http://localhost:3009/retrieve/$TEST_KEY")

if echo "$RETRIEVE_RESPONSE" | jq -e ".value == \"$TEST_VALUE\"" > /dev/null 2>&1; then
    echo "   ✓ Retrieve operation successful"
else
    echo "   ❌ Retrieve operation failed"
    exit 1
fi

# Test persistence by restarting container
echo "   Testing persistence across restart..."
docker restart "$NEW_CONTAINER_NAME"
sleep 5

# Wait for service to come back
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:3009/health > /dev/null 2>&1; then
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    sleep 2
done

# Verify data persisted
PERSIST_RESPONSE=$(curl -s "http://localhost:3009/retrieve/$TEST_KEY")
if echo "$PERSIST_RESPONSE" | jq -e ".value == \"$TEST_VALUE\"" > /dev/null 2>&1; then
    echo "   ✓ Data persisted across restart!"
else
    echo "   ❌ Data persistence failed"
    exit 1
fi

# Step 9: Display final status
echo ""
echo "================================================================"
echo "✅ Deployment Successful!"
echo "================================================================"
echo ""
echo "Service Status:"
curl -s http://localhost:3009/health | jq '.'
echo ""
echo "Statistics:"
curl -s http://localhost:3009/stats | jq '.statistics'
echo ""
echo "Access the service at: http://localhost:3009"
echo "API Documentation at: http://localhost:3009/docs"
echo ""
echo "To run comprehensive tests:"
echo "  python3 $PROJECT_ROOT/tests/mcp/test_extended_memory_persistence.py"
echo ""
echo "================================================================"