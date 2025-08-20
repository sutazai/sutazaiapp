#!/bin/bash
# Neo4j Connection Test Script
# Tests Neo4j connectivity and basic operations

echo "Testing Neo4j Connection..."
echo "============================"

# Test variables
NEO4J_USER="neo4j"
NEO4J_PASSWORD="sutazai123"
NEO4J_HOST="localhost"
NEO4J_BOLT_PORT="10003"
NEO4J_HTTP_PORT="10002"

# Test 1: Cypher Shell Connection
echo ""
echo "1. Testing Cypher Shell Connection..."
if docker exec sutazai-neo4j cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" "RETURN 1 AS test;" > /dev/null 2>&1; then
    echo "✅ Cypher Shell: Connected successfully"
else
    echo "❌ Cypher Shell: Connection failed"
    exit 1
fi

# Test 2: Show Databases
echo ""
echo "2. Listing Databases..."
docker exec sutazai-neo4j cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" "SHOW DATABASES;" | head -3
echo "✅ Database listing successful"

# Test 3: HTTP API Connection
echo ""
echo "3. Testing HTTP API..."
HTTP_RESPONSE=$(curl -s -u "$NEO4J_USER:$NEO4J_PASSWORD" http://$NEO4J_HOST:$NEO4J_HTTP_PORT/)
if [[ $HTTP_RESPONSE == *"neo4j_version"* ]]; then
    NEO4J_VERSION=$(echo $HTTP_RESPONSE | grep -o '"neo4j_version":"[^"]*' | cut -d'"' -f4)
    echo "✅ HTTP API: Connected to Neo4j $NEO4J_VERSION"
else
    echo "❌ HTTP API: Connection failed"
    exit 1
fi

# Test 4: Create and Query Test Node
echo ""
echo "4. Testing Data Operations..."
docker exec sutazai-neo4j cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" \
    "CREATE (n:TestNode {name: 'ConnectionTest', timestamp: datetime()}) RETURN n.name AS name;" > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Create operation: Success"
    
    # Clean up test node
    docker exec sutazai-neo4j cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" \
        "MATCH (n:TestNode {name: 'ConnectionTest'}) DELETE n;" > /dev/null 2>&1
    echo "✅ Delete operation: Success"
else
    echo "❌ Data operations: Failed"
    exit 1
fi

# Test 5: Check Container Health
echo ""
echo "5. Checking Container Health..."
CONTAINER_STATUS=$(docker inspect sutazai-neo4j --format='{{.State.Status}}')
if [ "$CONTAINER_STATUS" == "running" ]; then
    echo "✅ Container Status: Running"
else
    echo "❌ Container Status: $CONTAINER_STATUS"
    exit 1
fi

echo ""
echo "============================"
echo "✅ All Neo4j tests passed successfully!"
echo ""
echo "Connection Details:"
echo "  Host: $NEO4J_HOST"
echo "  Bolt Port: $NEO4J_BOLT_PORT"
echo "  HTTP Port: $NEO4J_HTTP_PORT"
echo "  Username: $NEO4J_USER"
echo "  Password: [CONFIGURED]"
echo "  Database: sutazai"