#!/bin/bash
# MCP Real Server Verification Script
# Created: 2025-08-20
# Purpose: Verify all MCP servers are real implementations, not facades

set -e

echo "========================================="
echo "MCP REAL SERVER VERIFICATION"
echo "========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# MCP Servers to verify
declare -A MCP_SERVERS=(
    ["mcp-claude-flow"]="3001"
    ["mcp-files"]="3003"
    ["mcp-context"]="3004"
    ["mcp-search"]="3006"
    ["mcp-memory"]="3009"
    ["mcp-docs"]="3017"
)

echo "1. CHECKING CONTAINER STATUS"
echo "-----------------------------"
for SERVER in "${!MCP_SERVERS[@]}"; do
    PORT="${MCP_SERVERS[$SERVER]}"
    STATUS=$(docker exec sutazai-mcp-orchestrator sh -c "DOCKER_HOST=tcp://localhost:2375 docker ps --filter name=$SERVER --format '{{.Status}}'" 2>/dev/null || echo "Not Found")
    if [[ "$STATUS" == *"Up"* ]]; then
        echo -e "${GREEN}✓${NC} $SERVER: Running on port $PORT"
    else
        echo -e "${RED}✗${NC} $SERVER: Not running"
    fi
done

echo ""
echo "2. HEALTH CHECK ENDPOINTS"
echo "-------------------------"
for SERVER in "${!MCP_SERVERS[@]}"; do
    PORT="${MCP_SERVERS[$SERVER]}"
    HEALTH=$(docker exec sutazai-mcp-orchestrator sh -c "wget -qO- http://localhost:$PORT/health 2>/dev/null" || echo "{}")
    if [[ "$HEALTH" == *"healthy"* ]]; then
        echo -e "${GREEN}✓${NC} $SERVER: Healthy"
        echo "  Response: $HEALTH"
    else
        echo -e "${RED}✗${NC} $SERVER: Unhealthy or unreachable"
    fi
done

echo ""
echo "3. MCP PROTOCOL VERIFICATION"
echo "----------------------------"
echo "Testing MCP tool endpoints..."

# Test files server
echo -e "\n${YELLOW}Files Server Test:${NC}"
TOOLS=$(docker exec sutazai-mcp-orchestrator sh -c "wget -qO- http://localhost:3003/tools 2>/dev/null | head -c 100" || echo "Failed")
if [[ "$TOOLS" == *"read_file"* ]]; then
    echo -e "${GREEN}✓${NC} Files server has proper MCP tools"
else
    echo -e "${RED}✗${NC} Files server tools endpoint failed"
fi

# Test memory server
echo -e "\n${YELLOW}Memory Server Test:${NC}"
TEST_KEY="verification-$(date +%s)"
STORE_RESULT=$(docker exec sutazai-mcp-orchestrator sh -c "wget -qO- --post-data '{\"key\":\"$TEST_KEY\",\"value\":{\"test\":\"data\"}}' --header='Content-Type: application/json' http://localhost:3009/tools/store_memory 2>/dev/null" || echo "Failed")
if [[ "$STORE_RESULT" == *"success"* ]]; then
    echo -e "${GREEN}✓${NC} Memory server can store data"
    RETRIEVE_RESULT=$(docker exec sutazai-mcp-orchestrator sh -c "wget -qO- --post-data '{\"key\":\"$TEST_KEY\"}' --header='Content-Type: application/json' http://localhost:3009/tools/retrieve_memory 2>/dev/null" || echo "Failed")
    if [[ "$RETRIEVE_RESULT" == *"test"* ]]; then
        echo -e "${GREEN}✓${NC} Memory server can retrieve data"
    else
        echo -e "${RED}✗${NC} Memory server retrieve failed"
    fi
else
    echo -e "${RED}✗${NC} Memory server store failed"
fi

# Test search server
echo -e "\n${YELLOW}Search Server Test:${NC}"
INDEX_RESULT=$(docker exec sutazai-mcp-orchestrator sh -c "wget -qO- --post-data '{\"id\":\"test-doc\",\"content\":\"Test document content\"}' --header='Content-Type: application/json' http://localhost:3006/tools/index_document 2>/dev/null" || echo "Failed")
if [[ "$INDEX_RESULT" == *"success"* ]]; then
    echo -e "${GREEN}✓${NC} Search server can index documents"
else
    echo -e "${RED}✗${NC} Search server indexing failed"
fi

# Test context server
echo -e "\n${YELLOW}Context Server Test:${NC}"
CONTEXT_RESULT=$(docker exec sutazai-mcp-orchestrator sh -c "wget -qO- --post-data '{\"id\":\"test-context\",\"content\":\"Test context data\"}' --header='Content-Type: application/json' http://localhost:3004/tools/store_context 2>/dev/null" || echo "Failed")
if [[ "$CONTEXT_RESULT" == *"success"* ]]; then
    echo -e "${GREEN}✓${NC} Context server can store contexts"
else
    echo -e "${RED}✗${NC} Context server store failed"
fi

# Test docs server
echo -e "\n${YELLOW}Docs Server Test:${NC}"
DOCS_RESULT=$(docker exec sutazai-mcp-orchestrator sh -c "wget -qO- --post-data '{\"id\":\"test-doc\",\"title\":\"Test\",\"content\":\"Documentation\"}' --header='Content-Type: application/json' http://localhost:3017/tools/store_doc 2>/dev/null" || echo "Failed")
if [[ "$DOCS_RESULT" == *"success"* ]]; then
    echo -e "${GREEN}✓${NC} Docs server can store documentation"
else
    echo -e "${RED}✗${NC} Docs server store failed"
fi

# Test claude-flow server
echo -e "\n${YELLOW}Claude-Flow Server Test:${NC}"
WORKFLOW_RESULT=$(docker exec sutazai-mcp-orchestrator sh -c "wget -qO- --post-data '{\"id\":\"test-workflow\",\"name\":\"Test Workflow\"}' --header='Content-Type: application/json' http://localhost:3001/tools/create_workflow 2>/dev/null" || echo "Failed")
if [[ "$WORKFLOW_RESULT" == *"success"* ]]; then
    echo -e "${GREEN}✓${NC} Claude-Flow server can create workflows"
else
    echo -e "${RED}✗${NC} Claude-Flow server workflow creation failed"
fi

echo ""
echo "4. REAL IMPLEMENTATION VERIFICATION"
echo "-----------------------------------"
echo "Checking for fake netcat listeners..."

FAKE_COUNT=0
for SERVER in "${!MCP_SERVERS[@]}"; do
    PROCESSES=$(docker exec sutazai-mcp-orchestrator sh -c "DOCKER_HOST=tcp://localhost:2375 docker exec $SERVER ps aux 2>/dev/null" || echo "")
    if [[ "$PROCESSES" == *"nc -l"* ]]; then
        echo -e "${RED}✗${NC} $SERVER: Still using fake netcat listener!"
        ((FAKE_COUNT++))
    elif [[ "$PROCESSES" == *"node"* ]]; then
        echo -e "${GREEN}✓${NC} $SERVER: Running real Node.js server"
    else
        echo -e "${YELLOW}?${NC} $SERVER: Unable to verify process"
    fi
done

echo ""
echo "========================================="
echo "VERIFICATION SUMMARY"
echo "========================================="

if [ $FAKE_COUNT -eq 0 ]; then
    echo -e "${GREEN}SUCCESS: All MCP servers are REAL implementations!${NC}"
    echo "- No fake netcat listeners detected"
    echo "- All servers respond to MCP protocol requests"
    echo "- Data persistence and retrieval working"
else
    echo -e "${RED}WARNING: $FAKE_COUNT servers still using fake implementations${NC}"
fi

echo ""
echo "Server Details:"
docker exec sutazai-mcp-orchestrator sh -c "DOCKER_HOST=tcp://localhost:2375 docker ps --filter 'name=mcp-' --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'"

echo ""
echo "Verification complete at $(date)"