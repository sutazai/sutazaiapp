#!/bin/bash
# Comprehensive System Audit Script for SutazAI Platform
# Validates all components, services, dependencies, and integrations

set -e

echo "============================================"
echo "SutazAI Platform Comprehensive System Audit"
echo "============================================"
echo "Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Audit results file
AUDIT_REPORT="/opt/sutazaiapp/audit_report_$(date +%Y%m%d_%H%M%S).md"

# Initialize report
cat > "$AUDIT_REPORT" << EOF
# SutazAI Platform System Audit Report
**Generated**: $(date -u '+%Y-%m-%d %H:%M:%S UTC')

## Executive Summary

EOF

# Function to log results
log_result() {
    local category=$1
    local test=$2
    local status=$3
    local details=$4
    
    echo -e "${status} ${category}: ${test}"
    echo "- [${status}] **${category}**: ${test} - ${details}" >> "$AUDIT_REPORT"
}

# Function to check service health
check_service() {
    local name=$1
    local port=$2
    local endpoint=${3:-"/"}
    
    if curl -s -o /dev/null -w "%{http_code}" "http://localhost:${port}${endpoint}" | grep -q "200\|301\|302"; then
        log_result "Service" "$name (Port $port)" "✅" "Healthy and responding"
        return 0
    else
        log_result "Service" "$name (Port $port)" "❌" "Not responding or unhealthy"
        return 1
    fi
}

# Function to check container status
check_container() {
    local name=$1
    
    if docker ps --format "{{.Names}}" | grep -q "^${name}$"; then
        status=$(docker ps --format "{{.Status}}" --filter "name=${name}")
        if echo "$status" | grep -q "healthy\|Up"; then
            log_result "Container" "$name" "✅" "$status"
            return 0
        else
            log_result "Container" "$name" "⚠️" "$status"
            return 1
        fi
    else
        log_result "Container" "$name" "❌" "Not running"
        return 1
    fi
}

echo "## 1. Infrastructure Services Audit" | tee -a "$AUDIT_REPORT"
echo "" | tee -a "$AUDIT_REPORT"

# Check core infrastructure
echo -e "${BLUE}Checking Infrastructure Services...${NC}"
check_container "sutazai-postgres"
check_container "sutazai-redis"
check_container "sutazai-neo4j"
check_container "sutazai-rabbitmq"
check_container "sutazai-consul"
check_container "sutazai-kong"
check_container "sutazai-chromadb"
check_container "sutazai-qdrant"
check_container "sutazai-faiss"
check_container "sutazai-backend"

echo ""
echo "## 2. AI Agents Status" | tee -a "$AUDIT_REPORT"
echo "" | tee -a "$AUDIT_REPORT"

# Count and check AI agents
echo -e "${BLUE}Checking AI Agents...${NC}"
agent_count=$(docker ps --format "{{.Names}}" | grep -E "sutazai-(crewai|aider|letta|gpt-engineer|finrobot|documind|shellgpt|langchain|autogpt|localagi|agentzero|bigagi|semgrep|autogen|browseruse|skyvern)" | wc -l)
echo "Total AI Agents Running: $agent_count" | tee -a "$AUDIT_REPORT"

# Check each agent's health endpoint
for port in 11401 11301 11101 11302 11601 11701 11502 11201 11102 11103 11105 11106 11801 11203 11703 11702; do
    response=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$port/health" 2>/dev/null || echo "000")
    if [ "$response" = "200" ]; then
        echo -e "${GREEN}✅${NC} Port $port: Healthy"
        echo "- ✅ Port $port: Healthy" >> "$AUDIT_REPORT"
    else
        echo -e "${YELLOW}⚠️${NC} Port $port: Not responding (HTTP $response)"
        echo "- ⚠️ Port $port: Not responding (HTTP $response)" >> "$AUDIT_REPORT"
    fi
done

echo ""
echo "## 3. System Resources" | tee -a "$AUDIT_REPORT"
echo "" | tee -a "$AUDIT_REPORT"

# Check system resources
echo -e "${BLUE}Checking System Resources...${NC}"
echo "### Memory Usage" | tee -a "$AUDIT_REPORT"
free -h | tee -a "$AUDIT_REPORT"
echo "" | tee -a "$AUDIT_REPORT"

echo "### Docker Resource Usage" | tee -a "$AUDIT_REPORT"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" | head -20 | tee -a "$AUDIT_REPORT"

echo ""
echo "## 4. Network Connectivity" | tee -a "$AUDIT_REPORT"
echo "" | tee -a "$AUDIT_REPORT"

# Check network
echo -e "${BLUE}Checking Network Configuration...${NC}"
docker network ls | grep sutazai | tee -a "$AUDIT_REPORT"
echo "" | tee -a "$AUDIT_REPORT"

# Check inter-container connectivity
echo "### Inter-container Connectivity Test" | tee -a "$AUDIT_REPORT"
if docker exec sutazai-backend ping -c 1 sutazai-postgres > /dev/null 2>&1; then
    log_result "Network" "Backend->PostgreSQL" "✅" "Connection successful"
else
    log_result "Network" "Backend->PostgreSQL" "❌" "Connection failed"
fi

if docker exec sutazai-backend ping -c 1 sutazai-redis > /dev/null 2>&1; then
    log_result "Network" "Backend->Redis" "✅" "Connection successful"
else
    log_result "Network" "Backend->Redis" "❌" "Connection failed"
fi

echo ""
echo "## 5. Database Connectivity" | tee -a "$AUDIT_REPORT"
echo "" | tee -a "$AUDIT_REPORT"

# Test PostgreSQL
echo -e "${BLUE}Testing Database Connections...${NC}"
if PGPASSWORD=sutazai_secure_2024 psql -h localhost -p 10000 -U jarvis -d jarvis_ai -c "SELECT 1;" > /dev/null 2>&1; then
    log_result "Database" "PostgreSQL" "✅" "Connection successful"
else
    log_result "Database" "PostgreSQL" "❌" "Connection failed"
fi

# Test Redis
if redis-cli -h localhost -p 10001 ping > /dev/null 2>&1; then
    log_result "Database" "Redis" "✅" "Connection successful"
else
    log_result "Database" "Redis" "❌" "Connection failed"
fi

# Test Neo4j
if curl -s -u neo4j:sutazai_secure_2024 http://localhost:10002 > /dev/null 2>&1; then
    log_result "Database" "Neo4j" "✅" "Connection successful"
else
    log_result "Database" "Neo4j" "❌" "Connection failed"
fi

echo ""
echo "## 6. Ollama LLM Status" | tee -a "$AUDIT_REPORT"
echo "" | tee -a "$AUDIT_REPORT"

# Check Ollama
echo -e "${BLUE}Checking Ollama Status...${NC}"
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    models=$(curl -s http://localhost:11434/api/tags | jq -r '.models[].name' 2>/dev/null | tr '\n' ', ' | sed 's/,$//')
    log_result "Ollama" "Service" "✅" "Running with models: $models"
else
    log_result "Ollama" "Service" "❌" "Not running"
fi

echo ""
echo "## 7. API Endpoints Test" | tee -a "$AUDIT_REPORT"
echo "" | tee -a "$AUDIT_REPORT"

# Test backend API
echo -e "${BLUE}Testing API Endpoints...${NC}"
if curl -s http://localhost:10200/api/v1/health | jq . > /dev/null 2>&1; then
    log_result "API" "Backend Health" "✅" "Responding correctly"
else
    log_result "API" "Backend Health" "❌" "Not responding"
fi

# Test Kong API Gateway
if curl -s http://localhost:10009 > /dev/null 2>&1; then
    log_result "API" "Kong Admin" "✅" "Admin API accessible"
else
    log_result "API" "Kong Admin" "❌" "Admin API not accessible"
fi

echo ""
echo "## 8. Security Audit" | tee -a "$AUDIT_REPORT"
echo "" | tee -a "$AUDIT_REPORT"

echo -e "${BLUE}Running Security Checks...${NC}"

# Check for exposed ports
echo "### Exposed Ports" | tee -a "$AUDIT_REPORT"
netstat -tuln | grep LISTEN | grep -E "0.0.0.0|:::" | tee -a "$AUDIT_REPORT"

# Check for default passwords in environment
echo "### Environment Security" | tee -a "$AUDIT_REPORT"
if grep -r "password\|secret\|key" /opt/sutazaiapp/.env 2>/dev/null | grep -v "^#"; then
    log_result "Security" "Credentials" "⚠️" "Found potential exposed credentials"
else
    log_result "Security" "Credentials" "✅" "No exposed credentials in .env"
fi

echo ""
echo "## 9. File System Audit" | tee -a "$AUDIT_REPORT"
echo "" | tee -a "$AUDIT_REPORT"

echo -e "${BLUE}Checking File System...${NC}"

# Check critical directories
for dir in /opt/sutazaiapp/agents /opt/sutazaiapp/backend /opt/sutazaiapp/frontend /opt/sutazaiapp/scripts; do
    if [ -d "$dir" ]; then
        file_count=$(find "$dir" -type f | wc -l)
        log_result "FileSystem" "$dir" "✅" "$file_count files"
    else
        log_result "FileSystem" "$dir" "❌" "Directory missing"
    fi
done

# Check for orphaned containers
echo ""
echo "### Orphaned Containers" | tee -a "$AUDIT_REPORT"
orphaned=$(docker ps -a --filter "status=exited" --format "{{.Names}}" | grep sutazai- | wc -l)
if [ "$orphaned" -gt 0 ]; then
    log_result "Docker" "Orphaned Containers" "⚠️" "$orphaned found"
    docker ps -a --filter "status=exited" --format "table {{.Names}}\t{{.Status}}" | grep sutazai- | tee -a "$AUDIT_REPORT"
else
    log_result "Docker" "Orphaned Containers" "✅" "None found"
fi

echo ""
echo "## 10. Performance Metrics" | tee -a "$AUDIT_REPORT"
echo "" | tee -a "$AUDIT_REPORT"

echo -e "${BLUE}Collecting Performance Metrics...${NC}"

# Response time test for key services
echo "### Service Response Times" | tee -a "$AUDIT_REPORT"
for service in "localhost:10200/api/v1/health" "localhost:11401/health" "localhost:11102/health"; do
    if time=$(curl -o /dev/null -s -w '%{time_total}\n' "http://$service" 2>/dev/null); then
        echo "- $service: ${time}s" | tee -a "$AUDIT_REPORT"
    fi
done

echo ""
echo "## Summary" | tee -a "$AUDIT_REPORT"
echo "" | tee -a "$AUDIT_REPORT"

# Count results
total_tests=$(grep -c "^\- \[" "$AUDIT_REPORT")
passed=$(grep -c "^\- \[✅\]" "$AUDIT_REPORT")
warnings=$(grep -c "^\- \[⚠️\]" "$AUDIT_REPORT") 
failed=$(grep -c "^\- \[❌\]" "$AUDIT_REPORT")

echo "### Audit Results Summary" | tee -a "$AUDIT_REPORT"
echo "- Total Tests: $total_tests" | tee -a "$AUDIT_REPORT"
echo "- Passed: $passed" | tee -a "$AUDIT_REPORT"
echo "- Warnings: $warnings" | tee -a "$AUDIT_REPORT"
echo "- Failed: $failed" | tee -a "$AUDIT_REPORT"
echo "" | tee -a "$AUDIT_REPORT"

if [ "$failed" -eq 0 ]; then
    echo -e "${GREEN}✅ System audit completed successfully!${NC}"
    echo "**Status: HEALTHY** ✅" >> "$AUDIT_REPORT"
elif [ "$failed" -lt 5 ]; then
    echo -e "${YELLOW}⚠️ System audit completed with minor issues${NC}"
    echo "**Status: NEEDS ATTENTION** ⚠️" >> "$AUDIT_REPORT"
else
    echo -e "${RED}❌ System audit found critical issues${NC}"
    echo "**Status: CRITICAL** ❌" >> "$AUDIT_REPORT"
fi

echo ""
echo "Full audit report saved to: $AUDIT_REPORT"
echo ""

# Generate recommendations
echo "## Recommendations" | tee -a "$AUDIT_REPORT"
echo "" | tee -a "$AUDIT_REPORT"

if [ "$failed" -gt 0 ]; then
    echo "### Critical Actions Required:" | tee -a "$AUDIT_REPORT"
    grep "^\- \[❌\]" "$AUDIT_REPORT" | while read -r line; do
        echo "  $line" | tee -a "$AUDIT_REPORT"
    done
fi

if [ "$warnings" -gt 0 ]; then
    echo "### Warnings to Address:" | tee -a "$AUDIT_REPORT"
    grep "^\- \[⚠️\]" "$AUDIT_REPORT" | while read -r line; do
        echo "  $line" | tee -a "$AUDIT_REPORT"
    done
fi

echo ""
echo "---" | tee -a "$AUDIT_REPORT"
echo "*Audit completed at $(date -u '+%Y-%m-%d %H:%M:%S UTC')*" | tee -a "$AUDIT_REPORT"