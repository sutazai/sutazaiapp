#!/bin/bash

# Performance Baseline Script for SutazAI
# Captures current system performance metrics for optimization tracking

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Create reports directory
REPORT_DIR="/opt/sutazaiapp/reports/performance"
mkdir -p "$REPORT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="$REPORT_DIR/baseline_${TIMESTAMP}.json"

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}🚀 SutazAI Performance Baseline Capture${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo

# Function to test API endpoint performance
test_endpoint() {
    local endpoint=$1
    local name=$2
    local method=${3:-GET}
    local data=${4:-""}
    
    echo -e "${YELLOW}Testing $name...${NC}"
    
    if [ "$method" = "POST" ] && [ -n "$data" ]; then
        response_time=$(curl -o /dev/null -s -w '%{time_total}' -X POST \
            -H "Content-Type: application/json" \
            -d "$data" \
            "http://localhost:8000$endpoint")
    else
        response_time=$(curl -o /dev/null -s -w '%{time_total}' "http://localhost:8000$endpoint")
    fi
    
    echo "$response_time"
}

# Start JSON report
cat > "$REPORT_FILE" << EOF
{
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "system": {
EOF

# Capture system metrics
echo -e "${GREEN}📊 Capturing System Metrics${NC}"
cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1 || echo "5.0")
memory_info=$(free -b | grep Mem)
memory_total=$(echo $memory_info | awk '{print $2}')
memory_used=$(echo $memory_info | awk '{print $3}')
memory_percent=$(echo "scale=2; $memory_used * 100 / $memory_total" | bc)
load_average=$(uptime | awk -F'load average:' '{print $2}')

cat >> "$REPORT_FILE" << EOF
    "cpu_usage_percent": $cpu_usage,
    "memory_total_bytes": $memory_total,
    "memory_used_bytes": $memory_used,
    "memory_percent": $memory_percent,
    "load_average": "$load_average"
  },
  "containers": {
EOF

# Capture container metrics
echo -e "${GREEN}🐳 Capturing Container Metrics${NC}"
docker stats --no-stream --format "{{json .}}" | jq -s '.' > /tmp/docker_stats.json

# Add container data to report
echo '    "stats": ' >> "$REPORT_FILE"
cat /tmp/docker_stats.json >> "$REPORT_FILE"
echo '  },' >> "$REPORT_FILE"

# API Performance Tests
echo -e "${GREEN}⚡ Testing API Performance${NC}"
echo '  "api_performance": {' >> "$REPORT_FILE"

# Test endpoints
health_time=$(test_endpoint "/health" "Health Check")
echo "    \"health_check_ms\": $(echo "$health_time * 1000" | bc)," >> "$REPORT_FILE"

docs_time=$(test_endpoint "/docs" "API Documentation")
echo "    \"docs_load_ms\": $(echo "$docs_time * 1000" | bc)," >> "$REPORT_FILE"

agents_time=$(test_endpoint "/api/v1/agents" "List Agents")
echo "    \"list_agents_ms\": $(echo "$agents_time * 1000" | bc)," >> "$REPORT_FILE"

# Test model inference if available
echo -e "${YELLOW}Testing Ollama inference...${NC}"
inference_start=$(date +%s.%N)
inference_response=$(curl -s -X POST http://localhost:11434/api/generate \
    -d '{
        "model": "gpt-oss:latest",
        "prompt": "Hello, how are you?",
        "stream": false,
        "options": {"num_predict": 10}
    }' 2>/dev/null || echo '{}')
inference_end=$(date +%s.%N)
inference_time=$(echo "$inference_end - $inference_start" | bc)

if [ -n "$inference_response" ] && [ "$inference_response" != "{}" ]; then
    echo "    \"ollama_inference_ms\": $(echo "$inference_time * 1000" | bc)," >> "$REPORT_FILE"
else
    echo "    \"ollama_inference_ms\": null," >> "$REPORT_FILE"
fi

# Database query performance
echo -e "${YELLOW}Testing database performance...${NC}"
db_start=$(date +%s.%N)
docker exec sutazai-postgres-minimal psql -U sutazai -d sutazai -c "SELECT 1;" >/dev/null 2>&1
db_end=$(date +%s.%N)
db_time=$(echo "$db_end - $db_start" | bc)
echo "    \"postgres_query_ms\": $(echo "$db_time * 1000" | bc)," >> "$REPORT_FILE"

# Redis performance
redis_start=$(date +%s.%N)
docker exec sutazai-redis-minimal redis-cli -a redis_password ping >/dev/null 2>&1
redis_end=$(date +%s.%N)
redis_time=$(echo "$redis_end - $redis_start" | bc)
echo "    \"redis_ping_ms\": $(echo "$redis_time * 1000" | bc)" >> "$REPORT_FILE"

echo '  },' >> "$REPORT_FILE"

# Service availability
echo -e "${GREEN}🏥 Checking Service Availability${NC}"
echo '  "service_availability": {' >> "$REPORT_FILE"

services=("postgres:5432" "redis:6379" "backend:8000" "frontend:8501" "ollama:11434")
available_count=0
total_count=${#services[@]}

for service in "${services[@]}"; do
    IFS=':' read -r host port <<< "$service"
    if nc -z localhost $port 2>/dev/null; then
        echo "    \"$host\": true," >> "$REPORT_FILE"
        ((available_count++))
    else
        echo "    \"$host\": false," >> "$REPORT_FILE"
    fi
done

availability_percent=$(echo "scale=2; $available_count * 100 / $total_count" | bc)
echo "    \"availability_percent\": $availability_percent" >> "$REPORT_FILE"

echo '  },' >> "$REPORT_FILE"

# Summary metrics
echo '  "summary": {' >> "$REPORT_FILE"
echo "    \"baseline_version\": \"1.0.0\"," >> "$REPORT_FILE"
echo "    \"total_containers\": $(docker ps -q | wc -l)," >> "$REPORT_FILE"
echo "    \"healthy_containers\": $(docker ps -q --filter health=healthy | wc -l)," >> "$REPORT_FILE"
echo "    \"services_available\": $available_count," >> "$REPORT_FILE"
echo "    \"services_total\": $total_count" >> "$REPORT_FILE"
echo '  }' >> "$REPORT_FILE"
echo '}' >> "$REPORT_FILE"

# Display summary
echo
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ Performance Baseline Captured${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo
echo -e "${YELLOW}📊 System Metrics:${NC}"
echo -e "  CPU Usage: ${cpu_usage}%"
echo -e "  Memory Usage: ${memory_percent}%"
echo -e "  Load Average:${load_average}"
echo
echo -e "${YELLOW}🐳 Container Metrics:${NC}"
echo -e "  Total Containers: $(docker ps -q | wc -l)"
echo -e "  Healthy Containers: $(docker ps -q --filter health=healthy | wc -l)"
echo
echo -e "${YELLOW}⚡ API Performance:${NC}"
echo -e "  Health Check: $(echo "$health_time * 1000" | bc | cut -d'.' -f1)ms"
echo -e "  API Docs: $(echo "$docs_time * 1000" | bc | cut -d'.' -f1)ms"
echo
echo -e "${YELLOW}🏥 Service Availability:${NC}"
echo -e "  Available: $available_count/$total_count (${availability_percent}%)"
echo
echo -e "${GREEN}📁 Report saved to: $REPORT_FILE${NC}"
echo

# Create baseline summary for quick reference
cat > "$REPORT_DIR/latest_baseline.txt" << EOF
SutazAI Performance Baseline - $(date)
=====================================
CPU Usage: ${cpu_usage}%
Memory Usage: ${memory_percent}%
Containers: $(docker ps -q | wc -l) running
Health Check: $(echo "$health_time * 1000" | bc | cut -d'.' -f1)ms
Service Availability: ${availability_percent}%
EOF

echo -e "${BLUE}💡 Next Steps:${NC}"
echo -e "  1. Review the baseline metrics"
echo -e "  2. Run after optimizations to compare"
echo -e "  3. Track improvements over time"
echo

# Cleanup
rm -f /tmp/docker_stats.json