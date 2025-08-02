#!/bin/bash
# Quick SutazAI Deployment Verification Script
# Testing QA Validator Agent
# Version: 1.0.0

set -e

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m'
BOLD='\033[1m'

# Global counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0

# Logging
LOG_FILE="/opt/sutazaiapp/logs/quick_deployment_check_$(date +%Y%m%d_%H%M%S).log"
mkdir -p /opt/sutazaiapp/logs

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

print_banner() {
    echo -e "\n${CYAN}===============================================================================${NC}"
    echo -e "${CYAN}${BOLD}🔍 SutazAI Quick Deployment Verification${NC}"
    echo -e "${CYAN}Testing QA Validator Agent | Version 1.0.0${NC}"
    echo -e "${CYAN}===============================================================================${NC}\n"
}

print_section() {
    local title="$1"
    local icon="${2:-📋}"
    echo -e "\n${YELLOW}${BOLD}${icon} ${title}${NC}"
    echo -e "${YELLOW}$(printf '%.0s-' {1..50})${NC}"
}

check_command() {
    local cmd="$1"
    if command -v "$cmd" >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

check_port() {
    local host="$1"
    local port="$2"
    local service="$3"
    
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    if nc -z "$host" "$port" 2>/dev/null; then
        echo -e "  ${GREEN}✅ ${service}${NC} (port ${port})"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
        log "SUCCESS: $service port $port is open"
        return 0
    else
        echo -e "  ${RED}❌ ${service}${NC} (port ${port})"
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
        log "FAILED: $service port $port is not accessible"
        return 1
    fi
}

check_http() {
    local url="$1"
    local service="$2"
    local expected_code="${3:-200}"
    
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    if command -v curl >/dev/null 2>&1; then
        response=$(curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null || echo "000")
        if [[ "$response" =~ $expected_code ]]; then
            echo -e "  ${GREEN}✅ ${service}${NC} (HTTP $response)"
            PASSED_CHECKS=$((PASSED_CHECKS + 1))
            log "SUCCESS: $service HTTP check passed ($response)"
            return 0
        else
            echo -e "  ${RED}❌ ${service}${NC} (HTTP $response)"
            FAILED_CHECKS=$((FAILED_CHECKS + 1))
            log "FAILED: $service HTTP check failed ($response)"
            return 1
        fi
    else
        echo -e "  ${YELLOW}⚠️ ${service}${NC} (curl not available)"
        log "WARNING: curl not available for $service check"
        return 1
    fi
}

check_docker_containers() {
    print_section "Docker Container Status" "🐳"
    
    if ! check_command docker; then
        echo -e "  ${RED}❌ Docker command not available${NC}"
        log "ERROR: Docker command not found"
        return 1
    fi
    
    if ! docker info >/dev/null 2>&1; then
        echo -e "  ${RED}❌ Docker daemon not running${NC}"
        log "ERROR: Docker daemon not accessible"
        return 1
    fi
    
    local containers=$(docker ps --filter "name=sutazai-*" --format "table {{.Names}}\t{{.Status}}" 2>/dev/null || echo "")
    
    if [[ -z "$containers" ]]; then
        echo -e "  ${RED}❌ No SutazAI containers found${NC}"
        log "ERROR: No SutazAI containers running"
        return 1
    fi
    
    local running_count=$(docker ps --filter "name=sutazai-*" -q | wc -l)
    local total_count=$(docker ps -a --filter "name=sutazai-*" -q | wc -l)
    
    echo -e "  ${BLUE}📊 Running containers: ${running_count}/${total_count}${NC}"
    log "INFO: Docker containers running: $running_count/$total_count"
    
    # List key containers
    docker ps --filter "name=sutazai-*" --format "table {{.Names}}\t{{.Status}}" | tail -n +2 | head -10 | while read -r name status; do
        if [[ "$status" == *"Up"* ]]; then
            echo -e "  ${GREEN}✅ $name${NC}"
        else
            echo -e "  ${RED}❌ $name${NC} ($status)"
        fi
    done
    
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    if [[ $running_count -gt 0 ]]; then
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
        return 0
    else
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
        return 1
    fi
}

check_core_services() {
    print_section "Core Service Health" "🏥"
    
    # Core infrastructure
    check_port "localhost" "5432" "PostgreSQL Database"
    check_port "localhost" "6379" "Redis Cache"
    check_port "localhost" "7687" "Neo4j Graph DB"
    
    # Vector databases
    check_port "localhost" "8001" "ChromaDB"
    check_port "localhost" "6333" "Qdrant"
    
    # Core application
    check_port "localhost" "8000" "Backend API"
    check_port "localhost" "8501" "Frontend UI"
    
    # AI services
    check_port "localhost" "11434" "Ollama"
    check_port "localhost" "4000" "LiteLLM Proxy"
    
    # Monitoring
    check_port "localhost" "9090" "Prometheus"
    check_port "localhost" "3000" "Grafana"
}

check_api_endpoints() {
    print_section "API Endpoint Validation" "🔗"
    
    # Backend health
    check_http "http://localhost:8000/health" "Backend Health"
    check_http "http://localhost:8000/agents" "Agents API"
    check_http "http://localhost:8000/models" "Models API"
    check_http "http://localhost:8000/public/metrics" "Public Metrics"
    
    # Frontend health (may return 302 redirect)
    check_http "http://localhost:8501" "Frontend Health" "200|302"
    
    # Ollama API
    check_http "http://localhost:11434/api/tags" "Ollama Tags API"
    
    # Vector databases
    check_http "http://localhost:8001/api/v1/heartbeat" "ChromaDB Heartbeat"
    check_http "http://localhost:6333/cluster" "Qdrant Cluster Info"
    
    # Monitoring
    check_http "http://localhost:9090/-/healthy" "Prometheus Health"
    check_http "http://localhost:3000/api/health" "Grafana Health"
}

check_agent_services() {
    print_section "AI Agent Services" "🤖"
    
    # Check agent ports
    check_port "localhost" "8080" "AutoGPT Agent"
    check_port "localhost" "8096" "CrewAI Team"
    check_port "localhost" "8095" "Aider Code Assistant"
    check_port "localhost" "8097" "GPT Engineer"
    check_port "localhost" "8091" "AgentGPT"
    check_port "localhost" "8092" "PrivateGPT"
    check_port "localhost" "8105" "AgentZero"
}

check_workflow_services() {
    print_section "Workflow & Automation" "🔄"
    
    check_port "localhost" "8090" "LangFlow"
    check_port "localhost" "8099" "Flowise"
    check_port "localhost" "8107" "Dify"
    check_port "localhost" "5678" "n8n"
}

test_model_inference() {
    print_section "Model Inference Testing" "🧠"
    
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    if check_command curl && check_command jq; then
        # Get available models
        local models=$(curl -s "http://localhost:11434/api/tags" 2>/dev/null | jq -r '.models[].name' 2>/dev/null | head -1)
        
        if [[ -n "$models" ]]; then
            echo -e "  ${BLUE}📚 Testing model: $models${NC}"
            
            # Test inference with timeout
            local test_response=$(timeout 30 curl -s -X POST "http://localhost:11434/api/generate" \
                -H "Content-Type: application/json" \
                -d "{\"model\":\"$models\",\"prompt\":\"Hello\",\"stream\":false}" 2>/dev/null | \
                jq -r '.response' 2>/dev/null)
            
            if [[ -n "$test_response" && "$test_response" != "null" ]]; then
                echo -e "  ${GREEN}✅ Model inference working${NC}"
                PASSED_CHECKS=$((PASSED_CHECKS + 1))
                log "SUCCESS: Model inference test passed"
            else
                echo -e "  ${RED}❌ Model inference failed${NC}"
                FAILED_CHECKS=$((FAILED_CHECKS + 1))
                log "FAILED: Model inference test failed"
            fi
        else
            echo -e "  ${RED}❌ No models available${NC}"
            FAILED_CHECKS=$((FAILED_CHECKS + 1))
            log "FAILED: No Ollama models found"
        fi
    else
        echo -e "  ${YELLOW}⚠️ curl or jq not available for testing${NC}"
        log "WARNING: curl or jq not available for model testing"
    fi
}

check_system_resources() {
    print_section "System Resource Usage" "📊"
    
    # CPU usage
    if check_command top; then
        local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//' 2>/dev/null || echo "unknown")
        echo -e "  ${BLUE}🖥️  CPU Usage: ${cpu_usage}${NC}"
        log "INFO: CPU usage: $cpu_usage"
    fi
    
    # Memory usage
    if check_command free; then
        local memory_info=$(free -h | grep "Mem:" | awk '{print $3 "/" $2 " (" int($3/$2 * 100) "%)"}' 2>/dev/null || echo "unknown")
        echo -e "  ${BLUE}🧠 Memory: ${memory_info}${NC}"
        log "INFO: Memory usage: $memory_info"
    fi
    
    # Disk usage
    if check_command df; then
        local disk_usage=$(df -h / | tail -1 | awk '{print $3 "/" $2 " (" $5 ")"}' 2>/dev/null || echo "unknown")
        echo -e "  ${BLUE}💽 Disk: ${disk_usage}${NC}"
        log "INFO: Disk usage: $disk_usage"
    fi
    
    # Load average
    if [[ -f /proc/loadavg ]]; then
        local load_avg=$(cat /proc/loadavg | awk '{print $1 ", " $2 ", " $3}' 2>/dev/null || echo "unknown")
        echo -e "  ${BLUE}⚖️  Load Average: ${load_avg}${NC}"
        log "INFO: Load average: $load_avg"
    fi
    
    # Docker resource usage
    if check_command docker; then
        local docker_stats=$(timeout 5 docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" 2>/dev/null | tail -n +2 | head -5)
        if [[ -n "$docker_stats" ]]; then
            echo -e "  ${BLUE}🐳 Top Docker Containers:${NC}"
            echo "$docker_stats" | while read -r line; do
                echo -e "    ${CYAN}$line${NC}"
            done
        fi
    fi
}

generate_summary() {
    print_section "Deployment Summary" "📋"
    
    local success_rate=0
    if [[ $TOTAL_CHECKS -gt 0 ]]; then
        success_rate=$((PASSED_CHECKS * 100 / TOTAL_CHECKS))
    fi
    
    local status_color=$RED
    local status_text="CRITICAL"
    
    if [[ $success_rate -ge 90 ]]; then
        status_color=$GREEN
        status_text="EXCELLENT"
    elif [[ $success_rate -ge 80 ]]; then
        status_color=$GREEN
        status_text="GOOD"
    elif [[ $success_rate -ge 70 ]]; then
        status_color=$YELLOW
        status_text="ACCEPTABLE"
    elif [[ $success_rate -ge 50 ]]; then
        status_color=$YELLOW
        status_text="DEGRADED"
    fi
    
    echo -e "\n${CYAN}===============================================================================${NC}"
    echo -e "${CYAN}${BOLD}📊 DEPLOYMENT VERIFICATION SUMMARY${NC}"
    echo -e "${CYAN}===============================================================================${NC}"
    
    echo -e "\n${status_color}${BOLD}Overall Status: ${status_text} (${success_rate}%)${NC}"
    echo -e "${BLUE}Checks Passed: ${PASSED_CHECKS}/${TOTAL_CHECKS}${NC}"
    echo -e "${BLUE}Failed Checks: ${FAILED_CHECKS}${NC}"
    echo -e "${BLUE}Log File: ${LOG_FILE}${NC}"
    
    if [[ $success_rate -ge 80 ]]; then
        echo -e "\n${YELLOW}${BOLD}🌐 Key Access Points:${NC}"
        echo -e "  • Frontend UI: http://localhost:8501"
        echo -e "  • Backend API: http://localhost:8000"
        echo -e "  • API Documentation: http://localhost:8000/docs"
        echo -e "  • Grafana Dashboard: http://localhost:3000"
        echo -e "  • LangFlow: http://localhost:8090"
        echo -e "  • Flowise: http://localhost:8099"
        echo -e "  • Dify: http://localhost:8107"
        echo -e "  • n8n Workflows: http://localhost:5678"
    fi
    
    echo -e "\n${YELLOW}${BOLD}📌 Recommendations:${NC}"
    if [[ $success_rate -ge 90 ]]; then
        echo -e "  • System is ready for production use"
        echo -e "  • Run full Python verification: python3 scripts/comprehensive_deployment_verification.py"
        echo -e "  • Monitor system with: scripts/monitor_dashboard.sh"
    elif [[ $success_rate -ge 80 ]]; then
        echo -e "  • System is functional with minor issues"
        echo -e "  • Review failed checks in log file"
        echo -e "  • Check individual service logs: docker-compose logs [service-name]"
    elif [[ $success_rate -ge 60 ]]; then
        echo -e "  • System has significant issues requiring attention"
        echo -e "  • Restart failed services: docker-compose restart [service-name]"
        echo -e "  • Check resource constraints"
    else
        echo -e "  • System requires immediate attention"
        echo -e "  • Check Docker and system status"
        echo -e "  • Review docker-compose.yml configuration"
        echo -e "  • Restart entire stack: docker-compose down && docker-compose up -d"
    fi
    
    echo -e "\n${CYAN}===============================================================================${NC}"
    
    log "SUMMARY: Overall status: $status_text ($success_rate%)"
    log "SUMMARY: Checks passed: $PASSED_CHECKS/$TOTAL_CHECKS"
    
    # Return appropriate exit code
    if [[ $success_rate -ge 80 ]]; then
        return 0  # Success
    elif [[ $success_rate -ge 60 ]]; then
        return 1  # Warning
    else
        return 2  # Critical
    fi
}

main() {
    local start_time=$(date +%s)
    
    print_banner
    log "INFO: Starting quick deployment verification"
    
    # Check prerequisites
    if ! check_command nc; then
        echo -e "${YELLOW}⚠️ netcat (nc) not available - some checks may fail${NC}"
        log "WARNING: netcat not available"
    fi
    
    # Run verification steps
    check_docker_containers
    check_core_services
    check_api_endpoints
    check_agent_services
    check_workflow_services
    test_model_inference
    check_system_resources
    
    # Generate summary and exit
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    log "INFO: Verification completed in ${duration} seconds"
    
    generate_summary
    exit_code=$?
    
    log "INFO: Verification finished with exit code: $exit_code"
    exit $exit_code
}

# Handle script interruption
trap 'echo -e "\n${RED}Verification interrupted${NC}"; log "WARNING: Verification interrupted"; exit 130' INT TERM

# Run main function
main "$@"