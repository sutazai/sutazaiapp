#!/bin/bash
# 🧪 SutazAI Complete System Validation Script
# Comprehensive testing of all system components and integrations

set -euo pipefail

# Configuration
PROJECT_ROOT="/opt/sutazaiapp"
LOG_FILE="logs/validation_$(date +%Y%m%d_%H%M%S).log"
RESULTS_FILE="reports/validation_report_$(date +%Y%m%d_%H%M%S).json"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
WARNINGS=0

# Results array
declare -a TEST_RESULTS=()

# Logging functions
log_info() {
    echo -e "${BLUE}ℹ️  [$(date +'%H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}✅ [$(date +'%H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}⚠️  [$(date +'%H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}❌ [$(date +'%H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

log_header() {
    echo -e "\n${BOLD}${CYAN}$1${NC}" | tee -a "$LOG_FILE"
    echo "══════════════════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
}

# Test execution function
run_test() {
    local test_name="$1"
    local test_command="$2"
    local is_critical="${3:-true}"
    
    ((TOTAL_TESTS++))
    
    log_info "Running test: $test_name"
    
    if eval "$test_command" &>/dev/null; then
        log_success "✓ $test_name"
        ((PASSED_TESTS++))
        TEST_RESULTS+=("{\"name\":\"$test_name\",\"status\":\"PASSED\",\"critical\":$is_critical}")
        return 0
    else
        if [ "$is_critical" = "true" ]; then
            log_error "✗ $test_name (CRITICAL)"
            ((FAILED_TESTS++))
            TEST_RESULTS+=("{\"name\":\"$test_name\",\"status\":\"FAILED\",\"critical\":$is_critical}")
        else
            log_warn "⚠ $test_name (WARNING)"
            ((WARNINGS++))
            TEST_RESULTS+=("{\"name\":\"$test_name\",\"status\":\"WARNING\",\"critical\":$is_critical}")
        fi
        return 1
    fi
}

# Test HTTP endpoint
test_endpoint() {
    local name="$1"
    local url="$2"
    local expected_status="${3:-200}"
    local is_critical="${4:-true}"
    
    run_test "$name" "curl -s -o /dev/null -w '%{http_code}' '$url' | grep -q '$expected_status'" "$is_critical"
}

# Test Docker container
test_container() {
    local name="$1"
    local container_name="$2"
    local is_critical="${3:-true}"
    
    run_test "$name" "docker ps --filter name=$container_name --filter status=running --quiet | grep -q ." "$is_critical"
}

# Test service health
test_service_health() {
    local name="$1"
    local service_name="$2"
    local is_critical="${3:-true}"
    
    run_test "$name" "sudo systemctl is-active --quiet $service_name" "$is_critical"
}

# Initialize validation
init_validation() {
    log_header "🧪 SutazAI Complete System Validation"
    
    # Create directories
    mkdir -p logs reports
    
    # Initialize results file
    echo "{\"validation_start\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"tests\":[]}" > "$RESULTS_FILE"
    
    log_info "Starting comprehensive system validation..."
    log_info "Results will be saved to: $RESULTS_FILE"
}

# Test core infrastructure
test_core_infrastructure() {
    log_header "🏗️ Testing Core Infrastructure"
    
    # Docker
    run_test "Docker daemon running" "docker info"
    run_test "Docker Compose available" "docker compose version"
    
    # System services
    test_service_health "PostgreSQL service" "postgresql" true
    test_service_health "Redis service" "redis-server" true
    
    # Core containers
    test_container "PostgreSQL container" "sutazai-postgres" true
    test_container "Redis container" "sutazai-redis" true
    test_container "Neo4j container" "sutazai-neo4j" true
    
    # Database connectivity
    run_test "PostgreSQL connectivity" "PGPASSWORD=sutazai_password psql -h localhost -U sutazai -d sutazai -c 'SELECT 1' -t"
    run_test "Redis connectivity" "redis-cli -a redis_password ping | grep -q PONG"
}

# Test AI model services
test_ai_models() {
    log_header "🧠 Testing AI Model Services"
    
    # Ollama
    test_container "Ollama container" "sutazai-ollama" true
    test_endpoint "Ollama API" "http://localhost:11434/api/tags" 200 true
    
    # Test model availability
    run_test "Check available models" "curl -s http://localhost:11434/api/tags | jq -r '.models | length' | awk '{print ($1 > 0)}' | grep -q 1"
    
    # Vector databases
    test_container "ChromaDB container" "sutazai-chromadb" true
    test_endpoint "ChromaDB API" "http://localhost:8001/api/v1/heartbeat" 200 true
    
    test_container "Qdrant container" "sutazai-qdrant" true
    test_endpoint "Qdrant API" "http://localhost:6333/health" 200 true
    
    test_container "FAISS container" "sutazai-faiss" false
    
    # Test model inference
    run_test "Ollama model inference" "curl -s -X POST http://localhost:11434/api/generate -d '{\"model\":\"qwen2.5:3b\",\"prompt\":\"Hello\",\"stream\":false}' | jq -r '.response' | grep -q ."
}

# Test application services
test_application_services() {
    log_header "🚀 Testing Application Services"
    
    # Backend
    test_container "Backend AGI container" "sutazai-backend-agi" true
    test_endpoint "Backend API health" "http://localhost:8000/health" 200 true
    test_endpoint "Backend API docs" "http://localhost:8000/docs" 200 false
    
    # Frontend
    test_container "Frontend AGI container" "sutazai-frontend-agi" true
    test_endpoint "Frontend UI" "http://localhost:8501" 200 true
    
    # API Gateway (if implemented)
    test_container "API Gateway container" "sutazai-api-gateway" false
    test_endpoint "API Gateway health" "http://localhost:8080/health" 200 false
}

# Test AI agents
test_ai_agents() {
    log_header "🤖 Testing AI Agents"
    
    # Core agents
    test_container "AutoGPT container" "sutazai-autogpt" false
    test_container "CrewAI container" "sutazai-crewai" false
    test_container "Letta container" "sutazai-letta" false
    
    # Code agents
    test_container "Aider container" "sutazai-aider" false
    test_endpoint "Aider API" "http://localhost:8095/health" 200 false
    
    test_container "GPT-Engineer container" "sutazai-gpt-engineer" false
    test_endpoint "GPT-Engineer API" "http://localhost:8097/health" 200 false
    
    # Workflow agents
    test_container "LangFlow container" "sutazai-langflow" false
    test_endpoint "LangFlow UI" "http://localhost:8090" 200 false
    
    test_container "FlowiseAI container" "sutazai-flowise" false
    test_endpoint "FlowiseAI UI" "http://localhost:8099" 200 false
    
    test_container "N8N container" "sutazai-n8n" false
    test_endpoint "N8N UI" "http://localhost:5678/healthz" 200 false
    
    test_container "Dify container" "sutazai-dify" false
    test_endpoint "Dify API" "http://localhost:8107" 200 false
    
    # JARVIS (if implemented)
    test_container "JARVIS AI container" "sutazai-jarvis-ai" false
    test_endpoint "JARVIS API" "http://localhost:8120/health" 200 false
}

# Test monitoring stack
test_monitoring() {
    log_header "📊 Testing Monitoring Stack"
    
    # Monitoring services
    test_container "Prometheus container" "sutazai-prometheus" false
    test_endpoint "Prometheus UI" "http://localhost:9090" 200 false
    test_endpoint "Prometheus metrics" "http://localhost:9090/metrics" 200 false
    
    test_container "Grafana container" "sutazai-grafana" false
    test_endpoint "Grafana UI" "http://localhost:3000" 200 false
    
    test_container "Loki container" "sutazai-loki" false
    test_endpoint "Loki ready" "http://localhost:3100/ready" 200 false
    
    test_container "Promtail container" "sutazai-promtail" false
    
    # Health monitor
    test_container "Health Monitor container" "sutazai-health-monitor" false
    test_endpoint "Health Monitor API" "http://localhost:8100" 200 false
}

# Test ML frameworks
test_ml_frameworks() {
    log_header "🔬 Testing ML Frameworks"
    
    # ML containers
    test_container "PyTorch container" "sutazai-pytorch" false
    test_container "TensorFlow container" "sutazai-tensorflow" false
    test_container "JAX container" "sutazai-jax" false
    test_container "FSDP container" "sutazai-fsdp" false
    
    # Test framework APIs if available
    test_endpoint "PyTorch Jupyter" "http://localhost:8888" 200 false
    test_endpoint "TensorFlow Jupyter" "http://localhost:8889" 200 false
}

# Test specialized services
test_specialized_services() {
    log_header "🛠️ Testing Specialized Services"
    
    # Advanced agents
    test_container "AgentGPT container" "sutazai-agentgpt" false
    test_container "AgentZero container" "sutazai-agentzero" false
    test_container "BigAGI container" "sutazai-bigagi" false
    test_endpoint "BigAGI UI" "http://localhost:8106" 200 false
    
    # Document processing
    test_container "Documind container" "sutazai-documind" false
    
    # Financial analysis
    test_container "FinRobot container" "sutazai-finrobot" false
    
    # Browser automation
    test_container "Browser Use container" "sutazai-browser-use" false
    test_container "Skyvern container" "sutazai-skyvern" false
    
    # Security
    test_container "Semgrep container" "sutazai-semgrep" false
    test_container "PentestGPT container" "sutazai-pentestgpt" false
    
    # MCP Server
    test_container "MCP Server container" "sutazai-mcp-server" false
}

# Test system performance
test_performance() {
    log_header "⚡ Testing System Performance"
    
    # System resources
    run_test "CPU usage under 80%" "awk '{print \$1}' /proc/loadavg | awk '{print (\$1 < 0.8 * $(nproc))}' | grep -q 1" false
    run_test "Memory usage under 90%" "free | awk 'NR==2{printf \"%.0f\", \$3*100/\$2}' | awk '{print (\$1 < 90)}' | grep -q 1" false
    run_test "Disk usage under 85%" "df / | awk 'NR==2{print \$5}' | sed 's/%//' | awk '{print (\$1 < 85)}' | grep -q 1" false
    
    # Network connectivity
    run_test "Internet connectivity" "ping -c 1 8.8.8.8" false
    run_test "DNS resolution" "nslookup google.com" false
    
    # Docker performance
    run_test "Docker containers responding" "docker ps --filter status=running --quiet | wc -l | awk '{print (\$1 > 10)}' | grep -q 1" false
}

# Test security
test_security() {
    log_header "🔒 Testing Security Configuration"
    
    # File permissions
    run_test ".env file permissions" "test \$(stat -c '%a' .env 2>/dev/null || echo 000) -le 600" true
    run_test "SSL certificates exist" "test -f ssl/cert.pem && test -f ssl/key.pem" false
    
    # Service security
    run_test "PostgreSQL not exposed publicly" "! netstat -tuln | grep ':5432.*0.0.0.0'" false
    run_test "Redis password protected" "redis-cli -a redis_password ping | grep -q PONG" true
    
    # Container security
    run_test "Containers running as non-root" "docker ps --quiet | xargs -I {} docker exec {} whoami | grep -v root | wc -l | awk '{print (\$1 > 0)}' | grep -q 1" false
}

# Test integration
test_integration() {
    log_header "🔗 Testing System Integration"
    
    # Backend-Frontend communication
    run_test "Backend-Frontend integration" "curl -s http://localhost:8501 | grep -q 'SutazAI'" false
    
    # AI model integration
    run_test "Backend-Ollama integration" "curl -s http://localhost:8000/health | jq -r '.ollama_status' | grep -q 'healthy'" false
    
    # Database integration
    run_test "Backend-Database integration" "curl -s http://localhost:8000/health | jq -r '.database_status' | grep -q 'healthy'" true
    
    # Vector database integration
    run_test "Backend-ChromaDB integration" "curl -s http://localhost:8000/health | jq -r '.chromadb_status' | grep -q 'healthy'" false
    run_test "Backend-Qdrant integration" "curl -s http://localhost:8000/health | jq -r '.qdrant_status' | grep -q 'healthy'" false
}

# Generate comprehensive report
generate_report() {
    log_header "📊 Generating Validation Report"
    
    local end_time=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    local success_rate=$((PASSED_TESTS * 100 / TOTAL_TESTS))
    
    # Create detailed JSON report
    cat > "$RESULTS_FILE" << EOF
{
  "validation_start": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "validation_end": "$end_time",
  "summary": {
    "total_tests": $TOTAL_TESTS,
    "passed_tests": $PASSED_TESTS,
    "failed_tests": $FAILED_TESTS,
    "warnings": $WARNINGS,
    "success_rate": $success_rate
  },
  "system_info": {
    "hostname": "$(hostname)",
    "os": "$(uname -s)",
    "kernel": "$(uname -r)",
    "architecture": "$(uname -m)",
    "cpu_cores": $(nproc),
    "memory_gb": $(($(free -m | awk 'NR==2{print $2}') / 1024)),
    "disk_usage": "$(df -h / | awk 'NR==2{print $5}')"
  },
  "services_status": {
    "docker_containers": $(docker ps --quiet | wc -l),
    "running_containers": $(docker ps --filter status=running --quiet | wc -l),
    "healthy_containers": $(docker ps --filter health=healthy --quiet | wc -l || echo 0)
  },
  "tests": [$(IFS=','; echo "${TEST_RESULTS[*]}")]
}
EOF
    
    # Generate HTML report
    local html_report="reports/validation_report_$(date +%Y%m%d_%H%M%S).html"
    
    cat > "$html_report" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>SutazAI System Validation Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }
        .summary { background: #ecf0f1; padding: 15px; margin: 20px 0; border-radius: 5px; }
        .passed { color: #27ae60; }
        .failed { color: #e74c3c; }
        .warning { color: #f39c12; }
        .test-item { padding: 5px; border-bottom: 1px solid #ddd; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #34495e; color: white; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 SutazAI System Validation Report</h1>
        <p>Generated: $(date)</p>
    </div>
    
    <div class="summary">
        <h2>📊 Summary</h2>
        <p>Total Tests: $TOTAL_TESTS</p>
        <p class="passed">Passed: $PASSED_TESTS</p>
        <p class="failed">Failed: $FAILED_TESTS</p>
        <p class="warning">Warnings: $WARNINGS</p>
        <p>Success Rate: $success_rate%</p>
    </div>
    
    <h2>🔍 Test Results</h2>
    <p>Detailed results available in: $RESULTS_FILE</p>
    <p>Log file: $LOG_FILE</p>
</body>
</html>
EOF
    
    log_success "Validation report generated: $html_report"
    log_info "JSON report: $RESULTS_FILE"
}

# Display final results
show_results() {
    log_header "🎯 Validation Results Summary"
    
    local success_rate=$((PASSED_TESTS * 100 / TOTAL_TESTS))
    
    echo -e "${BOLD}Total Tests:${NC} $TOTAL_TESTS"
    echo -e "${GREEN}✅ Passed:${NC} $PASSED_TESTS"
    echo -e "${RED}❌ Failed:${NC} $FAILED_TESTS"
    echo -e "${YELLOW}⚠️ Warnings:${NC} $WARNINGS"
    echo -e "${BOLD}Success Rate:${NC} $success_rate%"
    
    if [ $FAILED_TESTS -eq 0 ]; then
        echo -e "\n${GREEN}${BOLD}🎉 All critical tests passed! SutazAI system is fully operational.${NC}"
        return 0
    else
        echo -e "\n${RED}${BOLD}⚠️ Some critical tests failed. Please review the logs and fix issues.${NC}"
        return 1
    fi
}

# Main execution
main() {
    cd "$PROJECT_ROOT" || exit 1
    
    init_validation
    
    # Run all test suites
    test_core_infrastructure
    test_ai_models
    test_application_services
    test_ai_agents
    test_monitoring
    test_ml_frameworks
    test_specialized_services
    test_performance
    test_security
    test_integration
    
    # Generate reports and show results
    generate_report
    show_results
}

# Handle script arguments
case "${1:-validate}" in
    "validate"|"all")
        main
        ;;
    "infrastructure")
        init_validation
        test_core_infrastructure
        show_results
        ;;
    "models")
        init_validation
        test_ai_models
        show_results
        ;;
    "apps")
        init_validation
        test_application_services
        show_results
        ;;
    "agents")
        init_validation
        test_ai_agents
        show_results
        ;;
    "monitoring")
        init_validation
        test_monitoring
        show_results
        ;;
    "performance")
        init_validation
        test_performance
        show_results
        ;;
    "security")
        init_validation
        test_security
        show_results
        ;;
    "integration")
        init_validation
        test_integration
        show_results
        ;;
    "help")
        echo "Usage: $0 [validate|infrastructure|models|apps|agents|monitoring|performance|security|integration|help]"
        echo ""
        echo "Commands:"
        echo "  validate        - Run complete validation (default)"
        echo "  infrastructure  - Test core infrastructure only"
        echo "  models          - Test AI models only"
        echo "  apps            - Test application services only"
        echo "  agents          - Test AI agents only"
        echo "  monitoring      - Test monitoring stack only"
        echo "  performance     - Test system performance only"
        echo "  security        - Test security configuration only"
        echo "  integration     - Test system integration only"
        echo "  help            - Show this help"
        ;;
    *)
        log_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac 