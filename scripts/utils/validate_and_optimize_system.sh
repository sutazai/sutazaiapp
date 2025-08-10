#!/bin/bash
# SutazAI Complete System Validation and Optimization
# Performs comprehensive checks and optimizations across all components

set -e

# Colors

# Signal handlers for graceful shutdown
cleanup_and_exit() {
    local exit_code="${1:-0}"
    echo "Script interrupted, cleaning up..." >&2
    # Clean up any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    exit "$exit_code"
}

trap 'cleanup_and_exit 130' INT
trap 'cleanup_and_exit 143' TERM
trap 'cleanup_and_exit 1' ERR

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$WORKSPACE_DIR/logs/validation_$(date +%Y%m%d_%H%M%S).log"
REPORT_FILE="$WORKSPACE_DIR/reports/system_report_$(date +%Y%m%d_%H%M%S).json"

# Counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNINGS=0
OPTIMIZATIONS_APPLIED=0

# Create directories
mkdir -p "$WORKSPACE_DIR/logs" "$WORKSPACE_DIR/reports"

# Logging functions
log() { echo -e "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"; }
log_success() { echo -e "${GREEN}✓${NC} $1" | tee -a "$LOG_FILE"; ((PASSED_CHECKS++)); }
log_error() { echo -e "${RED}✗${NC} $1" | tee -a "$LOG_FILE"; ((FAILED_CHECKS++)); }
log_warn() { echo -e "${YELLOW}⚠${NC} $1" | tee -a "$LOG_FILE"; ((WARNINGS++)); }
log_info() { echo -e "${BLUE}ℹ${NC} $1" | tee -a "$LOG_FILE"; }
log_header() { 
    echo -e "\n${PURPLE}═══════════════════════════════════════════════════════════${NC}" | tee -a "$LOG_FILE"
    echo -e "${PURPLE}$1${NC}" | tee -a "$LOG_FILE"
    echo -e "${PURPLE}═══════════════════════════════════════════════════════════${NC}\n" | tee -a "$LOG_FILE"
}

# Initialize report
init_report() {
    cat > "$REPORT_FILE" << EOF
{
    "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "system": "SutazAI Enhanced Coordinator v2.0",
    "checks": {
        "infrastructure": {},
        "services": {},
        "coordinator": {},
        "agents": {},
        "models": {},
        "performance": {},
        "security": {}
    },
    "optimizations": [],
    "recommendations": []
}
EOF
}

# Update report
update_report() {
    local category="$1"
    local subcategory="$2"
    local status="$3"
    local details="$4"
    
    python3 -c "
import json
with open('$REPORT_FILE', 'r') as f:
    report = json.load(f)
report['checks']['$category']['$subcategory'] = {
    'status': '$status',
    'details': '$details',
    'timestamp': '$(date -u +"%Y-%m-%dT%H:%M:%SZ")'
}
with open('$REPORT_FILE', 'w') as f:
    json.dump(report, f, indent=2)
"
}

# System resource checks
check_system_resources() {
    log_header "System Resource Validation"
    
    # CPU check
    ((TOTAL_CHECKS++))
    local cpu_count=$(nproc)
    if [ "$cpu_count" -ge 8 ]; then
        log_success "CPU cores: $cpu_count (recommended: 8+)"
        update_report "infrastructure" "cpu" "pass" "$cpu_count cores"
    else
        log_warn "CPU cores: $cpu_count (recommended: 8+)"
        update_report "infrastructure" "cpu" "warning" "$cpu_count cores"
    fi
    
    # Memory check
    ((TOTAL_CHECKS++))
    local total_mem=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$total_mem" -ge 48 ]; then
        log_success "RAM: ${total_mem}GB (recommended: 48GB+)"
        update_report "infrastructure" "memory" "pass" "${total_mem}GB"
    elif [ "$total_mem" -ge 16 ]; then
        log_warn "RAM: ${total_mem}GB (minimum: 16GB, recommended: 48GB)"
        update_report "infrastructure" "memory" "warning" "${total_mem}GB"
    else
        log_error "RAM: ${total_mem}GB (insufficient, minimum: 16GB)"
        update_report "infrastructure" "memory" "fail" "${total_mem}GB"
    fi
    
    # GPU check
    ((TOTAL_CHECKS++))
    if command -v nvidia-smi &> /dev/null; then
        local gpu_info=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
        log_success "GPU detected: $gpu_info"
        update_report "infrastructure" "gpu" "pass" "$gpu_info"
    else
        log_warn "No GPU detected (optional but recommended)"
        update_report "infrastructure" "gpu" "warning" "No GPU"
    fi
    
    # Disk space check
    ((TOTAL_CHECKS++))
    local available_space=$(df -BG /workspace | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$available_space" -ge 100 ]; then
        log_success "Disk space: ${available_space}GB available"
        update_report "infrastructure" "disk" "pass" "${available_space}GB"
    else
        log_warn "Disk space: ${available_space}GB (recommended: 100GB+)"
        update_report "infrastructure" "disk" "warning" "${available_space}GB"
    fi
}

# Docker service validation
check_docker_services() {
    log_header "Docker Services Validation"
    
    local required_services=(
        "sutazai-ollama"
        "sutazai-redis"
        "sutazai-postgresql"
        "sutazai-qdrant"
        "sutazai-chromadb"
        "sutazai-faiss"
        "sutazai-neo4j"
        "sutazai-pytorch"
        "sutazai-tensorflow"
        "sutazai-jax"
    )
    
    for service in "${required_services[@]}"; do
        ((TOTAL_CHECKS++))
        if docker ps --format "{{.Names}}" | grep -q "^$service$"; then
            local status=$(docker inspect -f '{{.State.Status}}' "$service")
            local health=$(docker inspect -f '{{.State.Health.Status}}' "$service" 2>/dev/null || echo "none")
            
            if [ "$status" = "running" ] && ([ "$health" = "healthy" ] || [ "$health" = "none" ]); then
                log_success "$service: running${health:+ ($health)}"
                update_report "services" "$service" "pass" "running"
            else
                log_error "$service: $status${health:+ ($health)}"
                update_report "services" "$service" "fail" "$status"
            fi
        else
            log_error "$service: not found"
            update_report "services" "$service" "fail" "not found"
        fi
    done
}

# Coordinator system validation
check_coordinator_system() {
    log_header "Coordinator System Validation"
    
    # Check if Coordinator is deployed
    ((TOTAL_CHECKS++))
    if docker ps --format "{{.Names}}" | grep -q "sutazai-coordinator-core"; then
        log_success "Coordinator Core: deployed"
        update_report "coordinator" "deployment" "pass" "deployed"
        
        # Test Coordinator API
        ((TOTAL_CHECKS++))
        if curl -sf http://localhost:8888/health > /dev/null; then
            log_success "Coordinator API: healthy"
            update_report "coordinator" "api" "pass" "healthy"
            
            # Test ULM integration
            ((TOTAL_CHECKS++))
            local ulm_test=$(curl -sf -X POST http://localhost:8888/process \
                -H 'Content-Type: application/json' \
                -d '{"input": "test"}' | grep -c "learning_progress" || echo 0)
            
            if [ "$ulm_test" -gt 0 ]; then
                log_success "Universal Learning Machine: integrated"
                update_report "coordinator" "ulm" "pass" "integrated"
            else
                log_warn "Universal Learning Machine: not detected"
                update_report "coordinator" "ulm" "warning" "not detected"
            fi
        else
            log_error "Coordinator API: not responding"
            update_report "coordinator" "api" "fail" "not responding"
        fi
        
        # Check enhanced agents
        local agents=("jarvis:8026" "autogen:8001" "crewai:8002" "localagi:8021")
        local active_agents=0
        
        for agent in "${agents[@]}"; do
            ((TOTAL_CHECKS++))
            IFS=':' read -ra PARTS <<< "$agent"
            local name="${PARTS[0]}"
            local port="${PARTS[1]}"
            
            if curl -sf "http://localhost:$port/health" > /dev/null; then
                log_success "Agent $name: active"
                update_report "agents" "$name" "pass" "active"
                ((active_agents++))
            else
                log_warn "Agent $name: inactive"
                update_report "agents" "$name" "warning" "inactive"
            fi
        done
        
        ((TOTAL_CHECKS++))
        if [ "$active_agents" -ge 3 ]; then
            log_success "Agent ecosystem: $active_agents/4 agents active"
            update_report "coordinator" "agents" "pass" "$active_agents active"
        else
            log_warn "Agent ecosystem: only $active_agents/4 agents active"
            update_report "coordinator" "agents" "warning" "$active_agents active"
        fi
    else
        log_warn "Coordinator system not deployed"
        update_report "coordinator" "deployment" "warning" "not deployed"
    fi
}

# Model availability check
check_models() {
    log_header "Model Availability Check"
    
    local required_models=(
        "tinyllama2.5:3b"
        "tinyllama2.5-coder:3b"
        "tinyllama2.5:3b"
        "nomic-embed-text"
    )
    
    local available_models=0
    for model in "${required_models[@]}"; do
        ((TOTAL_CHECKS++))
        if docker exec sutazai-ollama ollama list 2>/dev/null | grep -q "$model"; then
            log_success "Model $model: available"
            update_report "models" "$model" "pass" "available"
            ((available_models++))
        else
            log_warn "Model $model: not found"
            update_report "models" "$model" "warning" "not found"
        fi
    done
    
    # Check total model count
    local total_models=$(docker exec sutazai-ollama ollama list 2>/dev/null | grep -c ":" || echo 0)
    log_info "Total models available: $total_models"
}

# Performance checks
check_performance() {
    log_header "Performance Validation"
    
    # CPU usage
    ((TOTAL_CHECKS++))
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
    if (( $(echo "$cpu_usage < 80" | bc -l) )); then
        log_success "CPU usage: ${cpu_usage}%"
        update_report "performance" "cpu_usage" "pass" "${cpu_usage}%"
    else
        log_warn "CPU usage high: ${cpu_usage}%"
        update_report "performance" "cpu_usage" "warning" "${cpu_usage}%"
    fi
    
    # Memory usage
    ((TOTAL_CHECKS++))
    local mem_usage=$(free | grep Mem | awk '{print int($3/$2 * 100)}')
    if [ "$mem_usage" -lt 80 ]; then
        log_success "Memory usage: ${mem_usage}%"
        update_report "performance" "memory_usage" "pass" "${mem_usage}%"
    else
        log_warn "Memory usage high: ${mem_usage}%"
        update_report "performance" "memory_usage" "warning" "${mem_usage}%"
    fi
    
    # Response time test
    ((TOTAL_CHECKS++))
    local start_time=$(date +%s%N)
    curl -sf http://localhost:8000/health > /dev/null
    local end_time=$(date +%s%N)
    local response_time=$(( (end_time - start_time) / 1000000 ))
    
    if [ "$response_time" -lt 1000 ]; then
        log_success "API response time: ${response_time}ms"
        update_report "performance" "response_time" "pass" "${response_time}ms"
    else
        log_warn "API response time slow: ${response_time}ms"
        update_report "performance" "response_time" "warning" "${response_time}ms"
    fi
}

# Security checks
check_security() {
    log_header "Security Validation"
    
    # Check .env permissions
    ((TOTAL_CHECKS++))
    if [ -f "$WORKSPACE_DIR/.env" ]; then
        local perms=$(stat -c %a "$WORKSPACE_DIR/.env" 2>/dev/null || echo "000")
        if [ "$perms" = "600" ] || [ "$perms" = "400" ]; then
            log_success ".env permissions: secure ($perms)"
            update_report "security" "env_permissions" "pass" "$perms"
        else
            log_warn ".env permissions: $perms (should be 600)"
            update_report "security" "env_permissions" "warning" "$perms"
        fi
    fi
    
    # Check for default passwords
    ((TOTAL_CHECKS++))
    if [ -f "$WORKSPACE_DIR/.env" ]; then
        if grep -q "your-secret-key-here\|secure-password-here\|changeme" "$WORKSPACE_DIR/.env"; then
            log_error "Default passwords detected in .env"
            update_report "security" "passwords" "fail" "defaults detected"
        else
            log_success "No default passwords detected"
            update_report "security" "passwords" "pass" "secure"
        fi
    fi
    
    # Check exposed ports
    ((TOTAL_CHECKS++))
    local exposed_ports=$(docker ps --format "table {{.Ports}}" | grep -c "0.0.0.0:" || echo 0)
    log_info "Exposed ports: $exposed_ports services"
    update_report "security" "exposed_ports" "info" "$exposed_ports"
}

# Apply optimizations
apply_optimizations() {
    log_header "System Optimizations"
    
    # Clean up unused Docker resources
    log_info "Cleaning Docker resources..."
    docker system prune -f --volumes 2>/dev/null && ((OPTIMIZATIONS_APPLIED++))
    
    # Optimize database
    if docker ps --format "{{.Names}}" | grep -q "sutazai-postgresql"; then
        log_info "Optimizing PostgreSQL..."
        docker exec sutazai-postgresql psql -U sutazai -d sutazai_main -c "VACUUM ANALYZE;" 2>/dev/null && ((OPTIMIZATIONS_APPLIED++))
    fi
    
    # Clear old logs
    log_info "Cleaning old logs..."
    find "$WORKSPACE_DIR/logs" -name "*.log" -mtime +7 -delete 2>/dev/null && ((OPTIMIZATIONS_APPLIED++))
    
    # Optimize Redis
    if docker ps --format "{{.Names}}" | grep -q "sutazai-redis"; then
        log_info "Optimizing Redis..."
        docker exec sutazai-redis redis-cli BGREWRITEAOF 2>/dev/null && ((OPTIMIZATIONS_APPLIED++))
    fi
    
    log_success "Applied $OPTIMIZATIONS_APPLIED optimizations"
}

# Generate recommendations
generate_recommendations() {
    local recommendations=()
    
    # Resource recommendations
    local total_mem=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$total_mem" -lt 48 ]; then
        recommendations+=("Upgrade system RAM to 48GB for optimal performance")
    fi
    
    if ! command -v nvidia-smi &> /dev/null; then
        recommendations+=("Install NVIDIA GPU with 4GB+ VRAM for accelerated ML processing")
    fi
    
    # Coordinator recommendations
    if ! docker ps --format "{{.Names}}" | grep -q "sutazai-coordinator-core"; then
        recommendations+=("Deploy Enhanced Coordinator system: DEPLOY_BRAIN=true ./scripts/deploy_complete_system.sh")
    fi
    
    # Model recommendations
    local model_count=$(docker exec sutazai-ollama ollama list 2>/dev/null | grep -c ":" || echo 0)
    if [ "$model_count" -lt 10 ]; then
        recommendations+=("Download additional models for enhanced capabilities")
    fi
    
    # Security recommendations
    if [ -f "$WORKSPACE_DIR/.env" ]; then
        local perms=$(stat -c %a "$WORKSPACE_DIR/.env" 2>/dev/null || echo "000")
        if [ "$perms" != "600" ] && [ "$perms" != "400" ]; then
            recommendations+=("Secure .env file: chmod 600 $WORKSPACE_DIR/.env")
        fi
    fi
    
    # Update report with recommendations
    python3 -c "
import json
recs = $(printf '%s\n' "${recommendations[@]}" | jq -R . | jq -s .)
with open('$REPORT_FILE', 'r') as f:
    report = json.load(f)
report['recommendations'] = recs
with open('$REPORT_FILE', 'w') as f:
    json.dump(report, f, indent=2)
"
}

# Generate final summary
generate_summary() {
    log_header "Validation Summary"
    
    local total=$((PASSED_CHECKS + FAILED_CHECKS + WARNINGS))
    local score=$((PASSED_CHECKS * 100 / total))
    
    echo -e "\n${CYAN}════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}                    SYSTEM HEALTH REPORT${NC}"
    echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
    echo
    echo -e "Total Checks: $total"
    echo -e "${GREEN}Passed: $PASSED_CHECKS${NC}"
    echo -e "${YELLOW}Warnings: $WARNINGS${NC}"
    echo -e "${RED}Failed: $FAILED_CHECKS${NC}"
    echo -e "Health Score: ${score}%"
    echo
    
    if [ "$score" -ge 90 ]; then
        echo -e "${GREEN}✅ System Status: EXCELLENT${NC}"
        echo -e "The SutazAI system is operating at peak performance!"
    elif [ "$score" -ge 70 ]; then
        echo -e "${YELLOW}⚠️  System Status: GOOD${NC}"
        echo -e "The system is functional but has room for improvement."
    else
        echo -e "${RED}❌ System Status: NEEDS ATTENTION${NC}"
        echo -e "Critical issues detected. Please review the report."
    fi
    
    echo
    echo -e "Optimizations Applied: $OPTIMIZATIONS_APPLIED"
    echo -e "Full Report: $REPORT_FILE"
    echo -e "Detailed Logs: $LOG_FILE"
    echo
    
    # Update final report
    python3 -c "
import json
with open('$REPORT_FILE', 'r') as f:
    report = json.load(f)
report['summary'] = {
    'total_checks': $total,
    'passed': $PASSED_CHECKS,
    'warnings': $WARNINGS,
    'failed': $FAILED_CHECKS,
    'score': $score,
    'optimizations_applied': $OPTIMIZATIONS_APPLIED
}
with open('$REPORT_FILE', 'w') as f:
    json.dump(report, f, indent=2)
"
}

# Main execution
main() {
    log_header "🚀 SutazAI System Validation & Optimization"
    log_info "Starting comprehensive system analysis..."
    
    # Initialize report
    init_report
    
    # Run all checks
    check_system_resources
    check_docker_services
    check_coordinator_system
    check_models
    check_performance
    check_security
    
    # Apply optimizations
    apply_optimizations
    
    # Generate recommendations
    generate_recommendations
    
    # Generate summary
    generate_summary
    
    # Quick actions for common issues
    if [ "$FAILED_CHECKS" -gt 0 ]; then
        echo -e "\n${YELLOW}Quick Fix Commands:${NC}"
        
        if ! docker ps --format "{{.Names}}" | grep -q "sutazai-coordinator-core"; then
            echo "Deploy Coordinator: ./scripts/deploy_coordinator_enhanced.sh"
        fi
        
        if [ "$WARNINGS" -gt 5 ]; then
            echo "Run full deployment: ./scripts/deploy_complete_system.sh"
        fi
        
        echo "View live logs: tail -f $WORKSPACE_DIR/logs/*.log"
        echo
    fi
}

# Execute main function
main "$@"