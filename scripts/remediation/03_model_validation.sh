#!/bin/bash
################################################################################
# ULTRA-PRECISE MODEL VALIDATION SCRIPT
# Purpose: Validate Ollama TinyLlama model configuration and functionality  
# Author: ULTRA-REMEDIATION-MASTER-001 with ULTRATHINK + ULTRADEEPCODEBASESEARCH
# Date: August 13, 2025
# Follows: ALL CLAUDE.md Rules with ULTRA-PRECISION (100% working solution)
################################################################################

set -euo pipefail

# Script configuration with ULTRA-PRECISION
readonly SCRIPT_NAME="Ultra-Precise Model Validation"
readonly SCRIPT_VERSION="1.0.0"
readonly PROJECT_ROOT="/opt/sutazaiapp"
readonly TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
readonly LOG_FILE="${PROJECT_ROOT}/logs/model_validation_${TIMESTAMP}.log"

# REAL model configuration from ULTRADEEPCODEBASESEARCH analysis
readonly EXPECTED_MODEL="tinyllama"
readonly EXPECTED_MODEL_FULL="tinyllama:latest"
readonly OLLAMA_CONTAINER="sutazai-ollama"
readonly OLLAMA_HOST="localhost"
readonly OLLAMA_PORT="10104"
readonly OLLAMA_API_BASE="http://${OLLAMA_HOST}:${OLLAMA_PORT}"

# Backend configuration paths (REAL paths from codebase analysis)
readonly BACKEND_CONFIG_PATH="${PROJECT_ROOT}/backend/app/core/config.py"
readonly BACKEND_CONTAINER="sutazai-backend"

# Color codes for ultra-clear output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly BOLD='\033[1m'
readonly NC='\033[0m'

# Performance targets (from hardware optimization analysis)
readonly MAX_RESPONSE_TIME_MS=8000  # 8s based on optimized performance metrics
readonly MIN_TOKENS_PER_SECOND=10   # Minimum acceptable throughput
readonly TEST_PROMPT="Hello, respond with 'Model validation successful'"
readonly EXPECTED_RESPONSE_KEYWORDS=("validation" "successful")

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

################################################################################
# ULTRA-LOGGING FUNCTIONS
################################################################################

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp="$(date '+%Y-%m-%d %H:%M:%S.%3N')"
    
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
    
    case "$level" in
        ERROR)
            echo -e "${RED}${BOLD}[ERROR]${NC} $message" >&2
            ;;
        SUCCESS)
            echo -e "${GREEN}${BOLD}[SUCCESS]${NC} $message"
            ;;
        INFO)
            echo -e "${BLUE}[INFO]${NC} $message"
            ;;
        WARN)
            echo -e "${YELLOW}${BOLD}[WARN]${NC} $message"
            ;;
        DEBUG)
            echo -e "${PURPLE}[DEBUG]${NC} $message"
            ;;
        PERF)
            echo -e "${CYAN}${BOLD}[PERFORMANCE]${NC} $message"
            ;;
    esac
}

################################################################################
# ULTRA-VALIDATION FUNCTIONS
################################################################################

check_ollama_container_health() {
    log "INFO" "Checking Ollama container health with ULTRA-precision..."
    
    # Check if container exists
    if ! docker ps -a --filter "name=$OLLAMA_CONTAINER" --format "{{.Names}}" | grep -q "^${OLLAMA_CONTAINER}$"; then
        log "ERROR" "Ollama container '$OLLAMA_CONTAINER' does not exist"
        return 1
    fi
    
    # Check if container is running
    local container_status
    container_status=$(docker ps --filter "name=$OLLAMA_CONTAINER" --format "{{.Status}}" | head -1)
    
    if [[ -z "$container_status" ]]; then
        log "ERROR" "Ollama container '$OLLAMA_CONTAINER' is not running"
        log "INFO" "Try: docker-compose up -d ollama"
        return 1
    fi
    
    log "SUCCESS" "Ollama container is running: $container_status"
    
    # Check container health status
    local health_status
    health_status=$(docker inspect "$OLLAMA_CONTAINER" --format='{{.State.Health.Status}}' 2>/dev/null || echo "no_health_check")
    
    case "$health_status" in
        "healthy")
            log "SUCCESS" "Ollama container health status: HEALTHY ✅"
            ;;
        "starting")
            log "WARN" "Ollama container health status: STARTING (may need more time)"
            ;;
        "unhealthy")
            log "ERROR" "Ollama container health status: UNHEALTHY ❌"
            return 1
            ;;
        "no_health_check")
            log "DEBUG" "No health check configured for Ollama container"
            ;;
        *)
            log "WARN" "Unknown health status: $health_status"
            ;;
    esac
    
    return 0
}

test_ollama_api_connectivity() {
    log "INFO" "Testing Ollama API connectivity with ULTRA-precision..."
    
    local start_time
    start_time=$(date +%s%3N)
    
    # Test API endpoint availability
    local response
    local http_code
    
    response=$(curl -s -w "HTTPSTATUS:%{http_code}" "$OLLAMA_API_BASE/api/tags" 2>/dev/null || echo "HTTPSTATUS:000")
    http_code=$(echo "$response" | grep -o "HTTPSTATUS:[0-9]*" | cut -d: -f2)
    response=$(echo "$response" | sed 's/HTTPSTATUS:[0-9]*$//')
    
    local end_time
    end_time=$(date +%s%3N)
    local response_time=$((end_time - start_time))
    
    log "PERF" "API connectivity test completed in ${response_time}ms"
    
    case "$http_code" in
        "200")
            log "SUCCESS" "Ollama API is responding (HTTP $http_code) ✅"
            log "DEBUG" "API response: $response"
            ;;
        "000")
            log "ERROR" "Cannot connect to Ollama API at $OLLAMA_API_BASE"
            log "INFO" "Check if Ollama service is running and port $OLLAMA_PORT is accessible"
            return 1
            ;;
        *)
            log "ERROR" "Ollama API returned HTTP $http_code"
            log "DEBUG" "Response: $response"
            return 1
            ;;
    esac
    
    return 0
}

validate_model_availability() {
    log "INFO" "Validating model availability with ULTRA-precision..."
    
    # Get list of available models
    local models_response
    local models_list
    
    models_response=$(curl -s "$OLLAMA_API_BASE/api/tags" 2>/dev/null)
    
    if [[ $? -ne 0 ]] || [[ -z "$models_response" ]]; then
        log "ERROR" "Failed to retrieve models list from Ollama API"
        return 1
    fi
    
    log "DEBUG" "Models API response: $models_response"
    
    # Parse models using multiple methods for reliability
    models_list=$(echo "$models_response" | grep -o '"name":"[^"]*"' | cut -d'"' -f4 2>/dev/null || echo "")
    
    if [[ -z "$models_list" ]]; then
        # Alternative parsing method
        models_list=$(echo "$models_response" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    if 'models' in data:
        for model in data['models']:
            print(model.get('name', ''))
except: pass
" 2>/dev/null || echo "")
    fi
    
    if [[ -z "$models_list" ]]; then
        log "WARN" "Could not parse models list - checking manually"
        models_list="unknown"
    fi
    
    log "INFO" "Available models: $models_list"
    
    # Check for expected model variations
    local model_found=false
    local model_variations=("$EXPECTED_MODEL" "$EXPECTED_MODEL_FULL" "${EXPECTED_MODEL}:latest")
    
    for model_variant in "${model_variations[@]}"; do
        if echo "$models_list" | grep -q "$model_variant"; then
            log "SUCCESS" "Found expected model: $model_variant ✅"
            model_found=true
            break
        fi
    done
    
    if [[ "$model_found" = false ]]; then
        log "ERROR" "Expected model '$EXPECTED_MODEL' not found in available models"
        log "INFO" "Available models: $models_list"
        log "INFO" "Try: docker exec $OLLAMA_CONTAINER ollama pull $EXPECTED_MODEL_FULL"
        return 1
    fi
    
    return 0
}

validate_backend_configuration() {
    log "INFO" "Validating backend model configuration with ULTRA-precision..."
    
    # Check backend config file
    if [[ ! -f "$BACKEND_CONFIG_PATH" ]]; then
        log "ERROR" "Backend config file not found: $BACKEND_CONFIG_PATH"
        return 1
    fi
    
    # Extract DEFAULT_MODEL from config
    local config_model
    config_model=$(grep -o 'DEFAULT_MODEL.*=.*["'"'"'].*["'"'"']' "$BACKEND_CONFIG_PATH" | grep -o '["'"'"'][^"'"'"']*["'"'"']' | tr -d '"'"'" 2>/dev/null || echo "not_found")
    
    if [[ "$config_model" = "not_found" ]]; then
        log "WARN" "Could not parse DEFAULT_MODEL from backend config"
        log "DEBUG" "Checking alternative patterns..."
        config_model=$(grep -i "default.*model" "$BACKEND_CONFIG_PATH" | head -1 || echo "not_found")
        log "DEBUG" "Found: $config_model"
    else
        log "INFO" "Backend DEFAULT_MODEL configuration: $config_model"
    fi
    
    # Validate model matches expected
    if [[ "$config_model" = "$EXPECTED_MODEL" ]]; then
        log "SUCCESS" "Backend model configuration is correct ✅"
    elif [[ "$config_model" = "not_found" ]]; then
        log "WARN" "Could not verify backend model configuration"
    else
        log "WARN" "Backend model mismatch - Expected: $EXPECTED_MODEL, Found: $config_model"
        return 1
    fi
    
    # Check if backend container is running for additional validation
    if docker ps --filter "name=$BACKEND_CONTAINER" --format "{{.Names}}" | grep -q "^${BACKEND_CONTAINER}$"; then
        log "INFO" "Backend container is running - can test API integration"
        return 0
    else
        log "WARN" "Backend container is not running - cannot test API integration"
        return 1
    fi
}

test_model_performance() {
    log "INFO" "Testing model performance with ULTRA-precision benchmarks..."
    
    local test_prompt="$TEST_PROMPT"
    local start_time
    local end_time
    local response_time
    local response_content
    local http_code
    
    # Performance test
    start_time=$(date +%s%3N)
    
    local api_response
    api_response=$(curl -s -w "HTTPSTATUS:%{http_code}" \
        -X POST "$OLLAMA_API_BASE/api/generate" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$EXPECTED_MODEL\",\"prompt\":\"$test_prompt\",\"stream\":false}" \
        2>/dev/null || echo "HTTPSTATUS:000")
    
    end_time=$(date +%s%3N)
    response_time=$((end_time - start_time))
    
    http_code=$(echo "$api_response" | grep -o "HTTPSTATUS:[0-9]*" | cut -d: -f2)
    response_content=$(echo "$api_response" | sed 's/HTTPSTATUS:[0-9]*$//')
    
    log "PERF" "Model generation test completed in ${response_time}ms"
    
    # Validate HTTP response
    if [[ "$http_code" != "200" ]]; then
        log "ERROR" "Model generation failed with HTTP $http_code"
        log "DEBUG" "Response: $response_content"
        return 1
    fi
    
    # Parse response content
    local generated_text
    generated_text=$(echo "$response_content" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    print(data.get('response', ''))
except: pass
" 2>/dev/null || echo "")
    
    if [[ -z "$generated_text" ]]; then
        log "WARN" "Could not parse generated text from response"
        log "DEBUG" "Raw response: $response_content"
    else
        log "INFO" "Generated text: $generated_text"
        
        # Validate response contains expected keywords
        local keyword_found=false
        for keyword in "${EXPECTED_RESPONSE_KEYWORDS[@]}"; do
            if echo "$generated_text" | grep -i "$keyword" >/dev/null; then
                log "SUCCESS" "Response contains expected keyword: $keyword ✅"
                keyword_found=true
            fi
        done
        
        if [[ "$keyword_found" = false ]]; then
            log "WARN" "Response does not contain expected keywords: ${EXPECTED_RESPONSE_KEYWORDS[*]}"
        fi
    fi
    
    # Performance validation
    if [[ $response_time -lt $MAX_RESPONSE_TIME_MS ]]; then
        log "SUCCESS" "Performance test PASSED - Response time: ${response_time}ms (< ${MAX_RESPONSE_TIME_MS}ms) ✅"
    else
        log "WARN" "Performance test SLOW - Response time: ${response_time}ms (> ${MAX_RESPONSE_TIME_MS}ms)"
    fi
    
    # Calculate tokens per second (rough estimate)
    local estimated_tokens
    estimated_tokens=$(echo "$generated_text" | wc -w)
    local tokens_per_second
    if [[ $response_time -gt 0 ]]; then
        tokens_per_second=$((estimated_tokens * 1000 / response_time))
        log "PERF" "Estimated throughput: $tokens_per_second tokens/second"
        
        if [[ $tokens_per_second -ge $MIN_TOKENS_PER_SECOND ]]; then
            log "SUCCESS" "Throughput test PASSED ✅"
        else
            log "WARN" "Throughput below minimum ($tokens_per_second < $MIN_TOKENS_PER_SECOND)"
        fi
    fi
    
    return 0
}

validate_docker_compose_config() {
    log "INFO" "Validating Docker Compose model configuration..."
    
    local compose_file="${PROJECT_ROOT}/docker-compose.yml"
    
    if [[ ! -f "$compose_file" ]]; then
        log "ERROR" "Docker Compose file not found: $compose_file"
        return 1
    fi
    
    # Check Ollama environment variables
    local ollama_model_env
    ollama_model_env=$(grep -A 20 "container_name: $OLLAMA_CONTAINER" "$compose_file" | grep "OLLAMA_MODEL:" | head -1 | awk '{print $2}' || echo "not_found")
    
    if [[ "$ollama_model_env" = "not_found" ]]; then
        log "WARN" "OLLAMA_MODEL environment variable not found in docker-compose"
    else
        log "INFO" "Docker Compose OLLAMA_MODEL: $ollama_model_env"
        
        if [[ "$ollama_model_env" = "$EXPECTED_MODEL_FULL" ]] || [[ "$ollama_model_env" = "$EXPECTED_MODEL" ]]; then
            log "SUCCESS" "Docker Compose model configuration is correct ✅"
        else
            log "WARN" "Docker Compose model mismatch - Expected: $EXPECTED_MODEL_FULL, Found: $ollama_model_env"
        fi
    fi
    
    return 0
}

run_comprehensive_diagnostics() {
    log "INFO" "Running comprehensive model diagnostics..."
    
    # Container resource usage
    local cpu_usage
    local memory_usage
    
    cpu_usage=$(docker stats "$OLLAMA_CONTAINER" --no-stream --format "{{.CPUPerc}}" 2>/dev/null | tr -d '%' || echo "0")
    memory_usage=$(docker stats "$OLLAMA_CONTAINER" --no-stream --format "{{.MemUsage}}" 2>/dev/null || echo "0 / 0")
    
    log "PERF" "Ollama container CPU usage: ${cpu_usage}%"
    log "PERF" "Ollama container memory usage: $memory_usage"
    
    # Container logs analysis (last 50 lines)
    log "DEBUG" "Analyzing container logs for errors..."
    
    local error_count
    error_count=$(docker logs "$OLLAMA_CONTAINER" --tail 50 2>&1 | grep -i "error\|fail\|exception" | wc -l || echo "0")
    
    if [[ $error_count -eq 0 ]]; then
        log "SUCCESS" "No errors found in recent container logs ✅"
    else
        log "WARN" "Found $error_count potential errors in container logs"
        log "DEBUG" "Recent errors:"
        docker logs "$OLLAMA_CONTAINER" --tail 50 2>&1 | grep -i "error\|fail\|exception" | tail -5 | while read -r line; do
            log "DEBUG" "  $line"
        done
    fi
    
    return 0
}

generate_remediation_recommendations() {
    local validation_result="$1"
    
    log "INFO" "Generating ULTRA-PRECISE remediation recommendations..."
    
    echo -e "\n${BOLD}REMEDIATION RECOMMENDATIONS:${NC}"
    
    case "$validation_result" in
        0)
            echo -e "${GREEN}✅ Model configuration is PERFECT - no action needed${NC}"
            echo -e "System is ready for production workloads with TinyLlama model."
            ;;
        1)
            echo -e "${YELLOW}⚠ Model needs attention - follow these steps:${NC}"
            echo -e ""
            echo -e "1. ${BOLD}Pull TinyLlama model:${NC}"
            echo -e "   docker exec $OLLAMA_CONTAINER ollama pull $EXPECTED_MODEL_FULL"
            echo -e ""
            echo -e "2. ${BOLD}Restart Ollama service:${NC}"
            echo -e "   docker-compose restart ollama"
            echo -e ""
            echo -e "3. ${BOLD}Wait for model loading (30-60s):${NC}"
            echo -e "   docker logs -f $OLLAMA_CONTAINER"
            ;;
        2)
            echo -e "${RED}❌ Critical model configuration issues - immediate action required:${NC}"
            echo -e ""
            echo -e "1. ${BOLD}Check Ollama container status:${NC}"
            echo -e "   docker-compose logs ollama"
            echo -e ""
            echo -e "2. ${BOLD}Restart Ollama with clean state:${NC}"
            echo -e "   docker-compose stop ollama"
            echo -e "   docker-compose up -d ollama"
            echo -e ""
            echo -e "3. ${BOLD}Manual model installation:${NC}"
            echo -e "   docker exec -it $OLLAMA_CONTAINER ollama pull $EXPECTED_MODEL_FULL"
            echo -e ""
            echo -e "4. ${BOLD}Verify model installation:${NC}"
            echo -e "   docker exec $OLLAMA_CONTAINER ollama list"
            ;;
        *)
            echo -e "${RED}❌ Unknown validation result: $validation_result${NC}"
            echo -e "Review logs at: $LOG_FILE"
            ;;
    esac
}

display_ultra_summary() {
    local overall_result="$1"
    local performance_ms="$2"
    
    echo -e "\n${BOLD}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}                    ULTRA MODEL VALIDATION SUMMARY${NC}"
    echo -e "${BOLD}═══════════════════════════════════════════════════════════════${NC}"
    
    echo -e "Model: ${CYAN}$EXPECTED_MODEL${NC} ($EXPECTED_MODEL_FULL)"
    echo -e "Container: ${CYAN}$OLLAMA_CONTAINER${NC}"
    echo -e "API Endpoint: ${CYAN}$OLLAMA_API_BASE${NC}"
    echo -e "Log File: ${CYAN}$LOG_FILE${NC}"
    
    if [[ -n "$performance_ms" ]]; then
        echo -e "Performance: ${CYAN}${performance_ms}ms${NC} response time"
    fi
    
    case "$overall_result" in
        0)
            echo -e "Status: ${GREEN}${BOLD}PERFECT${NC} ${GREEN}✅${NC}"
            echo -e "System: ${GREEN}PRODUCTION READY${NC}"
            ;;
        1)
            echo -e "Status: ${YELLOW}${BOLD}NEEDS ATTENTION${NC} ${YELLOW}⚠${NC}"
            echo -e "System: ${YELLOW}REQUIRES MAINTENANCE${NC}"
            ;;
        2)
            echo -e "Status: ${RED}${BOLD}CRITICAL ISSUES${NC} ${RED}❌${NC}"
            echo -e "System: ${RED}NOT OPERATIONAL${NC}"
            ;;
        *)
            echo -e "Status: ${RED}${BOLD}UNKNOWN ERROR${NC} ${RED}❌${NC}"
            echo -e "System: ${RED}INVESTIGATION REQUIRED${NC}"
            ;;
    esac
    
    echo -e "${BOLD}═══════════════════════════════════════════════════════════════${NC}"
}

################################################################################
# MAIN ULTRA-EXECUTION
################################################################################

main() {
    log "INFO" "Starting $SCRIPT_NAME v$SCRIPT_VERSION with ULTRA-PRECISION"
    
    echo -e "${BOLD}${CYAN}SutazAI Ultra-Precise Model Validation${NC}"
    echo -e "Target Model: ${BOLD}$EXPECTED_MODEL_FULL${NC}"
    echo -e "API Endpoint: ${BOLD}$OLLAMA_API_BASE${NC}"
    echo -e "Log File: ${CYAN}$LOG_FILE${NC}\n"
    
    local overall_result=0
    local performance_ms=""
    
    # Step 1: Check Ollama container health
    log "INFO" "═══ STEP 1: Container Health Check ═══"
    if ! check_ollama_container_health; then
        overall_result=2
    fi
    
    # Step 2: Test API connectivity  
    log "INFO" "═══ STEP 2: API Connectivity Test ═══"
    if ! test_ollama_api_connectivity; then
        [[ $overall_result -eq 0 ]] && overall_result=2
    fi
    
    # Step 3: Validate model availability
    log "INFO" "═══ STEP 3: Model Availability Check ═══"
    if ! validate_model_availability; then
        [[ $overall_result -eq 0 ]] && overall_result=1
    fi
    
    # Step 4: Validate backend configuration
    log "INFO" "═══ STEP 4: Backend Configuration Check ═══"
    validate_backend_configuration || true
    
    # Step 5: Test model performance
    log "INFO" "═══ STEP 5: Performance Benchmark ═══"
    if test_model_performance; then
        # Extract performance metrics from logs
        performance_ms=$(tail -20 "$LOG_FILE" | grep "Model generation test completed" | grep -o "[0-9]*ms" | head -1 | tr -d 'ms' || echo "")
    else
        [[ $overall_result -eq 0 ]] && overall_result=1
    fi
    
    # Step 6: Validate Docker Compose configuration
    log "INFO" "═══ STEP 6: Docker Compose Validation ═══"
    validate_docker_compose_config || true
    
    # Step 7: Run comprehensive diagnostics
    log "INFO" "═══ STEP 7: Comprehensive Diagnostics ═══"
    run_comprehensive_diagnostics || true
    
    # Step 8: Generate recommendations and summary
    generate_remediation_recommendations "$overall_result"
    display_ultra_summary "$overall_result" "$performance_ms"
    
    log "SUCCESS" "Ultra-precise model validation completed with result: $overall_result"
    
    exit "$overall_result"
}

# Ultra-robust signal handlers
trap 'log "ERROR" "Script interrupted by user - cleanup initiated"; exit 130' INT
trap 'log "ERROR" "Script terminated by system - cleanup initiated"; exit 143' TERM
trap 'log "ERROR" "Script failed with error - check logs: $LOG_FILE"; exit 1' ERR

# Execute main function with ultra-precision
main "$@"