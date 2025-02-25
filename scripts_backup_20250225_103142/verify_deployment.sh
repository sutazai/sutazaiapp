#!/bin/bash
# SutazAi Deployment Verification

# Get the absolute path to the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Enhanced verification patterns
PATTERNS=(
    'SutazAi'
    'sutazai'
    'SUTAZAI'
    's-'
    '_s_'
    's_'
    'Sutaz'
    'sutaz'
    'SUTAZ'
)

# Add error handling and permission checks
if [ ! -f "${PROJECT_ROOT}/deploy_all.sh" ]; then
    echo "❌ deploy_all.sh not found in ${PROJECT_ROOT}"
    exit 1
fi
if [ ! -x "${PROJECT_ROOT}/deploy_all.sh" ]; then
    echo "❌ deploy_all.sh not executable"
    exit 1
fi

check_naming() {
    local search_path="${PROJECT_ROOT}"
    for pattern in "${PATTERNS[@]}"; do
        if grep -rni --exclude-dir={.venv,node_modules,.git} --binary-files=without-match \
            -e "${pattern}" "$search_path"; then
            echo "❌ Invalid naming pattern found: ${pattern} references found"
            return 1
        fi
    done
    return 0
}

# Add logging function if not already defined
if ! declare -F log_error > /dev/null; then
    log_error() {
        echo "ERROR: $*" >&2
    }
fi

check_services() {
    if ! docker ps | grep -q sutazai-core; then
        log_error "Core service not running"
        return 1
    fi
    if ! curl -sSf http://localhost:3000/api/health > /dev/null; then
        log_error "Health check failed"
        return 1
    fi
    return 0
}

check_security() {
    # Verify file permissions
    local secure_files=(
        "/etc/sutazai/database.conf"
        "/etc/sutazai/api_keys.conf"
    )
    
    for file in "${secure_files[@]}"; do
        if [ ! -f "$file" ]; then
            log_error "Security file $file not found"
            return 1
        fi
        if [[ $(stat -c %a "$file") != "600" ]]; then
            log_error "Insecure permissions for $file"
            return 1
        fi
    done
    
    # Verify no world-writable files in important directories
    local world_writable
    world_writable=$(find "${PROJECT_ROOT}" -type f -perm -0002 2>/dev/null)
    if [[ -n "$world_writable" ]]; then
        log_error "World-writable files found:\n$world_writable"
        return 1
    fi
    return 0
}

verify_services() {
    local services=("sutazai-ui" "sutazai-api" "sutazai-worker")
    local failed=0
    
    for service in "${services[@]}"; do
        if ! systemctl is-active --quiet "$service"; then
            log_error "Service $service is not running"
            failed=1
        fi
    done
    return $failed
}

verify_resources() {
    # Source config if MIN_CPU_CORES not defined
    if [ -z "${MIN_CPU_CORES:-}" ] && [ -f "${PROJECT_ROOT}/.deployment_config" ]; then
        source "${PROJECT_ROOT}/.deployment_config"
    fi
    
    local cpu=$(nproc)
    local memory=$(free -g | awk '/^Mem:/{print $2}')
    
    if (( cpu < ${MIN_CPU_CORES%$'\r'} )) || (( memory < ${MIN_MEMORY%G} )); then
        log_error "Insufficient resources after deployment"
        return 1
    fi
    return 0
}

verify_deployment() {
    local max_retries=5
    local retry_count=0
    local health_endpoint="http://localhost:8080/health"
    
    while [[ $retry_count -lt $max_retries ]]; do
        if curl -sSf "$health_endpoint" > /dev/null; then
            echo "Deployment verified successfully"
            return 0
        else
            retry_count=$((retry_count+1))
            echo "Verification failed, retrying ($retry_count/$max_retries)..."
            sleep 10
        fi
    done
    
    log_error "Max verification retries reached. Deployment may have failed."
    return 1
}

check_directories() {
    local required_dirs=(
        "/var/log/sutazai"
        "/etc/sutazai"
        "/var/backups/sutazai"
    )
    
    for dir in "${required_dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            log_error "Required directory $dir not found"
            return 1
        fi
        
        # Check directory permissions
        if [[ $(stat -c %a "$dir") != "750" ]]; then
            log_error "Incorrect permissions on $dir"
            return 1
        fi
    done
    return 0
}

main() {
    local failed=0
    
    echo "Starting deployment verification..."
    
    check_directories || failed=1
    check_naming || failed=1
    check_services || failed=1
    check_security || failed=1
    verify_resources || failed=1
    verify_services || failed=1
    verify_deployment || failed=1
    
    if [ $failed -eq 0 ]; then
        echo "✅ All systems verified and operational"
        return 0
    else
        log_error "Deployment verification failed"
        return 1
    fi
}

# Run main if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi 