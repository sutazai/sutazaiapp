#!/usr/bin/env bash
# Kong API Gateway Configuration Script
# Description: Idempotent Kong service/route configuration via Kong Admin API
# Usage: scripts/configure_kong.sh <service_name> <path_prefix>
# Example: scripts/configure_kong.sh backend-api /api/v1
#
# This script:
# 1. Accepts CLI arguments: service_name, path_prefix
# 2. Checks if Kong Service/Route already exist (idempotency)
# 3. Uses curl to create Kong Service pointing to Consul service discovery
# 4. Creates Kong Route with path prefix
# 5. Logs each step with timestamps
# 6. Includes comprehensive error handling

set -euo pipefail

# Configuration
KONG_ADMIN_URL=${KONG_ADMIN_URL:-"http://localhost:8001"}
CONSUL_DOMAIN=${CONSUL_DOMAIN:-"service.consul"}
DEFAULT_SERVICE_PORT=${DEFAULT_SERVICE_PORT:-"8080"}
LOG_LEVEL=${LOG_LEVEL:-"INFO"}

# Logging function with timestamp and level
log() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S UTC')
    echo "[${level}] ${timestamp} - ${message}" >&2
}

# Error handler that logs and exits
die() {
    log "ERROR" "$1"
    exit 1
}

# Input validation function
validate_inputs() {
    if [[ $# -ne 2 ]]; then
        echo "Usage: $0 <service_name> <path_prefix>" >&2
        echo ""
        echo "Arguments:"
        echo "  service_name  - Name of the service to register with Kong"
        echo "  path_prefix   - URL path prefix for routing (e.g., /api/v1)"
        echo ""
        echo "Environment Variables:"
        echo "  KONG_ADMIN_URL      - Kong Admin API URL (default: http://localhost:8001)"
        echo "  CONSUL_DOMAIN       - Consul service domain (default: service.consul)"
        echo "  DEFAULT_SERVICE_PORT- Default service port (default: 8080)"
        echo ""
        echo "Examples:"
        echo "  $0 backend-api /api/v1"
        echo "  $0 user-service /users"
        exit 2
    fi

    local service_name="$1"
    local path_prefix="$2"

    if [[ -z "$service_name" ]]; then
        die "service_name cannot be empty"
    fi

    if [[ -z "$path_prefix" ]]; then
        die "path_prefix cannot be empty"
    fi

    # Validate path_prefix format
    if [[ ! "$path_prefix" =~ ^/.+ ]]; then
        die "path_prefix must start with '/' (e.g., /api/v1)"
    fi

    # Validate service_name format (no spaces, special characters)
    if [[ ! "$service_name" =~ ^[a-zA-Z0-9_-]+$ ]]; then
        die "service_name must contain only alphanumeric characters, hyphens, and underscores"
    fi

    log "INFO" "Input validation passed - service: $service_name, path: $path_prefix"
}

# Check Kong Admin API connectivity
check_kong_connectivity() {
    log "INFO" "Checking Kong Admin API connectivity at $KONG_ADMIN_URL"
    
    if ! curl -sSf --connect-timeout 5 --max-time 10 "$KONG_ADMIN_URL/status" >/dev/null 2>&1; then
        die "Kong Admin API is not accessible at $KONG_ADMIN_URL. Please ensure Kong is running and accessible."
    fi
    
    log "INFO" "Kong Admin API is accessible"
}

# Check if Kong service exists
check_service_exists() {
    local service_name="$1"
    local http_code
    
    log "INFO" "Checking if Kong service '$service_name' exists"
    
    http_code=$(curl -sSf -o /dev/null -w "%{http_code}" "$KONG_ADMIN_URL/services/$service_name" 2>/dev/null || echo "000")
    
    case "$http_code" in
        200)
            log "INFO" "Service '$service_name' already exists"
            return 0
            ;;
        404)
            log "INFO" "Service '$service_name' does not exist"
            return 1
            ;;
        *)
            die "Unexpected HTTP response $http_code when checking service '$service_name'"
            ;;
    esac
}

# Create or update Kong service
create_or_update_service() {
    local service_name="$1"
    local service_url="http://${service_name}.${CONSUL_DOMAIN}:${DEFAULT_SERVICE_PORT}"
    
    log "INFO" "Creating/updating Kong service '$service_name' -> $service_url"
    
    if check_service_exists "$service_name"; then
        # Update existing service
        log "INFO" "Updating existing service '$service_name'"
        if curl -sSf -X PATCH "$KONG_ADMIN_URL/services/$service_name" \
             -H 'Content-Type: application/json' \
             -d "{\"url\": \"$service_url\"}" >/dev/null; then
            log "INFO" "Service '$service_name' updated successfully"
        else
            die "Failed to update service '$service_name'"
        fi
    else
        # Create new service
        log "INFO" "Creating new service '$service_name'"
        if curl -sSf -X POST "$KONG_ADMIN_URL/services" \
             -H 'Content-Type: application/json' \
             -d "{\"name\": \"$service_name\", \"url\": \"$service_url\"}" >/dev/null; then
            log "INFO" "Service '$service_name' created successfully"
        else
            die "Failed to create service '$service_name'"
        fi
    fi
}

# Check if route exists for service with given path
check_route_exists() {
    local service_name="$1"
    local path_prefix="$2"
    local routes_response
    
    log "INFO" "Checking if route exists for service '$service_name' with path '$path_prefix'"
    
    # Get routes for the service
    if ! routes_response=$(curl -sSf "$KONG_ADMIN_URL/routes?service.name=$service_name" 2>/dev/null); then
        die "Failed to query routes for service '$service_name'"
    fi
    
    # Check if response contains data array
    if ! echo "$routes_response" | grep -q '"data"'; then
        die "Unexpected routes response format from Kong API"
    fi
    
    # Check if any route has the specified path
    if echo "$routes_response" | grep -q "\"paths\":\[\"$path_prefix\"\]"; then
        log "INFO" "Route already exists for path '$path_prefix'"
        return 0
    else
        log "INFO" "No route exists for path '$path_prefix'"
        return 1
    fi
}

# Create Kong route
create_route() {
    local service_name="$1"
    local path_prefix="$2"
    local route_name="${service_name}-route"
    
    log "INFO" "Creating Kong route '$route_name' for service '$service_name' with path '$path_prefix'"
    
    if curl -sSf -X POST "$KONG_ADMIN_URL/routes" \
         -H 'Content-Type: application/json' \
         -d "{
           \"name\": \"$route_name\",
           \"service\": {\"name\": \"$service_name\"},
           \"paths\": [\"$path_prefix\"],
           \"strip_path\": false,
           \"preserve_host\": true
         }" >/dev/null; then
        log "INFO" "Route '$route_name' created successfully"
    else
        die "Failed to create route '$route_name'"
    fi
}

# Configure Kong route (create if doesn't exist)
configure_route() {
    local service_name="$1"
    local path_prefix="$2"
    
    if check_route_exists "$service_name" "$path_prefix"; then
        log "INFO" "Route configuration is already up to date"
    else
        create_route "$service_name" "$path_prefix"
    fi
}

# Verify configuration
verify_configuration() {
    local service_name="$1"
    local path_prefix="$2"
    
    log "INFO" "Verifying Kong configuration for service '$service_name'"
    
    # Verify service exists
    if ! check_service_exists "$service_name"; then
        die "Service verification failed - '$service_name' does not exist"
    fi
    
    # Verify route exists
    if ! check_route_exists "$service_name" "$path_prefix"; then
        die "Route verification failed - route with path '$path_prefix' does not exist"
    fi
    
    log "INFO" "Configuration verification completed successfully"
}

# Main execution function
main() {
    # Step 1: Validate inputs first (handles argument count checking)
    validate_inputs "$@"
    
    local service_name="$1"
    local path_prefix="$2"
    
    log "INFO" "Starting Kong configuration for service: $service_name, path: $path_prefix"
    
    # Step 2: Check Kong connectivity
    check_kong_connectivity
    
    # Step 3: Create or update service
    create_or_update_service "$service_name"
    
    # Step 4: Configure route
    configure_route "$service_name" "$path_prefix"
    
    # Step 5: Verify configuration
    verify_configuration "$service_name" "$path_prefix"
    
    log "INFO" "Kong configuration completed successfully for '$service_name'"
    log "INFO" "Service URL: http://${service_name}.${CONSUL_DOMAIN}:${DEFAULT_SERVICE_PORT}"
    log "INFO" "Route Path: $path_prefix"
    log "INFO" "Access via Kong: http://localhost:10005$path_prefix"
}

# Execute main function with all arguments
main "$@"