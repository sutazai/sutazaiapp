#!/bin/bash

# Security Validation Script: Verify Non-Root Container Implementation
# This script validates that containers are running with non-root users

set -e

# Colors for output

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
NC='\033[0m' # No Color

# Configuration
REPORT_FILE="/opt/sutazaiapp/logs/security-validation-$(date +%Y%m%d-%H%M%S).txt"

# Create log directory
mkdir -p "$(dirname "$REPORT_FILE")"

# Header
echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}Container Security Validation Report${NC}"
echo -e "${BLUE}Date: $(date)${NC}"
echo -e "${BLUE}=========================================${NC}"

# Initialize counters
TOTAL_CONTAINERS=0
ROOT_CONTAINERS=0
NONROOT_CONTAINERS=0
ERROR_CONTAINERS=0

# Arrays to store container details
declare -a ROOT_LIST
declare -a NONROOT_LIST
declare -a ERROR_LIST

# Function to check container security
check_container_security() {
    local container=$1
    local user=$(docker exec "$container" whoami 2>/dev/null || echo "ERROR")
    local uid=$(docker exec "$container" id -u 2>/dev/null || echo "ERROR")
    local gid=$(docker exec "$container" id -g 2>/dev/null || echo "ERROR")
    
    TOTAL_CONTAINERS=$((TOTAL_CONTAINERS + 1))
    
    if [ "$user" = "root" ] || [ "$uid" = "0" ]; then
        echo -e "  ${RED}✗${NC} $container"
        echo -e "    User: ${RED}$user${NC} (UID: $uid, GID: $gid)"
        ROOT_CONTAINERS=$((ROOT_CONTAINERS + 1))
        ROOT_LIST+=("$container (user: $user, uid: $uid)")
        
        # Additional security checks for root containers
        local caps=$(docker inspect --format='{{.HostConfig.CapAdd}}' "$container" 2>/dev/null || echo "[]")
        local priv=$(docker inspect --format='{{.HostConfig.Privileged}}' "$container" 2>/dev/null || echo "false")
        local readonly=$(docker inspect --format='{{.HostConfig.ReadonlyRootfs}}' "$container" 2>/dev/null || echo "false")
        
        echo -e "    Capabilities: $caps"
        echo -e "    Privileged: $priv"
        echo -e "    Read-only root: $readonly"
        
    elif [ "$user" = "ERROR" ]; then
        echo -e "  ${YELLOW}⚠${NC} $container"
        echo -e "    ${YELLOW}Unable to determine user${NC}"
        ERROR_CONTAINERS=$((ERROR_CONTAINERS + 1))
        ERROR_LIST+=("$container")
        
    else
        echo -e "  ${GREEN}✓${NC} $container"
        echo -e "    User: ${GREEN}$user${NC} (UID: $uid, GID: $gid)"
        NONROOT_CONTAINERS=$((NONROOT_CONTAINERS + 1))
        NONROOT_LIST+=("$container (user: $user, uid: $uid)")
        
        # Check additional security features
        local caps=$(docker inspect --format='{{.HostConfig.CapDrop}}' "$container" 2>/dev/null || echo "[]")
        local secopt=$(docker inspect --format='{{.HostConfig.SecurityOpt}}' "$container" 2>/dev/null || echo "[]")
        local readonly=$(docker inspect --format='{{.HostConfig.ReadonlyRootfs}}' "$container" 2>/dev/null || echo "false")
        
        if [ "$caps" != "[]" ]; then
            echo -e "    Dropped capabilities: ${GREEN}$caps${NC}"
        fi
        if [ "$secopt" != "[]" ]; then
            echo -e "    Security options: ${GREEN}$secopt${NC}"
        fi
        if [ "$readonly" = "true" ]; then
            echo -e "    Read-only filesystem: ${GREEN}enabled${NC}"
        fi
    fi
    
    # Check for security options
    local no_new_privs=$(docker inspect --format='{{range .HostConfig.SecurityOpt}}{{.}} {{end}}' "$container" 2>/dev/null | grep -c "no-new-privileges" || echo "0")
    if [ "$no_new_privs" -gt 0 ]; then
        echo -e "    No new privileges: ${GREEN}enabled${NC}"
    fi
    
    echo ""
}

# Main validation
echo -e "\n${BLUE}Container User Analysis:${NC}"
echo -e "${BLUE}------------------------${NC}\n"

for container in $(docker ps --format "{{.Names}}" | sort); do
    check_container_security "$container"
done

# Volume permissions check
echo -e "${BLUE}Volume Permissions Check:${NC}"
echo -e "${BLUE}------------------------${NC}\n"

# Check volume ownership
for volume in $(docker volume ls -q | grep -E "postgres|redis|neo4j|ollama|chromadb|qdrant|rabbitmq"); do
    echo -e "Checking volume: $volume"
    
    # Create temporary container to check permissions
    owner=$(docker run --rm -v "$volume:/data" alpine:latest stat -c "%u:%g" /data 2>/dev/null || echo "ERROR")
    
    if [ "$owner" = "0:0" ]; then
        echo -e "  ${YELLOW}⚠${NC} Owned by root (0:0)"
    elif [ "$owner" = "ERROR" ]; then
        echo -e "  ${RED}✗${NC} Unable to check ownership"
    else
        echo -e "  ${GREEN}✓${NC} Owned by $owner"
    fi
done

echo ""

# Network exposure check
echo -e "${BLUE}Network Exposure Analysis:${NC}"
echo -e "${BLUE}------------------------${NC}\n"

for container in $(docker ps --format "{{.Names}}" | sort); do
    ports=$(docker inspect --format='{{range $p, $conf := .NetworkSettings.Ports}}{{$p}} {{end}}' "$container" 2>/dev/null)
    
    if [ -n "$ports" ]; then
        echo -e "$container:"
        for port in $ports; do
            # Check if port is below 1024 (privileged)
            port_num=$(echo "$port" | cut -d'/' -f1)
            if [ "$port_num" -lt 1024 ] 2>/dev/null; then
                echo -e "  ${YELLOW}⚠${NC} Privileged port: $port"
            else
                echo -e "  ${GREEN}✓${NC} Non-privileged port: $port"
            fi
        done
    fi
done

echo ""

# Security capabilities check
echo -e "${BLUE}Security Capabilities Analysis:${NC}"
echo -e "${BLUE}------------------------------${NC}\n"

for container in $(docker ps --format "{{.Names}}" | sort); do
    caps_add=$(docker inspect --format='{{.HostConfig.CapAdd}}' "$container" 2>/dev/null)
    caps_drop=$(docker inspect --format='{{.HostConfig.CapDrop}}' "$container" 2>/dev/null)
    privileged=$(docker inspect --format='{{.HostConfig.Privileged}}' "$container" 2>/dev/null)
    
    if [ "$privileged" = "true" ]; then
        echo -e "${RED}✗${NC} $container: Running in privileged mode!"
    elif [ "$caps_add" != "[]" ] && [ "$caps_add" != "<no value>" ]; then
        echo -e "${YELLOW}⚠${NC} $container: Additional capabilities: $caps_add"
    elif [ "$caps_drop" != "[]" ] && [ "$caps_drop" != "<no value>" ]; then
        echo -e "${GREEN}✓${NC} $container: Dropped capabilities: $caps_drop"
    fi
done

echo ""

# Summary Report
echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}Security Validation Summary${NC}"
echo -e "${BLUE}=========================================${NC}\n"

echo -e "Total Containers: $TOTAL_CONTAINERS"
echo -e "Non-Root Containers: ${GREEN}$NONROOT_CONTAINERS${NC}"
echo -e "Root Containers: ${RED}$ROOT_CONTAINERS${NC}"
echo -e "Unknown Status: ${YELLOW}$ERROR_CONTAINERS${NC}"

echo ""

# Calculate security score
SECURITY_SCORE=$((NONROOT_CONTAINERS * 100 / TOTAL_CONTAINERS))

echo -e "Security Score: "
if [ $SECURITY_SCORE -ge 90 ]; then
    echo -e "${GREEN}$SECURITY_SCORE%${NC} - Excellent"
elif [ $SECURITY_SCORE -ge 70 ]; then
    echo -e "${GREEN}$SECURITY_SCORE%${NC} - Good"
elif [ $SECURITY_SCORE -ge 50 ]; then
    echo -e "${YELLOW}$SECURITY_SCORE%${NC} - Needs Improvement"
else
    echo -e "${RED}$SECURITY_SCORE%${NC} - Critical"
fi

echo ""

# Detailed lists
if [ ${#ROOT_LIST[@]} -gt 0 ]; then
    echo -e "${RED}Containers Running as Root:${NC}"
    for item in "${ROOT_LIST[@]}"; do
        echo "  - $item"
    done
    echo ""
fi

if [ ${#ERROR_LIST[@]} -gt 0 ]; then
    echo -e "${YELLOW}Containers with Unknown Status:${NC}"
    for item in "${ERROR_LIST[@]}"; do
        echo "  - $item"
    done
    echo ""
fi

# Recommendations
echo -e "${BLUE}Security Recommendations:${NC}"
echo -e "${BLUE}------------------------${NC}\n"

if [ $ROOT_CONTAINERS -gt 0 ]; then
    echo -e "${RED}Critical:${NC}"
    echo "  - Migrate $ROOT_CONTAINERS containers from root to non-root users"
    echo "  - Run: /opt/sutazaiapp/scripts/security/migrate_to_nonroot.sh"
    echo ""
fi

echo -e "${YELLOW}Recommended:${NC}"
echo "  - Enable 'no-new-privileges' security option for all containers"
echo "  - Drop unnecessary Linux capabilities"
echo "  - Use read-only root filesystem where possible"
echo "  - Implement network policies to restrict container communication"
echo "  - Regular security scanning with tools like Trivy or Clair"

echo ""

# Save report
{
    echo "Container Security Validation Report"
    echo "===================================="
    echo "Date: $(date)"
    echo ""
    echo "Summary:"
    echo "  Total Containers: $TOTAL_CONTAINERS"
    echo "  Non-Root: $NONROOT_CONTAINERS"
    echo "  Root: $ROOT_CONTAINERS"
    echo "  Unknown: $ERROR_CONTAINERS"
    echo "  Security Score: $SECURITY_SCORE%"
    echo ""
    
    if [ ${#ROOT_LIST[@]} -gt 0 ]; then
        echo "Root Containers:"
        for item in "${ROOT_LIST[@]}"; do
            echo "  - $item"
        done
        echo ""
    fi
    
    if [ ${#NONROOT_LIST[@]} -gt 0 ]; then
        echo "Non-Root Containers:"
        for item in "${NONROOT_LIST[@]}"; do
            echo "  - $item"
        done
    fi
} > "$REPORT_FILE"

echo -e "${GREEN}Report saved to: $REPORT_FILE${NC}"

# Exit with appropriate code
if [ $ROOT_CONTAINERS -eq 0 ]; then
    exit 0  # Success - no root containers
else
    exit 1  # Warning - root containers found
fi