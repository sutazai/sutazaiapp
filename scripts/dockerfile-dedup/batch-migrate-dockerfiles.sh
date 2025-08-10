#!/bin/bash
# Batch Dockerfile Migration Script
# Safely migrates multiple Dockerfiles to use master base images
# Author: System Architect
# Date: August 10, 2025

set -euo pipefail

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
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
BATCH_SIZE=${1:-5}
CATEGORY=${2:-"python-agents"}
DRY_RUN=${3:-"false"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORT_DIR="/opt/sutazaiapp/reports/dockerfile-dedup"
MIGRATION_LOG="$REPORT_DIR/batch-migration-$(date +%Y%m%d-%H%M%S).log"

# Create directories
mkdir -p "$REPORT_DIR"

echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  Dockerfile Batch Migration Tool${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "Batch Size: ${YELLOW}$BATCH_SIZE${NC}"
echo -e "Category: ${YELLOW}$CATEGORY${NC}"
echo -e "Dry Run: ${YELLOW}$DRY_RUN${NC}"
echo -e "Log: ${YELLOW}$MIGRATION_LOG${NC}"
echo ""

# Initialize counters
TOTAL_PROCESSED=0
SUCCESSFUL_MIGRATIONS=0
FAILED_MIGRATIONS=0
SKIPPED_MIGRATIONS=0

# Function to log messages
log_message() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        "INFO")
            echo -e "${BLUE}[INFO]${NC} $message"
            ;;
        "SUCCESS")
            echo -e "${GREEN}[SUCCESS]${NC} $message"
            ;;
        "WARNING")
            echo -e "${YELLOW}[WARNING]${NC} $message"
            ;;
        "ERROR")
            echo -e "${RED}[ERROR]${NC} $message"
            ;;
    esac
    
    echo "[$timestamp] [$level] $message" >> "$MIGRATION_LOG"
}

# Function to get services for migration
get_migration_candidates() {
    local category=$1
    local candidates=()
    
    case $category in
        "python-agents")
            # Find Python-based agent services
            while IFS= read -r dockerfile; do
                if grep -q "FROM python:3.11" "$dockerfile" 2>/dev/null; then
                    service=$(basename $(dirname "$dockerfile"))
                    # Skip if already migrated
                    if ! grep -q "FROM sutazai-" "$dockerfile" 2>/dev/null; then
                        candidates+=("$service")
                    fi
                fi
            done < <(find /opt/sutazaiapp/docker -name "Dockerfile" -type f)
            ;;
        
        "nodejs-services")
            # Find Node.js services
            while IFS= read -r dockerfile; do
                if grep -q "FROM node:" "$dockerfile" 2>/dev/null; then
                    service=$(basename $(dirname "$dockerfile"))
                    if ! grep -q "FROM sutazai-" "$dockerfile" 2>/dev/null; then
                        candidates+=("$service")
                    fi
                fi
            done < <(find /opt/sutazaiapp/docker -name "Dockerfile" -type f)
            ;;
        
        "ml-heavy")
            # Find ML/AI heavy services
            while IFS= read -r dockerfile; do
                if grep -E "tensorflow|pytorch|transformers" "$dockerfile" 2>/dev/null; then
                    service=$(basename $(dirname "$dockerfile"))
                    if ! grep -q "FROM sutazai-" "$dockerfile" 2>/dev/null; then
                        candidates+=("$service")
                    fi
                fi
            done < <(find /opt/sutazaiapp/docker -name "Dockerfile" -type f)
            ;;
        
        *)
            log_message "ERROR" "Unknown category: $category"
            exit 1
            ;;
    esac
    
    echo "${candidates[@]}"
}

# Function to migrate a single service
migrate_service() {
    local service=$1
    local base_image=$2
    local dockerfile_path="/opt/sutazaiapp/docker/$service/Dockerfile"
    
    if [ ! -f "$dockerfile_path" ]; then
        dockerfile_path="/opt/sutazaiapp/agents/$service/Dockerfile"
    fi
    
    if [ ! -f "$dockerfile_path" ]; then
        log_message "ERROR" "Dockerfile not found for $service"
        return 1
    fi
    
    log_message "INFO" "Migrating $service to use $base_image"
    
    # Create backup
    local backup_file="${dockerfile_path}.backup-$(date +%Y%m%d-%H%M%S)"
    cp "$dockerfile_path" "$backup_file"
    log_message "INFO" "Backup created: $backup_file"
    
    if [ "$DRY_RUN" = "true" ]; then
        log_message "INFO" "DRY RUN: Would migrate $dockerfile_path"
        return 0
    fi
    
    # Create new Dockerfile
    local temp_dockerfile="${dockerfile_path}.new"
    
    {
        echo "# Migrated to use SutazAI base image"
        echo "# Migration date: $(date)"
        echo "# Original base: $(grep '^FROM' "$dockerfile_path" | head -1)"
        echo ""
        echo "FROM $base_image"
        echo ""
        
        # Extract service-specific dependencies
        if [ -f "$(dirname "$dockerfile_path")/requirements.txt" ]; then
            echo "# Service-specific dependencies"
            echo "COPY requirements.txt ."
            echo "RUN pip install --no-cache-dir -r requirements.txt"
            echo ""
        fi
        
        # Copy the rest of the Dockerfile, skipping FROM, user creation, and base packages
        awk '
            /^FROM/ { next }
            /^RUN.*useradd/ { next }
            /^RUN.*groupadd/ { next }
            /^RUN.*apt-get.*update/ { next }
            /^RUN.*pip.*install.*pip/ { next }
            /^USER/ { print "# " $0 " (handled by base image)"; next }
            { print }
        ' "$dockerfile_path"
        
        # Ensure we switch to non-root user if not already done
        if ! grep -q "^USER" "$dockerfile_path"; then
            echo ""
            echo "# Switch to non-root user (from base image)"
            echo "USER appuser"
        fi
        
    } > "$temp_dockerfile"
    
    # Validate new Dockerfile
    if docker build -t "test-migration-$service" -f "$temp_dockerfile" "$(dirname "$dockerfile_path")" > /dev/null 2>&1; then
        log_message "SUCCESS" "Build successful with new base image"
        mv "$temp_dockerfile" "$dockerfile_path"
        docker rmi "test-migration-$service" > /dev/null 2>&1
        return 0
    else
        log_message "ERROR" "Build failed with new base image"
        rm "$temp_dockerfile"
        return 1
    fi
}

# Function to validate migration
validate_migration() {
    local service=$1
    
    log_message "INFO" "Validating migration for $service"
    
    # Run post-migration validation
    if "$SCRIPT_DIR/validate-after-migration.sh" "$service" > /dev/null 2>&1; then
        log_message "SUCCESS" "Validation passed for $service"
        return 0
    else
        log_message "WARNING" "Validation failed for $service"
        return 1
    fi
}

# Main migration process
main() {
    log_message "INFO" "Starting batch migration process"
    
    # Get migration candidates
    log_message "INFO" "Identifying migration candidates for category: $CATEGORY"
    IFS=' ' read -r -a candidates <<< "$(get_migration_candidates "$CATEGORY")"
    
    if [ ${#candidates[@]} -eq 0 ]; then
        log_message "WARNING" "No candidates found for migration"
        exit 0
    fi
    
    log_message "INFO" "Found ${#candidates[@]} candidates for migration"
    
    # Determine base image for category
    case $CATEGORY in
        "python-agents")
            BASE_IMAGE="sutazai-python-agent-master:latest"
            ;;
        "nodejs-services")
            BASE_IMAGE="sutazai-nodejs-agent-master:latest"
            ;;
        "ml-heavy")
            BASE_IMAGE="sutazai-python-ml-heavy:latest"
            ;;
        *)
            BASE_IMAGE="sutazai-python-agent-master:latest"
            ;;
    esac
    
    log_message "INFO" "Using base image: $BASE_IMAGE"
    
    # Process candidates in batches
    local batch_count=0
    for service in "${candidates[@]:0:$BATCH_SIZE}"; do
        ((TOTAL_PROCESSED++))
        ((batch_count++))
        
        echo ""
        echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${CYAN}Processing [$batch_count/$BATCH_SIZE]: $service${NC}"
        echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        
        # Check if already migrated
        if [ -f "$REPORT_DIR/${service}.migrated" ]; then
            log_message "INFO" "Service already migrated, skipping"
            ((SKIPPED_MIGRATIONS++))
            continue
        fi
        
        # Pre-migration validation
        if ! "$SCRIPT_DIR/validate-before-migration.sh" "$service" > /dev/null 2>&1; then
            log_message "WARNING" "Pre-migration validation failed, skipping"
            ((SKIPPED_MIGRATIONS++))
            continue
        fi
        
        # Migrate service
        if migrate_service "$service" "$BASE_IMAGE"; then
            # Post-migration validation
            if validate_migration "$service"; then
                ((SUCCESSFUL_MIGRATIONS++))
                log_message "SUCCESS" "Successfully migrated $service"
                touch "$REPORT_DIR/${service}.migrated"
            else
                log_message "ERROR" "Post-migration validation failed for $service"
                ((FAILED_MIGRATIONS++))
                
                # Rollback on validation failure
                if [ "$DRY_RUN" != "true" ]; then
                    log_message "INFO" "Rolling back $service"
                    local dockerfile_path="/opt/sutazaiapp/docker/$service/Dockerfile"
                    if [ ! -f "$dockerfile_path" ]; then
                        dockerfile_path="/opt/sutazaiapp/agents/$service/Dockerfile"
                    fi
                    
                    # Find most recent backup
                    local backup=$(ls -t "${dockerfile_path}.backup-"* 2>/dev/null | head -1)
                    if [ -f "$backup" ]; then
                        cp "$backup" "$dockerfile_path"
                        log_message "INFO" "Rolled back to $backup"
                    fi
                fi
            fi
        else
            ((FAILED_MIGRATIONS++))
            log_message "ERROR" "Migration failed for $service"
        fi
        
        # Progress update
        echo ""
        echo -e "${BLUE}Progress: Processed: $TOTAL_PROCESSED | Success: $SUCCESSFUL_MIGRATIONS | Failed: $FAILED_MIGRATIONS | Skipped: $SKIPPED_MIGRATIONS${NC}"
    done
    
    # Final summary
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  Migration Summary${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "Total Processed: ${YELLOW}$TOTAL_PROCESSED${NC}"
    echo -e "Successful: ${GREEN}$SUCCESSFUL_MIGRATIONS${NC}"
    echo -e "Failed: ${RED}$FAILED_MIGRATIONS${NC}"
    echo -e "Skipped: ${YELLOW}$SKIPPED_MIGRATIONS${NC}"
    echo -e "Success Rate: ${YELLOW}$(echo "scale=2; $SUCCESSFUL_MIGRATIONS * 100 / $TOTAL_PROCESSED" | bc)%${NC}"
    echo ""
    echo -e "Full log: ${YELLOW}$MIGRATION_LOG${NC}"
    
    # Exit with appropriate code
    if [ $FAILED_MIGRATIONS -gt 0 ]; then
        exit 1
    else
        exit 0
    fi
}

# Run main function
main