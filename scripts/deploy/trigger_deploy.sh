#!/bin/bash
# SutazAI Deployment Trigger Script
# This script provides a secure way to trigger deployments,
# typically used in CI/CD pipelines or by authorized users

# Navigate to the project root directory
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Log file
TRIGGER_LOG="${PROJECT_ROOT}/logs/trigger_deploy.log"
mkdir -p "$(dirname "$TRIGGER_LOG")"

# Config file for deployment settings
CONFIG_FILE="${PROJECT_ROOT}/.deploy_config"
OTP_OVERRIDE_FILE="${PROJECT_ROOT}/.otp_override"

# Logging function
log() {
    local message="$1"
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo -e "$message"
    echo "[$timestamp] $message" >> "$TRIGGER_LOG"
}

log "${BLUE}Starting SutazAI deployment trigger...${NC}"

# Function to check if a command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Generate a simple OTP if python is available
generate_otp() {
    if command_exists python; then
        python -c "import random; print(''.join([str(random.randint(0, 9)) for _ in range(6)]))"
    else
        # Fallback to a simpler method if python is not available
        echo $((RANDOM % 900000 + 100000))
    fi
}

# Check for deployment configuration
if [ ! -f "$CONFIG_FILE" ]; then
    log "${YELLOW}Deployment configuration not found. Creating a default one...${NC}"
    
    # Create a basic configuration file
    cat > "$CONFIG_FILE" << EOF
# SutazAI Deployment Configuration
DEPLOY_ENV="production"
DEPLOY_USER="$(whoami)"
REQUIRE_OTP=true
NOTIFY_EMAIL=""
AUTO_BACKUP=true
RESTART_SERVICES=true
EOF
    
    log "${GREEN}Default deployment configuration created.${NC}"
    log "${YELLOW}Please review and update the configuration file: $CONFIG_FILE${NC}"
fi

# Source the configuration file
source "$CONFIG_FILE"

# Parse command line arguments
FORCE_DEPLOY=false
SKIP_CHECKS=false
QUIET_MODE=false
TARGET_ENV="$DEPLOY_ENV"

while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE_DEPLOY=true
            shift
            ;;
        --skip-checks)
            SKIP_CHECKS=true
            shift
            ;;
        --quiet)
            QUIET_MODE=true
            shift
            ;;
        --env=*)
            TARGET_ENV="${1#*=}"
            shift
            ;;
        --env)
            TARGET_ENV="$2"
            shift 2
            ;;
        *)
            log "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Log deployment details
log "Deployment target environment: $TARGET_ENV"
log "Triggered by user: $(whoami)"
log "Timestamp: $(date)"

# Verify if the user has permission to deploy
CURRENT_USER=$(whoami)

if [ "$CURRENT_USER" != "$DEPLOY_USER" ] && [ "$FORCE_DEPLOY" != "true" ]; then
    log "${RED}Error: You ($CURRENT_USER) don't have permission to trigger a deployment.${NC}"
    log "Deployment is only allowed for $DEPLOY_USER or with --force flag."
    exit 1
fi

# Check if we need an OTP for verification
if [ "$REQUIRE_OTP" = "true" ] && [ "$FORCE_DEPLOY" != "true" ]; then
    # Check if there's an OTP override file (for automated deployments)
    if [ -f "$OTP_OVERRIDE_FILE" ]; then
        log "${YELLOW}Using OTP override file.${NC}"
        source "$OTP_OVERRIDE_FILE"
    else
        # Generate a new OTP
        OTP=$(generate_otp)
        
        # Save the OTP (in a real system, this would be sent via a secure channel)
        echo "Generated OTP: $OTP"
        echo "OTP_VALUE=$OTP" > "$OTP_OVERRIDE_FILE"
        chmod 600 "$OTP_OVERRIDE_FILE"
        
        # Ask for OTP verification
        read -p "Enter the OTP to confirm deployment: " USER_OTP
        
        if [ "$USER_OTP" != "$OTP" ]; then
            log "${RED}Error: Invalid OTP. Deployment aborted.${NC}"
            rm -f "$OTP_OVERRIDE_FILE"
            exit 1
        fi
    fi
    
    # Clean up OTP file
    rm -f "$OTP_OVERRIDE_FILE"
fi

# Perform pre-deployment checks
if [ "$SKIP_CHECKS" != "true" ]; then
    log "${BLUE}Performing pre-deployment checks...${NC}"
    
    # Check if the deployment script exists
    if [ ! -f "${PROJECT_ROOT}/scripts/deploy.sh" ]; then
        log "${RED}Error: Deployment script not found at ${PROJECT_ROOT}/scripts/deploy.sh${NC}"
        exit 1
    fi
    
    # Check if there are uncommitted changes in git (if applicable)
    if command_exists git && [ -d "${PROJECT_ROOT}/.git" ]; then
        if [ -n "$(git status --porcelain)" ]; then
            log "${YELLOW}Warning: There are uncommitted changes in the repository.${NC}"
            read -p "Continue with deployment anyway? (y/n): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                log "Deployment aborted by user."
                exit 0
            fi
        fi
    fi
    
    # Create a backup if auto-backup is enabled
    if [ "$AUTO_BACKUP" = "true" ]; then
        log "${BLUE}Creating backup before deployment...${NC}"
        if [ -x "${PROJECT_ROOT}/scripts/create_archive.sh" ]; then
            if [ "$QUIET_MODE" = "true" ]; then
                # Create a non-interactive backup
                BACKUP_FILE="sutazai_pre_deploy_$(date +"%Y%m%d_%H%M%S").tar.gz"
                mkdir -p "${PROJECT_ROOT}/backups"
                tar czf "${PROJECT_ROOT}/backups/$BACKUP_FILE" --exclude="venv" --exclude="__pycache__" --exclude="*.pyc" --exclude="node_modules" --exclude=".git" -C "$PROJECT_ROOT" .
                log "${GREEN}Backup created: ${PROJECT_ROOT}/backups/$BACKUP_FILE${NC}"
            else
                bash "${PROJECT_ROOT}/scripts/create_archive.sh"
            fi
        else
            log "${YELLOW}Warning: Backup script not found or not executable.${NC}"
        fi
    fi
fi

# Trigger the actual deployment
log "${BLUE}Triggering deployment for environment: $TARGET_ENV${NC}"

# Add deployment marker file
echo "DEPLOY_START=$(date +%s)" > "${PROJECT_ROOT}/.deploy_marker"
echo "DEPLOY_ENV=$TARGET_ENV" >> "${PROJECT_ROOT}/.deploy_marker"
echo "DEPLOY_USER=$(whoami)" >> "${PROJECT_ROOT}/.deploy_marker"

# Run the deployment script
if [ -x "${PROJECT_ROOT}/scripts/deploy.sh" ]; then
    log "Executing deployment script..."
    
    # Pass environment to the deployment script
    bash "${PROJECT_ROOT}/scripts/deploy.sh" --env="$TARGET_ENV"
    DEPLOY_STATUS=$?
    
    if [ $DEPLOY_STATUS -eq 0 ]; then
        log "${GREEN}Deployment completed successfully!${NC}"
    else
        log "${RED}Deployment failed with status code: $DEPLOY_STATUS${NC}"
    fi
else
    log "${RED}Error: Deployment script is not executable.${NC}"
    log "Please run: chmod +x ${PROJECT_ROOT}/scripts/deploy.sh"
    exit 1
fi

# Update deployment marker file
echo "DEPLOY_END=$(date +%s)" >> "${PROJECT_ROOT}/.deploy_marker"
echo "DEPLOY_STATUS=$DEPLOY_STATUS" >> "${PROJECT_ROOT}/.deploy_marker"

# Restart services if needed and successful
if [ "$RESTART_SERVICES" = "true" ] && [ $DEPLOY_STATUS -eq 0 ]; then
    log "${BLUE}Restarting services...${NC}"
    
    # Stop services
    if [ -x "${PROJECT_ROOT}/scripts/stop_backend.sh" ]; then
        bash "${PROJECT_ROOT}/scripts/stop_backend.sh"
    fi
    
    if [ -x "${PROJECT_ROOT}/scripts/stop_superagi.sh" ]; then
        bash "${PROJECT_ROOT}/scripts/stop_superagi.sh"
    fi
    
    # Start services
    if [ -x "${PROJECT_ROOT}/scripts/start_backend.sh" ]; then
        bash "${PROJECT_ROOT}/scripts/start_backend.sh"
    fi
    
    if [ -x "${PROJECT_ROOT}/scripts/start_superagi.sh" ]; then
        bash "${PROJECT_ROOT}/scripts/start_superagi.sh" --service
    fi
    
    log "${GREEN}Services restarted.${NC}"
fi

# Send notification if email is configured
if [ -n "$NOTIFY_EMAIL" ]; then
    SUBJECT="SutazAI Deployment [$TARGET_ENV] - "
    if [ $DEPLOY_STATUS -eq 0 ]; then
        SUBJECT="${SUBJECT}Success"
    else
        SUBJECT="${SUBJECT}Failed"
    fi
    
    if command_exists mail; then
        log "Sending deployment notification to $NOTIFY_EMAIL"
        echo "SutazAI deployment to $TARGET_ENV completed with status: $DEPLOY_STATUS" | mail -s "$SUBJECT" "$NOTIFY_EMAIL"
    else
        log "${YELLOW}Warning: 'mail' command not found. Notification email not sent.${NC}"
    fi
fi

log "${GREEN}Deployment trigger process completed!${NC}"
exit $DEPLOY_STATUS
