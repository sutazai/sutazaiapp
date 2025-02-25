#!/bin/bash
# SutazAI GitHub Backup and Synchronization Script

# Configuration
REPO_NAME="SutazAI"
GITHUB_USERNAME="FlorinCristianSuta"
BACKUP_DIR="/opt/sutazai/backups"
LOG_FILE="/var/log/sutazai/github_backup.log"

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Ensure dependencies are installed
ensure_dependencies() {
    log "${YELLOW}Checking GitHub CLI and Git dependencies...${NC}"
    
    # Install GitHub CLI if not present
    if ! command -v gh &> /dev/null; then
        log "Installing GitHub CLI..."
        type -p curl >/dev/null || (sudo apt update && sudo apt install curl -y)
        curl -fsSL https://cli.github.com/packages/githubcli-archive.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
        && sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
        && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
        && sudo apt update \
        && sudo apt install gh -y
    fi
}

# Authenticate with GitHub
github_authenticate() {
    log "${YELLOW}Authenticating with GitHub...${NC}"
    if ! gh auth status &> /dev/null; then
        log "${RED}GitHub authentication required!${NC}"
        gh auth login
    fi
}

# Create backup repository if not exists
create_backup_repo() {
    log "${YELLOW}Checking/Creating backup repository...${NC}"
    
    # Check if repository exists
    if ! gh repo view "$GITHUB_USERNAME/$REPO_NAME-Backup" &> /dev/null; then
        log "Creating backup repository..."
        gh repo create "$REPO_NAME-Backup" --private --description "SutazAI Comprehensive Backup Repository"
    fi
}

# Perform comprehensive backup
perform_backup() {
    log "${GREEN}Starting comprehensive backup...${NC}"
    
    # Create timestamped backup directory
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    CURRENT_BACKUP_DIR="$BACKUP_DIR/$TIMESTAMP"
    mkdir -p "$CURRENT_BACKUP_DIR"
    
    # Copy entire project
    cp -R /home/ai/Desktop/SutazAI/v1/* "$CURRENT_BACKUP_DIR"
    
    # Create tar archive
    tar -czvf "$CURRENT_BACKUP_DIR.tar.gz" "$CURRENT_BACKUP_DIR"
    
    # Push to GitHub
    cd "$CURRENT_BACKUP_DIR" || exit
    git init
    git config user.name "SutazAI Backup Bot"
    git config user.email "backup@sutazai.ai"
    git add .
    git commit -m "Backup from $TIMESTAMP"
    gh repo sync "$GITHUB_USERNAME/$REPO_NAME-Backup"
    
    log "${GREEN}Backup completed successfully!${NC}"
}

# Cleanup old backups
cleanup_old_backups() {
    log "${YELLOW}Cleaning up old backups...${NC}"
    
    # Keep only last 10 backups
    BACKUPS=$(find "$BACKUP_DIR" -maxdepth 1 -type d | sort -r | tail -n +11)
    for backup in $BACKUPS; do
        log "Removing old backup: $backup"
        rm -rf "$backup"
    done
}

# Main execution
main() {
    log "${GREEN}🚀 SutazAI GitHub Backup Process Started${NC}"
    
    # Ensure dependencies and authentication
    ensure_dependencies
    github_authenticate
    create_backup_repo
    
    # Perform backup
    perform_backup
    
    # Cleanup
    cleanup_old_backups
    
    log "${GREEN}✅ GitHub Backup Process Completed${NC}"
}

# Execute main function
main 