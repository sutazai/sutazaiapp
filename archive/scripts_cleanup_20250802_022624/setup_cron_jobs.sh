#!/bin/bash
# title        :setup_cron_jobs.sh
# description  :This script sets up cron jobs for automated system maintenance
# author       :SutazAI Team
# version      :1.0
# usage        :sudo bash scripts/setup_cron_jobs.sh
# notes        :Requires bash 4.0+ and standard Linux utilities

# Navigate to the project root directory
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Check if script is run with sudo
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}[ERROR]${NC} Please run this script with sudo or as root"
    exit 1
fi

echo -e "${BLUE}[INFO]${NC} Setting up cron jobs for automated maintenance"

# Make sure scripts are executable
chmod +x "$PROJECT_ROOT/scripts/auto_maintenance.sh"
chmod +x "$PROJECT_ROOT/scripts/cleanup_redundancies.sh"
chmod +x "$PROJECT_ROOT/scripts/health_check.sh"

# Create cron job entries
CRON_AUTO_MAINTENANCE="0 */4 * * * $PROJECT_ROOT/scripts/auto_maintenance.sh --cron >> $PROJECT_ROOT/logs/cron.log 2>&1"
CRON_WEEKLY_CLEANUP="0 2 * * 0 $PROJECT_ROOT/scripts/cleanup_redundancies.sh >> $PROJECT_ROOT/logs/cron.log 2>&1"
CRON_HEALTH_CHECK="*/15 * * * * $PROJECT_ROOT/scripts/health_check.sh --cron >> $PROJECT_ROOT/logs/cron.log 2>&1"

# Set up cron jobs without duplicating them
setup_cron_job() {
    local cron_entry="$1"
    local job_name="$2"
    
    # Check if job already exists
    if crontab -l 2>/dev/null | grep -Fq "$cron_entry"; then
        echo -e "${YELLOW}[WARNING]${NC} $job_name cron job already exists. Skipping."
    else
        # Add new cron job
        (crontab -l 2>/dev/null; echo "$cron_entry") | crontab -
        echo -e "${GREEN}[SUCCESS]${NC} Added $job_name cron job"
    fi
}

# Set up each cron job
setup_cron_job "$CRON_AUTO_MAINTENANCE" "Auto-maintenance (every 4 hours)"
setup_cron_job "$CRON_WEEKLY_CLEANUP" "Weekly cleanup (Sunday at 2 AM)"
setup_cron_job "$CRON_HEALTH_CHECK" "Health check (every 15 minutes)"

# Create log rotation configuration
LOGROTATE_CONFIG="/etc/logrotate.d/sutazai"

if [ ! -f "$LOGROTATE_CONFIG" ]; then
    echo -e "${BLUE}[INFO]${NC} Setting up log rotation for maintenance logs"
    
    cat > "$LOGROTATE_CONFIG" << EOF
$PROJECT_ROOT/logs/*.log {
    weekly
    rotate 4
    compress
    delaycompress
    missingok
    notifempty
    create 0640 root root
}
EOF
    
    echo -e "${GREEN}[SUCCESS]${NC} Created log rotation configuration"
else
    echo -e "${YELLOW}[WARNING]${NC} Log rotation configuration already exists. Skipping."
fi

# Output database connection pool settings for SQLAlchemy
echo -e "${BLUE}[INFO]${NC} Optimizing database connection settings"

# Provide recommendations for SQLite databases
if grep -q "^DB_TYPE=sqlite" "$PROJECT_ROOT/.env"; then
    echo -e "${YELLOW}[INFO]${NC} Using SQLite database"
    echo -e "${BLUE}[INFO]${NC} Recommended settings for SQLite:"
    echo -e "  - Set pool_pre_ping=True to detect stale connections"
    echo -e "  - Use pool_recycle=3600 (1 hour) to refresh connections"
    echo -e "${BLUE}[INFO]${NC} These settings are already configured in the backend."
else
    # Recommendations for PostgreSQL
    echo -e "${YELLOW}[INFO]${NC} Using PostgreSQL database"
    echo -e "${BLUE}[INFO]${NC} Recommended settings for PostgreSQL:"
    echo -e "  - pool_size=10 (default)"
    echo -e "  - max_overflow=20 (default)"
    echo -e "  - pool_timeout=30 (default)"
    echo -e "  - pool_pre_ping=True (already configured)"
    echo -e "  - pool_recycle=3600 (already configured)"
fi

echo -e "${GREEN}[SUCCESS]${NC} Cron jobs setup completed"
echo -e "${BLUE}[INFO]${NC} The following jobs have been scheduled:"
echo -e "  - Auto-maintenance: Every 4 hours"
echo -e "  - Weekly cleanup: Sunday at 2 AM"
echo -e "  - Health check: Every 15 minutes"
echo -e "${BLUE}[INFO]${NC} Logs will be written to $PROJECT_ROOT/logs/cron.log"

exit 0 