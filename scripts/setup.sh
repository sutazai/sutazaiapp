#!/bin/bash
#
# setup.sh - Comprehensive installation and setup script for SutazAI
#

set -e

# Define colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Function to log messages
log() {
    echo -e "${BLUE}[$(date +"%Y-%m-%d %H:%M:%S")]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if running as root
if [ "$(id -u)" != "0" ]; then
   log_error "This script must be run as root"
   exit 1
fi

# Get the directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set up directories
APP_DIR="${APP_DIR:-/opt/sutazaiapp}"
DATA_DIR="$APP_DIR/data"
LOG_DIR="$APP_DIR/logs"

# Create directories if they don't exist
mkdir -p "$DATA_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$DATA_DIR/cache"
mkdir -p "$DATA_DIR/models"
mkdir -p "$DATA_DIR/documents"
mkdir -p "$DATA_DIR/qdrant"
mkdir -p "$LOG_DIR/maintenance"
mkdir -p "$LOG_DIR/web_ui"
mkdir -p "$LOG_DIR/qdrant"

# Set appropriate permissions
log "Setting appropriate permissions..."
chown -R sutazaiapp:sutazaiapp "$APP_DIR" || {
    log_warning "User sutazaiapp does not exist. Creating user..."
    useradd -m -d /home/sutazaiapp -s /bin/bash sutazaiapp
    chown -R sutazaiapp:sutazaiapp "$APP_DIR"
}

# Install system dependencies
log "Installing system dependencies..."
apt-get update
apt-get install -y \
    python3 python3-pip python3-venv \
    nodejs npm \
    postgresql postgresql-contrib \
    redis-server \
    nginx \
    supervisor \
    git curl wget \
    libpq-dev build-essential \
    ffmpeg libsm6 libxext6 \
    tesseract-ocr libtesseract-dev \
    poppler-utils

# Create Python virtual environment
log "Setting up Python virtual environment..."
if [ ! -d "$APP_DIR/venv" ]; then
    python3 -m venv "$APP_DIR/venv"
fi

# Activate virtual environment and install Python dependencies
source "$APP_DIR/venv/bin/activate"
pip install --upgrade pip
pip install -r "$APP_DIR/requirements.txt"

# Set up configuration
log "Setting up configuration..."
if [ ! -f "$APP_DIR/.env" ] && [ -f "$APP_DIR/.env.example" ]; then
    cp "$APP_DIR/.env.example" "$APP_DIR/.env"
    log_warning "Created .env file from example. Please update with your specific configurations."
fi

# Set up database
log "Setting up database..."
if ! sudo -u postgres psql -lqt | cut -d \| -f 1 | grep -qw sutazaiapp; then
    log "Creating PostgreSQL database..."
    sudo -u postgres createdb sutazaiapp
    sudo -u postgres createuser sutazaiapp
    sudo -u postgres psql -c "ALTER USER sutazaiapp WITH ENCRYPTED PASSWORD 'sutazaiapp';"
    sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE sutazaiapp TO sutazaiapp;"
else
    log_success "Database already exists."
fi

# Set up systemd services
log "Setting up systemd services..."
for SERVICE in "$APP_DIR"/systemd/*.service "$APP_DIR"/systemd/*.timer; do
    if [ -f "$SERVICE" ]; then
        SERVICE_NAME=$(basename "$SERVICE")
        log "Installing service: $SERVICE_NAME"
        cp "$SERVICE" /etc/systemd/system/
    fi
done

# Make scripts executable
log "Making scripts executable..."
find "$APP_DIR/scripts" -name "*.sh" -exec chmod +x {} \;

# Set up backend
log "Setting up backend..."
if [ -d "$APP_DIR/backend" ]; then
    # Create necessary directories
    mkdir -p "$APP_DIR/backend/core/migrations"
    touch "$APP_DIR/backend/core/migrations/__init__.py"
    
    # Run database migrations if needed
    if [ -f "$APP_DIR/backend/core/alembic.ini" ]; then
        cd "$APP_DIR"
        source "$APP_DIR/venv/bin/activate"
        alembic upgrade head || log_warning "Alembic migration failed, please run manually."
    fi
fi

# Set up web UI
log "Setting up web UI..."
if [ -d "$APP_DIR/web_ui" ]; then
    cd "$APP_DIR/web_ui"
    npm install
    npm run build
fi

# Set up Nginx
log "Setting up Nginx..."
if [ -f "$APP_DIR/config/nginx/sutazai.conf" ]; then
    cp "$APP_DIR/config/nginx/sutazai.conf" /etc/nginx/sites-available/
    ln -sf /etc/nginx/sites-available/sutazai.conf /etc/nginx/sites-enabled/
    systemctl restart nginx
fi

# Enable and start services
log "Enabling and starting services..."
systemctl daemon-reload
systemctl enable sutazai.service
systemctl enable sutazai-memory-optimizer.timer
systemctl start sutazai-memory-optimizer.timer
systemctl start sutazai.service

# Display system status
log_success "Installation completed! System status:"
systemctl status sutazai.service --no-pager

# Print final instructions
cat << EOF
${GREEN}=====================================${NC}
${GREEN}   SutazAI Setup Complete!   ${NC}
${GREEN}=====================================${NC}

Your system is now set up and running. Here are some useful commands:

- Start all services:
  ${BLUE}sudo systemctl start sutazai.service${NC}

- Check status:
  ${BLUE}sudo systemctl status sutazai.service${NC}

- View logs:
  ${BLUE}sudo journalctl -u sutazai-api -f${NC}
  ${BLUE}sudo journalctl -u sutazai-webui -f${NC}
  ${BLUE}sudo journalctl -u sutazai-vector-db -f${NC}

- Access the web interface:
  ${BLUE}http://localhost:3000${NC}

- Access the API:
  ${BLUE}http://localhost:8000/health${NC}

Thank you for using SutazAI!
EOF

exit 0 