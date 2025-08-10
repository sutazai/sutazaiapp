#!/bin/bash
# Purpose: Setup and install the garbage collection system
# Usage: ./setup-garbage-collection.sh
# Requires: Root privileges, Python 3.8+

set -e


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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="/opt/sutazaiapp"

echo "Setting up Garbage Collection System..."

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root (use sudo)"
    exit 1
fi

# Install required Python packages
echo "Installing Python dependencies..."
pip3 install psutil schedule

# Make script executable
chmod +x "$SCRIPT_DIR/garbage-collection-system.py"

# Create necessary directories
echo "Creating directories..."
mkdir -p "$BASE_DIR/archive/garbage-collection"
mkdir -p "$BASE_DIR/data"
mkdir -p "$BASE_DIR/dashboard"

# Copy systemd service file
echo "Installing systemd service..."
cp "$SCRIPT_DIR/garbage-collection.service" /etc/systemd/system/sutazai-garbage-collection.service

# Reload systemd
systemctl daemon-reload

# Enable and start the service
echo "Enabling and starting garbage collection service..."
systemctl enable sutazai-garbage-collection.service
systemctl start sutazai-garbage-collection.service

# Create cron job for dashboard generation
echo "Setting up dashboard generation..."
(crontab -l 2>/dev/null; echo "*/5 * * * * /usr/bin/python3 $SCRIPT_DIR/garbage-collection-system.py --dashboard") | crontab -

# Run initial collection in dry-run mode
echo "Running initial dry-run collection..."
python3 "$SCRIPT_DIR/garbage-collection-system.py" --dry-run

# Generate initial dashboard
echo "Generating initial dashboard..."
python3 "$SCRIPT_DIR/garbage-collection-system.py" --dashboard

echo "Garbage Collection System setup complete!"
echo ""
echo "Service status: systemctl status sutazai-garbage-collection"
echo "View logs: journalctl -u sutazai-garbage-collection -f"
echo "Dashboard: $BASE_DIR/dashboard/garbage-collection.html"
echo "Configuration: $BASE_DIR/config/garbage-collection.json"