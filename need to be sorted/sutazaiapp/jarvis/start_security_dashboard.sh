#!/bin/bash
"""
Startup script for SutazAI Security Monitoring Dashboard
"""

# Set script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create virtual environment if it doesn't exist
if [ ! -d "security_dashboard_env" ]; then
    echo "Creating virtual environment for security dashboard..."
    python3 -m venv security_dashboard_env
fi

# Activate virtual environment
source security_dashboard_env/bin/activate

# Install required packages
echo "Installing required packages..."
pip install -q streamlit plotly pandas sqlite3 numpy

# Start the dashboard
echo "Starting SutazAI Security Monitoring Dashboard..."
echo "Dashboard will be available at: http://localhost:8501"
echo "Press Ctrl+C to stop the dashboard"

streamlit run security_monitoring_dashboard.py --server.port 8501 --server.address 0.0.0.0