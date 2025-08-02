#!/bin/bash
# Test environment configuration for SutazAI

# Navigate to the project root directory
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# Check Python version
echo "Checking Python version..."
python_version=$(python3.11 --version 2>/dev/null)
if [ $? -ne 0 ]; then
    echo "ERROR: Python 3.11 is not installed or not in PATH"
    echo "Install Python 3.11 using: sudo apt-get install python3.11 python3.11-venv python3.11-dev"
    exit 1
else
    echo "✓ Python 3.11 detected: $python_version"
fi

# Check virtual environment
echo "Checking virtual environment..."
if [ -d "/opt/venv-sutazaiapp" ]; then
    echo "✓ Virtual environment found at /opt/venv-sutazaiapp"
else
    echo "WARNING: Virtual environment not found at /opt/venv-sutazaiapp"
    echo "Create it using: python3.11 -m venv /opt/venv-sutazaiapp"
fi

# Check if backend/main.py exists
echo "Checking application structure..."
if [ -f "backend/main.py" ]; then
    echo "✓ Backend main.py file found"
else
    echo "ERROR: backend/main.py not found"
    exit 1
fi

# Check if .env file exists
if [ -f ".env" ]; then
    echo "✓ Environment file (.env) found"
    
    # Check HTTPS configuration
    if grep -q "ENFORCE_HTTPS=true" ".env"; then
        echo "  - HTTPS redirection is ENABLED"
    else
        echo "  - HTTPS redirection is DISABLED"
    fi
    
    if grep -q "BEHIND_PROXY=true" ".env"; then
        echo "  - Application is configured to run behind a proxy"
    else
        echo "  - Application is configured to run directly"
    fi
else
    echo "WARNING: .env file not found"
fi

# Check if Nginx is configured
if [ -f "/etc/nginx/sites-enabled/sutazaiapp" ]; then
    echo "✓ Nginx configuration found"
    if grep -q "ssl_certificate" "/etc/nginx/sites-enabled/sutazaiapp"; then
        echo "  - SSL is configured in Nginx"
    else
        echo "  - SSL is NOT configured in Nginx"
    fi
else
    echo "NOTE: No Nginx configuration found for SutazAI"
    echo "  - For production HTTPS, run: sudo scripts/setup_https.sh"
fi

# Check SSL certificate directories
echo "Checking SSL certificates..."
if [ -d "ssl" ] && [ -f "ssl/cert.pem" ] && [ -f "ssl/key.pem" ]; then
    echo "✓ Development SSL certificates found in project directory"
elif [ -d "/etc/nginx/ssl/sutazaiapp" ] && [ -f "/etc/nginx/ssl/sutazaiapp/fullchain.pem" ]; then
    echo "✓ Production SSL certificates found in Nginx directory"
elif [ -d "/etc/letsencrypt/live" ]; then
    echo "✓ Let's Encrypt certificates may be available"
else
    echo "WARNING: No SSL certificates found"
fi

# Check services status
echo "Checking services status..."
backend_pid=$(cat ".backend.pid" 2>/dev/null)
if [ -n "$backend_pid" ] && ps -p $backend_pid > /dev/null; then
    echo "✓ Backend service is running (PID: $backend_pid)"
else
    echo "NOTE: Backend service is not running"
fi

qdrant_running=$(docker ps | grep qdrant | wc -l)
if [ "$qdrant_running" -gt 0 ]; then
    echo "✓ Qdrant database is running"
else
    echo "NOTE: Qdrant database is not running"
fi

# Print summary
echo ""
echo "=== Environment Summary ==="
echo "• Application directory: $PROJECT_ROOT"
echo "• Python version: $python_version"
echo "• Configuration: $([[ -f ".env" ]] && echo "Found" || echo "Missing")"
echo "• HTTPS setup: $([[ -f "/etc/nginx/sites-enabled/sutazaiapp" ]] && echo "Nginx Proxy" || echo "Direct/Development")"
echo "• Backend status: $([[ -n "$backend_pid" ]] && ps -p $backend_pid > /dev/null && echo "Running" || echo "Stopped")"
echo ""
echo "For a complete production setup with HTTPS:"
echo "1. Run: sudo scripts/setup_https.sh"
echo "2. Choose Let's Encrypt for production certificates"
echo "3. Restart services: sudo scripts/stop_all.sh && sudo scripts/start_all.sh"
echo "" 