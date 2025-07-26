#!/bin/bash
# Start the SutazAI Backend

# Navigate to the project root directory
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# Check if the virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Using system venv at /opt/venv-sutazaiapp"
fi

# Source the virtual environment
if [ -d "/opt/venv-sutazaiapp" ]; then
    source /opt/venv-sutazaiapp/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: No Python virtual environment found"
    exit 1
fi

# Fix PyYAML dependency to ensure compatibility with all packages
echo "Ensuring PyYAML compatibility..."
pip install --no-cache-dir --force-reinstall -I PyYAML==6.0 > /dev/null 2>&1

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found. Creating from example..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "Created .env file from example. Please edit it with your configuration."
    else
        echo "Error: .env.example file not found. Please create a .env file manually."
        exit 1
    fi
fi

# Create logs directory if it doesn't exist
LOGS_DIR="logs"
if [ ! -d "$LOGS_DIR" ]; then
    mkdir -p "$LOGS_DIR"
    echo "Created logs directory."
fi
chmod -R 777 "$LOGS_DIR" 2>/dev/null || echo "Warning: Could not set permissions on logs directory"

# Check if backend/main.py exists
if [ ! -f "backend/main.py" ]; then
    echo "Error: backend/main.py not found. Cannot start the backend server."
    exit 1
fi

# Kill existing backend process if running
if [ -f ".backend.pid" ]; then
    OLD_PID=$(cat .backend.pid)
    if ps -p "$OLD_PID" > /dev/null; then
        echo "Stopping existing backend process (PID: $OLD_PID)..."
        kill "$OLD_PID" 2>/dev/null || true
        sleep 2
        # Force kill if still running
        kill -9 "$OLD_PID" 2>/dev/null || true
        echo "Previous backend process stopped."
    else
        echo "Stale backend PID file found. Cleaning up..."
    fi
    rm -f ".backend.pid"
fi

# Check if port 8000 is already in use by a different process
if netstat -tuln | grep -q ":8000 "; then
    PID=$(lsof -i :8000 -t 2>/dev/null || echo "unknown")
    if [ "$PID" != "unknown" ]; then
        # Check if it's our own process
        OUR_PROCESS=$(ps -p "$PID" -o cmd= 2>/dev/null | grep -q "uvicorn.*backend" && echo "yes" || echo "no")
        if [ "$OUR_PROCESS" = "yes" ]; then
            echo "Backend is already running with PID: $PID"
            # Update PID file
            echo $PID > "$LOGS_DIR/backend.pid"
            echo $PID > ".backend.pid"
            echo "Backend PID files updated. Check health with: curl http://localhost:8000/api/health"
            exit 0
        else
            echo "Error: Port 8000 is already in use by another process (PID: $PID)."
            echo "Please stop that process first and try again."
            exit 1
        fi
    else
        echo "Error: Port 8000 is already in use, but cannot determine the process."
        echo "Please free up port 8000 and try again."
        exit 1
    fi
fi

# Check if nginx is configured for HTTPS (proxy setup)
NGINX_CONFIG="/etc/nginx/sites-enabled/sutazaiapp"
if [ -f "$NGINX_CONFIG" ]; then
    echo "Nginx HTTPS proxy detected. Starting in HTTP mode for proxy..."
    # Start the backend server in HTTP mode for Nginx proxy
    echo "Starting SutazAI Backend (HTTP mode for Nginx proxy)..."
    python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 > "$LOGS_DIR/backend.log" 2>&1 &
else
    # No Nginx proxy, check for SSL certificates
    SSL_DIR="$PROJECT_ROOT/ssl"
    if [ ! -d "$SSL_DIR" ]; then
        mkdir -p "$SSL_DIR"
        echo "Created SSL directory."
        
        echo "Generating self-signed SSL certificates for development..."
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
          -keyout "$SSL_DIR/key.pem" \
          -out "$SSL_DIR/cert.pem" \
          -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost" 2>/dev/null
        
        chmod 600 "$SSL_DIR/key.pem" "$SSL_DIR/cert.pem"
        echo "Self-signed certificates generated. For production, use setup_https.sh script."
    fi

    # Dispose any existing database connections before starting new backend process
    echo "Disposing database engine connections before starting backend..."
    python3 -c "
import sys
sys.path.append('.')
try:
    from backend.core.database import dispose_engine
    dispose_engine()
    print('Successfully disposed database engine connections')
except Exception as e:
    print(f'Warning: Could not dispose database engine: {e}')
"

    # For development, we'll run in HTTP mode for compatibility
    # Users can run setup_https.sh for proper HTTPS with Nginx
    echo "Starting SutazAI Backend (HTTP mode)..."
    python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 > "$LOGS_DIR/backend.log" 2>&1 &
    echo "NOTE: For secure HTTPS setup, run: sudo scripts/setup_https.sh"
fi

BACKEND_PID=$!
# Store PID in both locations for backward compatibility
echo $BACKEND_PID > "$LOGS_DIR/backend.pid"
# Use sudo if needed to ensure we can write the PID file
if [ -w "." ]; then
    echo $BACKEND_PID > ".backend.pid"
else
    echo "Using sudo to write PID file (may prompt for password)..."
    sudo sh -c "echo $BACKEND_PID > \"$PROJECT_ROOT/.backend.pid\""
fi
echo "Backend started with PID: $BACKEND_PID"
echo "Logs are being written to: $LOGS_DIR/backend.log"

# Verify backend has started correctly
sleep 3
if ! ps -p $BACKEND_PID > /dev/null; then
    echo "Error: Backend process failed to start or terminated immediately."
    echo "Check the logs at $LOGS_DIR/backend.log for details."
    if [ -f "$LOGS_DIR/backend.log" ]; then
        echo "Last 10 lines of log:"
        tail -n 10 "$LOGS_DIR/backend.log"
    fi
    exit 1
fi

# Check backend health
echo "Checking backend health endpoint..."
MAX_TRIES=30
for ((i=1; i<=$MAX_TRIES; i++)); do
    if curl -s "http://localhost:8000/api/health" > /dev/null; then
        echo "Backend health check successful! API is operational."
        break
    else
        if [ $i -eq $MAX_TRIES ]; then
            echo "Warning: Could not verify backend health endpoint after $MAX_TRIES attempts."
            echo "The server may be running but the health endpoint is not responding."
            echo "Check the logs at $LOGS_DIR/backend.log for details."
        else
            echo -n "."
            sleep 1
        fi
    fi
done

echo "Backend successfully started!"
echo "To stop the server, run: scripts/stop_backend.sh"
