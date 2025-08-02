#!/bin/bash
#
# start_qdrant.sh - Script to start Qdrant vector database
#

set -e

# Source environment variables
if [ -f /opt/sutazaiapp/.env ]; then
    source <(grep -v '^#' /opt/sutazaiapp/.env | sed -E 's/(.*)=(.*)/export \1="\2"/')
fi

# Configuration
QDRANT_DIR="${QDRANT_DIR:-/opt/sutazaiapp/data/qdrant}"
QDRANT_PORT="${QDRANT_PORT:-6333}"
QDRANT_LOG_DIR="${LOGS_DIR:-/opt/sutazaiapp/logs}/qdrant"
QDRANT_VERSION="${QDRANT_VERSION:-v1.13.4}"
QDRANT_CONFIG_DIR="${QDRANT_CONFIG_DIR:-/opt/sutazaiapp/config}"
QDRANT_CONFIG="$QDRANT_CONFIG_DIR/qdrant.yaml"
QDRANT_PID_FILE="/opt/sutazaiapp/run/qdrant.pid"

# Create directories if they don't exist
mkdir -p "$QDRANT_DIR" "$QDRANT_CONFIG_DIR" "$QDRANT_LOG_DIR" "/opt/sutazaiapp/run"

# Check if Qdrant is already running
if [ -f "$QDRANT_PID_FILE" ]; then
    PID=$(cat "$QDRANT_PID_FILE")
    if ps -p "$PID" > /dev/null; then
        echo "Qdrant is already running with PID: $PID"
        exit 0
    else
        echo "Removing stale PID file"
        rm "$QDRANT_PID_FILE"
    fi
fi

# Check if port 6333 is already in use
if command -v netstat &> /dev/null; then
    PORT_IN_USE=$(netstat -tuln | grep ":6333" || true)
    if [ -n "$PORT_IN_USE" ]; then
        echo "Error: Port 6333 is already in use. Cannot start Qdrant."
        echo "Use 'lsof -i :6333' or 'netstat -tulnp | grep :6333' to find the process."
        exit 1
    fi
fi

# Check if Qdrant binary exists, download if not
QDRANT_BIN="$QDRANT_DIR/qdrant"
if [ ! -f "$QDRANT_BIN" ]; then
    echo "Downloading Qdrant $QDRANT_VERSION..."
    
    # Determine system architecture
    ARCH=$(uname -m)
    if [ "$ARCH" = "x86_64" ]; then
        QDRANT_ARCH="x86_64"
    elif [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
        QDRANT_ARCH="aarch64"
    else
        echo "Unsupported architecture: $ARCH"
        exit 1
    fi
    
    # Download URL for Qdrant
    DOWNLOAD_URL="https://github.com/qdrant/qdrant/releases/download/$QDRANT_VERSION/qdrant-$QDRANT_VERSION-linux-$QDRANT_ARCH.tar.gz"
    
    # Create a temporary directory
    TMP_DIR=$(mktemp -d)
    
    # Download and extract
    if ! curl -L "$DOWNLOAD_URL" -o "$TMP_DIR/qdrant.tar.gz"; then
        echo "Failed to download Qdrant"
        rm -rf "$TMP_DIR"
        exit 1
    fi
    
    if ! tar -xzf "$TMP_DIR/qdrant.tar.gz" -C "$TMP_DIR"; then
        echo "Failed to extract Qdrant"
        rm -rf "$TMP_DIR"
        exit 1
    fi
    
    # Move the binary to the destination
    if [ -f "$TMP_DIR/qdrant" ]; then
        mv "$TMP_DIR/qdrant" "$QDRANT_BIN"
        chmod +x "$QDRANT_BIN"
    else
        echo "Qdrant binary not found in the extracted package"
        rm -rf "$TMP_DIR"
        exit 1
    fi
    
    # Clean up
    rm -rf "$TMP_DIR"
    
    echo "Qdrant binary downloaded and installed."
fi

# Create Qdrant configuration if it doesn't exist
if [ ! -f "$QDRANT_CONFIG" ]; then
    echo "Creating Qdrant configuration..."
    cat > "$QDRANT_CONFIG" << EOF
storage:
  storage_path: ${QDRANT_DIR}/storage
  
service:
  host: 0.0.0.0
  http_port: ${QDRANT_PORT}
  grpc_port: 6334
  
telemetry:
  disabled: false
EOF
    echo "Created default Qdrant configuration at $QDRANT_CONFIG"
fi

# Start Qdrant
echo "Starting Qdrant vector database on port $QDRANT_PORT..."
nohup "$QDRANT_BIN" --config-path "$QDRANT_CONFIG" > >(tee -a "$QDRANT_LOG_DIR/qdrant.log") 2>&1 &

# Save PID
echo $! > "$QDRANT_PID_FILE"

# Wait a moment to check if process is still running (to catch immediate failures)
sleep 2
if ! ps -p $! > /dev/null; then
    echo "Error: Qdrant failed to start. Check the logs at $QDRANT_LOG_DIR/qdrant.log"
    rm "$QDRANT_PID_FILE" 2>/dev/null
    exit 1
fi

# Check if the port is open
echo "Waiting for Qdrant to start (up to 10 seconds)..."
for i in {1..10}; do
    if command -v curl &> /dev/null && curl -s http://localhost:$QDRANT_PORT/health > /dev/null; then
        echo "Qdrant is now running with PID $(cat "$QDRANT_PID_FILE")"
        echo "Access web UI at http://localhost:$QDRANT_PORT/dashboard"
        exit 0
    fi
    sleep 1
done

echo "Warning: Qdrant started but health check timed out. It may take longer to initialize."
echo "Check logs at $QDRANT_LOG_DIR/qdrant.log"
exit 0 