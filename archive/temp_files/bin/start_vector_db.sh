#!/bin/bash
# SutazAI Vector Database (Qdrant) Startup Script

APP_ROOT="/opt/sutazaiapp"
STORAGE_DIR="$APP_ROOT/storage/qdrant"
LOGS_DIR="$APP_ROOT/logs"
PIDS_DIR="$APP_ROOT/pids"
PID_FILE="$PIDS_DIR/vector-db.pid"
LOG_FILE="$LOGS_DIR/vector-db.log"
QDRANT_PORT=6333

# Ensure directories exist
mkdir -p "$LOGS_DIR"
mkdir -p "$PIDS_DIR"
mkdir -p "$STORAGE_DIR"

# Kill any existing Qdrant process
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null; then
        echo "Stopping existing Qdrant process (PID: $OLD_PID)"
        # Use sudo to ensure permission
        sudo kill "$OLD_PID"
        sleep 2
        # Force kill if still running
        if ps -p "$OLD_PID" > /dev/null; then
            echo "Force killing Qdrant process"
            # Use sudo to ensure permission
            sudo kill -9 "$OLD_PID"
        fi
    fi
fi

# Check if Qdrant is installed via Docker
if command -v docker &> /dev/null; then
    # Check if Qdrant container is already running
    RUNNING_CONTAINER=$(docker ps --filter name=qdrant -q)
    if [ -n "$RUNNING_CONTAINER" ]; then
        echo "Qdrant is already running in Docker with container ID: $RUNNING_CONTAINER"
        # Get the container PID
        CONTAINER_PID=$(docker inspect -f '{{.State.Pid}}' "$RUNNING_CONTAINER")
        echo "$CONTAINER_PID" > "$PID_FILE"
        echo "Updated PID file with Docker container PID: $CONTAINER_PID"
        exit 0
    fi
    
    # Start Qdrant with Docker
    echo "Starting Qdrant with Docker..."
    docker run -d --name qdrant \
        -p 6333:6333 \
        -p 6334:6334 \
        -v "$STORAGE_DIR:/qdrant/storage" \
        qdrant/qdrant >> "$LOG_FILE" 2>&1
    
    # Get container ID
    CONTAINER_ID=$(docker ps --filter name=qdrant -q)
    if [ -n "$CONTAINER_ID" ]; then
        # Get the container PID
        CONTAINER_PID=$(docker inspect -f '{{.State.Pid}}' "$CONTAINER_ID")
        echo "$CONTAINER_PID" > "$PID_FILE"
        echo "Qdrant started in Docker with container ID: $CONTAINER_ID, PID: $CONTAINER_PID"
    else
        echo "Failed to start Qdrant with Docker."
        # Don't exit, proceed to Python fallback
        echo "Proceeding with Python fallback..."
    fi
# else # Don't use else, just fall through if docker fails
fi # End of initial docker check

# Try to start with Python qdrant-client if Docker failed or wasn't found
# Check if PID file exists from Docker attempt or previous run
if [ ! -f "$PID_FILE" ] || ! QDRANT_PID=$(cat "$PID_FILE") || ! ps -p "$QDRANT_PID" > /dev/null; then
    echo "Docker Qdrant not running or not found. Using Python qdrant-client fallback..."
    
    # Check if qdrant-client is installed
    if ! python3 -c "import qdrant_client" &> /dev/null; then
        echo "Installing qdrant-client..."
        python3 -m pip install qdrant-client >> "$LOG_FILE" 2>&1
    fi
    
    # Create a simple script to run Qdrant server
    QDRANT_SCRIPT="${APP_ROOT}/tmp/run_qdrant.py"
    mkdir -p "$(dirname "$QDRANT_SCRIPT")"
    cat > "$QDRANT_SCRIPT" << EOF
import os
import time
from qdrant_client import QdrantClient
from qdrant_client.http import models
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='${LOG_FILE}',
    filemode='a'
)
logger = logging.getLogger("QdrantServer")

# Create storage directory
os.makedirs("${STORAGE_DIR}", exist_ok=True)

# Create Qdrant client
client = QdrantClient(path="${STORAGE_DIR}")

# Check if initialized
initialized_file = "${STORAGE_DIR}/.qdrant-initialized"
if not os.path.exists(initialized_file):
    # Create a test collection to verify everything works
    logger.info("Initializing Qdrant...")
    try:
        client.create_collection(
            collection_name="sutazai_vectors",
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
        )
        # Mark as initialized
        with open(initialized_file, "w") as f:
            f.write("")
        logger.info("Qdrant initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant: {str(e)}")

# Keep the process running
logger.info("Qdrant server started")
try:
    while True:
        time.sleep(10)
        # Periodically verify the service is working
        collections = client.get_collections()
        logger.debug(f"Collections: {collections}")
except KeyboardInterrupt:
    logger.info("Qdrant server stopping")
except Exception as e:
    logger.error(f"Error in Qdrant server: {str(e)}")
EOF
    
    # Run the Qdrant server
    echo "Starting Qdrant with Python client..."
    python3 "$QDRANT_SCRIPT" >> "$LOG_FILE" 2>&1 &
    QDRANT_PID=$!
    
    # Save PID
    echo "$QDRANT_PID" > "$PID_FILE"
    echo "Qdrant started with PID: $QDRANT_PID"
fi # Closing fi for the python fallback logic

# Wait a moment to ensure it's running
sleep 5

# Verify Qdrant is accessible
if command -v curl &> /dev/null; then
    echo "Verifying Qdrant is accessible..."
    if curl -s "http://localhost:$QDRANT_PORT/collections" > /dev/null; then
        echo "Qdrant is running and accessible on port $QDRANT_PORT"
    else
        echo "Warning: Qdrant may not be accessible. Check logs at $LOG_FILE"
    fi
fi

echo "Vector database startup completed."
echo "Logs are being written to: $LOG_FILE" 