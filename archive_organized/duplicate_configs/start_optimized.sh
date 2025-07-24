#!/bin/bash

echo "=== Starting Optimized SutazAI System ==="
echo "Only essential services for port 8501"

# Ensure old containers are stopped
docker-compose down || true

# Start only essential services
docker-compose up -d postgres redis

# Wait for database
sleep 5

# Start vector databases if needed by the app
docker-compose up -d chromadb qdrant

# Start Ollama with resource limits
docker-compose up -d ollama

# Ensure Streamlit is running
if ! pgrep -f "streamlit.*8501" > /dev/null; then
    echo "Starting Streamlit app..."
    cd /opt/sutazaiapp
    source venv/bin/activate
    nohup streamlit run intelligent_chat_app_fixed.py \
        --server.address 0.0.0.0 \
        --server.port 8501 \
        --server.headless true \
        > streamlit_fixed.log 2>&1 &
fi

echo "=== System Status ==="
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo ""
echo "Main application: http://192.168.131.128:8501/"
echo "Resource usage:"
docker stats --no-stream