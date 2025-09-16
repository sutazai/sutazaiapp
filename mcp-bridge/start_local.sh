#!/bin/bash

# MCP Bridge Local Startup Script
# This runs the MCP Bridge without Docker as a fallback

echo "Starting MCP Bridge locally..."

# Set environment variables
export PYTHONUNBUFFERED=1
export LOG_LEVEL=INFO
export MCP_BRIDGE_PORT=11100

# Database connections (using Docker network hosts)
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_USER=jarvis
export POSTGRES_PASSWORD=sutazai_secure_2024
export POSTGRES_DB=jarvis_ai

export REDIS_HOST=localhost
export REDIS_PORT=6379

export RABBITMQ_HOST=localhost
export RABBITMQ_PORT=5672
export RABBITMQ_USER=sutazai
export RABBITMQ_PASSWORD=sutazai_secure_2024

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install minimal dependencies
echo "Installing minimal dependencies..."
pip install fastapi uvicorn httpx aiohttp redis 2>/dev/null || true

# Create necessary directories
mkdir -p logs data

# Start the server
echo "Starting MCP Bridge on port 11100..."
python services/mcp_bridge_server.py