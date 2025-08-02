#!/bin/bash
# Agent startup script with health check support

# Install dependencies if not already installed
if ! python -c "import requests" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install --no-cache-dir requests asyncio python-dotenv pydantic httpx aiohttp PyYAML
fi

# Copy agent_base.py if it doesn't exist
if [ ! -f "/app/agent_base.py" ]; then
    cp /app/shared/agent_base.py /app/agent_base.py 2>/dev/null || true
fi

# Copy agent_with_health.py if it doesn't exist
if [ ! -f "/app/agent_with_health.py" ]; then
    cp /app/shared/agent_with_health.py /app/agent_with_health.py 2>/dev/null || true
fi

# Start the agent
echo "Starting agent: $AGENT_NAME with health checks"

# Check if specific agent implementation exists
if [ -f "/app/agent.py" ]; then
    python /app/agent.py
elif [ -f "/app/shared/agent_with_health.py" ]; then
    python /app/shared/agent_with_health.py
elif [ -f "/app/shared/generic_agent.py" ]; then
    python /app/shared/generic_agent.py
else
    # Use the generic agent from the shared location
    python /app/shared/agent_base.py
fi