#!/bin/bash
# Agent startup script

# Install dependencies if not already installed
if ! python -c "import requests" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install --no-cache-dir requests asyncio python-dotenv pydantic httpx aiohttp PyYAML
fi

# Copy agent_base.py if it doesn't exist
if [ ! -f "/app/agent_base.py" ]; then
    cp /app/shared/agent_base.py /app/agent_base.py 2>/dev/null || true
fi

# Start the agent
echo "Starting agent: $AGENT_NAME"
python /app/agent.py