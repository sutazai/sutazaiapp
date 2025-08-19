#!/bin/bash
# Run Claude Code with accuracy-focused settings

# Set up proxy to intercept and modify requests
export HTTP_PROXY=http://127.0.0.1:8080
export HTTPS_PROXY=http://127.0.0.1:8080

# Install mitmproxy if not installed
if ! command -v mitmproxy &> /dev/null; then
    echo "Installing mitmproxy..."
    pip install mitmproxy
fi

# Start mitmproxy with the interceptor script in background
echo "Starting temperature interceptor..."
mitmdump -s /opt/sutazaiapp/scripts/claude-temp-interceptor.py &
PROXY_PID=$!

# Wait for proxy to start
sleep 2

# Run Claude Code with the proxy
echo "Starting Claude Code with temperature 0.2..."
claude "$@"

# Clean up
kill $PROXY_PID 2>/dev/null