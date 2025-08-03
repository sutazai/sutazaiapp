#!/bin/sh

echo "🔧 Configuring Dashboard Environment..."

# Get environment variables
BACKEND_API_URL=${BACKEND_API_URL:-"http://localhost:8080"}
RULE_API_URL=${RULE_API_URL:-"http://localhost:8100"}
WEBSOCKET_URL=${WEBSOCKET_URL:-"ws://localhost:8080/ws"}

# Update JavaScript configuration
cat > /usr/share/nginx/html/config.js << EOF
// Auto-generated configuration
window.HYGIENE_CONFIG = {
    BACKEND_API_URL: '${BACKEND_API_URL}',
    RULE_API_URL: '${RULE_API_URL}',
    WEBSOCKET_URL: '${WEBSOCKET_URL}',
    VERSION: '3.0.0',
    BUILD_TIME: '$(date -u +"%Y-%m-%dT%H:%M:%SZ")'
};
EOF

echo "✅ Dashboard configuration updated"
echo "   Backend API: ${BACKEND_API_URL}"
echo "   Rule API: ${RULE_API_URL}"
echo "   WebSocket: ${WEBSOCKET_URL}"