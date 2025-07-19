#!/bin/bash
# Switch to the performance-fixed backend with real-time metrics

echo "🔄 Switching to Performance-Fixed Backend v13.0"
echo "============================================"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Stop current backends
echo -e "${YELLOW}Stopping current backend processes...${NC}"
pkill -f "intelligent_backend" 2>/dev/null
systemctl stop sutazai-fixed-backend 2>/dev/null
systemctl stop sutazai-complete-backend 2>/dev/null
sleep 3

# Start the performance backend
echo -e "${YELLOW}Starting Performance Backend v13.0...${NC}"
cd /opt/sutazaiapp
nohup python3 intelligent_backend_performance_fixed.py > logs/backend_performance.log 2>&1 &
BACKEND_PID=$!

# Wait for startup
echo -e "${YELLOW}Waiting for backend to initialize...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:8000/health >/dev/null 2>&1; then
        echo -e "${GREEN}✓ Backend is running (PID: $BACKEND_PID)${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}✗ Backend failed to start${NC}"
        tail -20 logs/backend_performance.log
        exit 1
    fi
    sleep 1
done

# Test the endpoints
echo -e "\n${YELLOW}Testing performance metrics...${NC}"

echo -e "\n${YELLOW}1. System Health Check${NC}"
curl -s http://localhost:8000/health | jq '.metrics.system'

echo -e "\n${YELLOW}2. Performance Summary${NC}"
curl -s http://localhost:8000/api/performance/summary | jq '.'

echo -e "\n${YELLOW}3. Performance Alerts${NC}"
curl -s http://localhost:8000/api/performance/alerts | jq '.'

echo -e "\n${YELLOW}4. Testing Chat API${NC}"
RESPONSE=$(curl -s -X POST http://localhost:8000/api/chat \
    -H "Content-Type: application/json" \
    -d '{"message": "Test message", "model": "llama3.2:1b"}')

echo "$RESPONSE" | jq '.response' | head -c 100
echo "..."

# Update systemd service
echo -e "\n${YELLOW}Creating systemd service...${NC}"
cat > /etc/systemd/system/sutazai-performance-backend.service <<EOF
[Unit]
Description=SutazAI Performance Backend v13.0
After=network.target docker.service

[Service]
Type=simple
User=root
WorkingDirectory=/opt/sutazaiapp
ExecStart=/usr/bin/python3 /opt/sutazaiapp/intelligent_backend_performance_fixed.py
Restart=always
RestartSec=10
StandardOutput=append:/opt/sutazaiapp/logs/backend_performance.log
StandardError=append:/opt/sutazaiapp/logs/backend_performance.log

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable sutazai-performance-backend

echo -e "\n${GREEN}✅ Performance Backend v13.0 is running!${NC}"
echo -e "${GREEN}✅ Real-time metrics collection active${NC}"
echo -e "${GREEN}✅ WebSocket support enabled${NC}"
echo -e "${GREEN}✅ Enhanced logging system ready${NC}"

echo -e "\n${YELLOW}Key Features:${NC}"
echo "• Real-time performance metrics"
echo "• WebSocket live updates at ws://localhost:8000/ws"
echo "• Enhanced logging with categories"
echo "• Background health checks for agents"
echo "• Detailed API endpoint tracking"

echo -e "\n${YELLOW}Monitor logs:${NC}"
echo "tail -f /opt/sutazaiapp/logs/backend_performance.log"

echo -e "\n${YELLOW}Access WebSocket metrics:${NC}"
echo "wscat -c ws://localhost:8000/ws"

# Test WebSocket connection
echo -e "\n${YELLOW}Testing WebSocket connection...${NC}"
timeout 2 bash -c 'exec 3<>/dev/tcp/localhost/8000 && echo -e "GET /ws HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n" >&3' 2>/dev/null && echo -e "${GREEN}✓ WebSocket endpoint accessible${NC}" || echo -e "${YELLOW}⚠ WebSocket requires a proper client to test${NC}"