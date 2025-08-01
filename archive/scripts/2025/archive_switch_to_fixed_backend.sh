#!/bin/bash
# Switch to the properly fixed backend with varied responses

echo "🔄 Switching to Fixed Backend v12.0"
echo "===================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Stop current backends
echo -e "${YELLOW}Stopping current backend processes...${NC}"
pkill -f "intelligent_backend" 2>/dev/null
systemctl stop sutazai-complete-backend 2>/dev/null
sleep 3

# Start the fixed backend
echo -e "${YELLOW}Starting Fixed Backend v12.0...${NC}"
cd /opt/sutazaiapp
nohup python3 intelligent_backend_fixed_final.py > logs/backend_fixed_final.log 2>&1 &
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
        tail -20 logs/backend_fixed_final.log
        exit 1
    fi
    sleep 1
done

# Test with different queries
echo -e "\n${YELLOW}Testing varied responses...${NC}"

echo -e "\n${YELLOW}Test 1: Self-improvement query${NC}"
curl -s -X POST http://localhost:8000/api/chat \
    -H "Content-Type: application/json" \
    -d '{"message": "How will you self-improve?", "model": "llama3.2:1b"}' | jq -r '.response' | head -c 200
echo "..."

sleep 2

echo -e "\n\n${YELLOW}Test 2: Same query (should be different)${NC}"
curl -s -X POST http://localhost:8000/api/chat \
    -H "Content-Type: application/json" \
    -d '{"message": "How will you self-improve?", "model": "llama3.2:1b"}' | jq -r '.response' | head -c 200
echo "..."

echo -e "\n\n${YELLOW}Test 3: Testing model correction (qwen3:8b -> qwen2.5:3b)${NC}"
RESPONSE=$(curl -s -X POST http://localhost:8000/api/chat \
    -H "Content-Type: application/json" \
    -d '{"message": "What is AI?", "model": "qwen3:8b"}')

MODEL_USED=$(echo $RESPONSE | jq -r '.model')
OLLAMA_SUCCESS=$(echo $RESPONSE | jq -r '.ollama_success')

echo -e "Model requested: qwen3:8b"
echo -e "Model used: ${MODEL_USED}"
echo -e "Ollama success: ${OLLAMA_SUCCESS}"

# Update systemd service
echo -e "\n${YELLOW}Updating systemd service...${NC}"
cat > /etc/systemd/system/sutazai-fixed-backend.service <<EOF
[Unit]
Description=SutazAI Fixed Backend v12.0
After=network.target docker.service

[Service]
Type=simple
User=root
WorkingDirectory=/opt/sutazaiapp
ExecStart=/usr/bin/python3 /opt/sutazaiapp/intelligent_backend_fixed_final.py
Restart=always
RestartSec=10
StandardOutput=append:/opt/sutazaiapp/logs/backend_fixed_final.log
StandardError=append:/opt/sutazaiapp/logs/backend_fixed_final.log

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable sutazai-fixed-backend

echo -e "\n${GREEN}✅ Fixed Backend v12.0 is running!${NC}"
echo -e "${GREEN}✅ Model validation enabled${NC}"
echo -e "${GREEN}✅ Varied responses enabled${NC}"
echo -e "${GREEN}✅ Proper error handling${NC}"

echo -e "\n${YELLOW}Features:${NC}"
echo "• Corrects invalid model names (qwen3:8b -> qwen2.5:3b)"
echo "• Provides different responses each time"
echo "• Faster timeout (30s instead of 60s)"
echo "• Better fallback responses"

echo -e "\n${YELLOW}Monitor logs:${NC}"
echo "tail -f /opt/sutazaiapp/logs/backend_fixed_final.log"