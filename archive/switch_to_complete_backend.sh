#!/bin/bash
# Switch to the complete backend with full Ollama and agent support

echo "🔄 Switching to SutazAI Complete Backend v11.0"
echo "============================================"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Please run as root${NC}"
    exit 1
fi

# Stop all existing backend processes
echo -e "${YELLOW}Stopping existing backend processes...${NC}"
pkill -f "intelligent_backend" 2>/dev/null
pkill -f "simple_backend" 2>/dev/null
systemctl stop sutazai-backend 2>/dev/null
sleep 3

# Check if Ollama is running
echo -e "${YELLOW}Checking Ollama service...${NC}"
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo -e "${RED}Ollama is not running! Starting it...${NC}"
    docker start sutazai-ollama 2>/dev/null || docker run -d --name sutazai-ollama -p 11434:11434 ollama/ollama:latest
    sleep 5
fi

# List available models
echo -e "${YELLOW}Available Ollama models:${NC}"
curl -s http://localhost:11434/api/tags | jq -r '.models[].name' 2>/dev/null || echo "No models found"

# Pull required models if missing
echo -e "${YELLOW}Ensuring required models are available...${NC}"
docker exec sutazai-ollama ollama pull llama3.2:1b &
docker exec sutazai-ollama ollama pull qwen2.5:3b &
echo -e "${GREEN}Model pulls initiated in background${NC}"

# Start the complete backend
echo -e "${YELLOW}Starting Complete Backend v11.0...${NC}"
cd /opt/sutazaiapp
nohup python3 intelligent_backend_complete.py > logs/backend_complete.log 2>&1 &
BACKEND_PID=$!

# Wait for backend to start
echo -e "${YELLOW}Waiting for backend to initialize...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:8000/health >/dev/null 2>&1; then
        echo -e "${GREEN}✓ Backend is running (PID: $BACKEND_PID)${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}✗ Backend failed to start${NC}"
        tail -20 logs/backend_complete.log
        exit 1
    fi
    sleep 1
done

# Test the backend
echo -e "\n${YELLOW}Testing backend functionality...${NC}"

# Test health endpoint
echo -e "${YELLOW}1. Health check:${NC}"
curl -s http://localhost:8000/health | jq -r '.status' 2>/dev/null && echo -e "${GREEN}✓ Health check passed${NC}"

# Test models endpoint
echo -e "\n${YELLOW}2. Available models:${NC}"
curl -s http://localhost:8000/api/models | jq -r '.models[].name' 2>/dev/null

# Test agents endpoint
echo -e "\n${YELLOW}3. External agents status:${NC}"
curl -s http://localhost:8000/api/agents | jq -r '.agents[] | "\(.name): \(.status)"' 2>/dev/null

# Test chat with Ollama
echo -e "\n${YELLOW}4. Testing Ollama integration:${NC}"
RESPONSE=$(curl -s -X POST http://localhost:8000/api/chat \
    -H "Content-Type: application/json" \
    -d '{"message": "What is 2+2?", "model": "llama3.2:1b"}' 2>/dev/null)

if [ $? -eq 0 ]; then
    OLLAMA_SUCCESS=$(echo $RESPONSE | jq -r '.ollama_success' 2>/dev/null)
    if [ "$OLLAMA_SUCCESS" = "true" ]; then
        echo -e "${GREEN}✓ Ollama integration working!${NC}"
        echo "Response: $(echo $RESPONSE | jq -r '.response' 2>/dev/null | head -c 100)..."
    else
        echo -e "${YELLOW}⚠ Ollama not used, fallback response returned${NC}"
    fi
else
    echo -e "${RED}✗ Chat test failed${NC}"
fi

# Create systemd service for the new backend
echo -e "\n${YELLOW}Creating systemd service...${NC}"
cat > /etc/systemd/system/sutazai-complete-backend.service <<EOF
[Unit]
Description=SutazAI Complete Backend v11.0
After=network.target docker.service

[Service]
Type=simple
User=root
WorkingDirectory=/opt/sutazaiapp
ExecStart=/usr/bin/python3 /opt/sutazaiapp/intelligent_backend_complete.py
Restart=always
RestartSec=10
StandardOutput=append:/opt/sutazaiapp/logs/backend_complete.log
StandardError=append:/opt/sutazaiapp/logs/backend_complete.log

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable sutazai-complete-backend
echo -e "${GREEN}✓ Systemd service created${NC}"

# Show status
echo -e "\n${GREEN}======================================${NC}"
echo -e "${GREEN}✅ Complete Backend v11.0 is running!${NC}"
echo -e "${GREEN}======================================${NC}"

echo -e "\n${YELLOW}Features enabled:${NC}"
echo "• Full Ollama integration with proper error handling"
echo "• External AI agent support (AutoGPT, CrewAI, etc.)"
echo "• Intelligent routing based on query type"
echo "• Real-time metrics and monitoring"
echo "• Fallback responses when services unavailable"

echo -e "\n${YELLOW}Logs:${NC}"
echo "• Backend log: tail -f /opt/sutazaiapp/logs/backend_complete.log"
echo "• System status: systemctl status sutazai-complete-backend"

echo -e "\n${YELLOW}To test external agents:${NC}"
echo "• Start agents: docker-compose up -d autogpt crewai privategpt"
echo "• Use agent: curl -X POST http://localhost:8000/api/chat -H 'Content-Type: application/json' -d '{\"message\": \"Create a plan\", \"use_agent\": \"autogpt\"}'"

echo -e "\n${GREEN}Backend switch complete!${NC}"