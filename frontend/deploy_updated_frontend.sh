#!/bin/bash

# Deploy Updated Frontend Script
# This script deploys the updated frontend with full backend integration

set -e

echo "================================================"
echo "JARVIS Frontend Update Deployment"
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as appropriate user
echo -e "${YELLOW}Checking environment...${NC}"

# Install required Python packages
echo -e "${YELLOW}Installing required packages...${NC}"

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Using existing virtual environment"
    source venv/bin/activate
else
    echo "Creating virtual environment"
    python3 -m venv venv
    source venv/bin/activate
fi

# Install/upgrade required packages
pip install --upgrade pip
pip install --upgrade streamlit streamlit-mic-recorder streamlit-lottie streamlit-chat streamlit-option-menu
pip install --upgrade plotly numpy psutil docker requests websocket-client nest-asyncio python-dateutil

# Backup current app.py if it exists
if [ -f "app.py" ]; then
    echo -e "${YELLOW}Backing up current app.py...${NC}"
    cp app.py app_backup_$(date +%Y%m%d_%H%M%S).py
fi

# Deploy the updated app
echo -e "${YELLOW}Deploying updated frontend...${NC}"
cp app_updated.py app.py

# Check backend connectivity
echo -e "${YELLOW}Checking backend connectivity...${NC}"
BACKEND_URL="http://localhost:10200/health"
if curl -s -o /dev/null -w "%{http_code}" $BACKEND_URL | grep -q "200"; then
    echo -e "${GREEN}✓ Backend is running and accessible${NC}"
else
    echo -e "${RED}✗ Backend is not accessible at localhost:10200${NC}"
    echo -e "${YELLOW}Please ensure the backend is running:${NC}"
    echo "  cd /opt/sutazaiapp/backend"
    echo "  ./start_backend.sh"
fi

# Create startup script if it doesn't exist
if [ ! -f "start_frontend.sh" ]; then
    echo -e "${YELLOW}Creating startup script...${NC}"
    cat > start_frontend.sh << 'EOF'
#!/bin/bash
# JARVIS Frontend Startup Script

# Activate virtual environment
source venv/bin/activate

# Set Streamlit configuration
export STREAMLIT_SERVER_PORT=11000
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_THEME_BASE="dark"
export STREAMLIT_THEME_PRIMARY_COLOR="#00D4FF"

# Start Streamlit
echo "Starting JARVIS Frontend on port 11000..."
streamlit run app.py \
    --server.port=11000 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    --theme.base="dark" \
    --theme.primaryColor="#00D4FF" \
    --theme.backgroundColor="#0A0E27" \
    --theme.secondaryBackgroundColor="#1A1F3A" \
    --theme.textColor="#E6F3FF"
EOF
    chmod +x start_frontend.sh
fi

# Create systemd service file for auto-start (optional)
echo -e "${YELLOW}Creating systemd service file...${NC}"
cat > jarvis-frontend.service << EOF
[Unit]
Description=JARVIS Frontend Service
After=network.target docker.service

[Service]
Type=simple
User=$USER
WorkingDirectory=/opt/sutazaiapp/frontend
ExecStart=/opt/sutazaiapp/frontend/start_frontend.sh
Restart=always
RestartSec=10
StandardOutput=append:/opt/sutazaiapp/frontend/jarvis-frontend.log
StandardError=append:/opt/sutazaiapp/frontend/jarvis-frontend-error.log

[Install]
WantedBy=multi-user.target
EOF

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}Frontend update deployment complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "To start the frontend:"
echo -e "${YELLOW}  ./start_frontend.sh${NC}"
echo ""
echo "To install as a system service (requires sudo):"
echo -e "${YELLOW}  sudo cp jarvis-frontend.service /etc/systemd/system/${NC}"
echo -e "${YELLOW}  sudo systemctl daemon-reload${NC}"
echo -e "${YELLOW}  sudo systemctl enable jarvis-frontend${NC}"
echo -e "${YELLOW}  sudo systemctl start jarvis-frontend${NC}"
echo ""
echo "Access the JARVIS interface at:"
echo -e "${GREEN}  http://localhost:11000${NC}"
echo ""
echo "Features enabled:"
echo "  ✓ Real-time chat with backend"
echo "  ✓ Voice commands and TTS"
echo "  ✓ WebSocket for live updates"
echo "  ✓ Model selection"
echo "  ✓ Agent orchestration"
echo "  ✓ System monitoring"
echo "  ✓ Docker container stats"
echo ""

# Test import of all required modules
echo -e "${YELLOW}Testing module imports...${NC}"
python3 << EOF
try:
    import streamlit
    from services.backend_client_fixed import BackendClient
    from components.chat_interface import ChatInterface
    from components.voice_assistant import VoiceAssistant
    from components.system_monitor import SystemMonitor
    from config.settings import settings
    print("✓ All modules imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    exit(1)
EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ All dependencies verified${NC}"
else
    echo -e "${RED}✗ Some dependencies are missing${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Deployment successful! You can now start the frontend.${NC}"