#!/bin/bash
# Complete SutazAI Control System Startup Script

echo "🎯 Starting SutazAI Control System"
echo "=================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Check if we're in the right directory
if [ ! -f "control_interface.py" ]; then
    echo -e "${RED}Error: Please run this script from the SutazAI directory${NC}"
    exit 1
fi

# Step 1: Start core infrastructure if needed
echo -e "${YELLOW}Step 1: Checking core infrastructure...${NC}"
if ! docker ps | grep -q postgres; then
    echo "Starting minimal infrastructure..."
    docker-compose -f docker-compose-optimized.yml up -d postgresql redis ollama
    sleep 15
fi

# Step 2: Launch the backend control system
echo -e "${YELLOW}Step 2: Starting backend control system...${NC}"
./launch_control_system.sh

# Wait for backend to be ready
echo -e "${YELLOW}Waiting for backend to be ready...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Backend is ready${NC}"
        break
    fi
    echo "Waiting... ($i/30)"
    sleep 2
done

# Step 3: Start the web interface
echo -e "${YELLOW}Step 3: Starting web interface...${NC}"

# Install streamlit if not present
pip install -q streamlit plotly pandas requests || true

# Start streamlit interface
echo -e "${BLUE}Launching Control Interface...${NC}"
streamlit run control_interface.py --server.port=8502 --server.address=0.0.0.0 &
STREAMLIT_PID=$!

# Wait for streamlit to start
sleep 5

# Step 4: Auto-start the intelligence system
echo -e "${YELLOW}Step 4: Starting intelligence gathering...${NC}"
curl -s -X POST http://localhost:8000/system/start > /dev/null 2>&1

# Final status
echo -e "${GREEN}"
echo "🎉 SutazAI Control System is now fully operational!"
echo ""
echo "Access Points:"
echo "  🎯 Control Interface:  http://localhost:8502"
echo "  🔧 Backend API:        http://localhost:8000"
echo "  📚 API Documentation:  http://localhost:8000/docs"
echo ""
echo "Key Features Now Active:"
echo "  ✅ Silent Intelligence Gathering"
echo "  ✅ Chaos-to-Value Conversion"
echo "  ✅ Pattern Recognition"
echo "  ✅ Opportunity Detection"
echo "  ✅ Risk Assessment"
echo "  ✅ Competitive Intelligence"
echo "  ✅ Value Quantification"
echo ""
echo "System Status:"
curl -s http://localhost:8000/system/status | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    print(f'  • Intelligence Level: {data.get(\"silent_operator\", {}).get(\"intelligence_level\", \"Unknown\")}')
    print(f'  • Total Value Extracted: {data.get(\"total_value_extracted\", 0):.2f}')
    print(f'  • System Running: {\"Yes\" if data.get(\"system_running\") else \"No\"}')
except:
    print('  • Status: Loading...')
"
echo ""
echo "Process IDs:"
echo "  • Backend: $(cat logs/control_system.pid 2>/dev/null || echo 'Unknown')"
echo "  • Interface: $STREAMLIT_PID"
echo ""
echo "To stop the system:"
echo "  kill \$(cat logs/control_system.pid) && kill $STREAMLIT_PID"
echo ""
echo -e "${NC}"

# Save streamlit PID
echo $STREAMLIT_PID > logs/streamlit.pid

echo "The system is now extracting value from chaos autonomously."
echo "Navigate to http://localhost:8502 to control and monitor the system."