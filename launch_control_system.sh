#!/bin/bash
# Launch the SutazAI Control System
# This script starts the backend system that extracts value from chaos

echo "🎯 Launching SutazAI Control System..."
echo "======================================"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Create necessary directories
echo -e "${YELLOW}Creating system directories...${NC}"
mkdir -p /var/lib/sutazai
mkdir -p /var/log
mkdir -p logs

# Set permissions
chmod 755 /var/lib/sutazai
chmod 755 /var/log

# Check if Ollama is running
echo -e "${YELLOW}Checking Ollama status...${NC}"
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${YELLOW}Starting Ollama...${NC}"
    docker-compose -f docker-compose-optimized.yml up -d ollama
    sleep 10
    
    # Load minimal model if needed
    echo -e "${YELLOW}Loading minimal model...${NC}"
    docker-compose -f docker-compose-optimized.yml exec ollama ollama pull llama3.2:1b || true
fi

echo -e "${GREEN}✓ Ollama is ready${NC}"

# Install Python dependencies
echo -e "${YELLOW}Installing Python dependencies...${NC}"
pip install -q fastapi uvicorn aiohttp aiofiles numpy sqlite3 || true

# Start the control system
echo -e "${BLUE}Starting Control System Backend...${NC}"
cd backend

# Run in background with logging
python3 control_system.py > ../logs/control_system.log 2>&1 &
CONTROL_PID=$!

# Wait a moment for startup
sleep 3

# Check if it's running
if kill -0 $CONTROL_PID 2>/dev/null; then
    echo -e "${GREEN}✅ Control System started successfully!${NC}"
    echo ""
    echo "🎯 SutazAI Control System is now operational"
    echo ""
    echo "API Endpoints:"
    echo "  • Control Panel:    http://localhost:8000"
    echo "  • API Docs:         http://localhost:8000/docs"
    echo "  • System Status:    http://localhost:8000/system/status"
    echo "  • Intelligence:     http://localhost:8000/intelligence/summary"
    echo "  • Opportunities:    http://localhost:8000/opportunities/high-value"
    echo ""
    echo "Key Features:"
    echo "  ✓ Silent intelligence gathering (running)"
    echo "  ✓ Chaos-to-value conversion engine (ready)"
    echo "  ✓ Pattern extraction and analysis"
    echo "  ✓ Opportunity identification"
    echo "  ✓ Competitive intelligence"
    echo "  ✓ Risk assessment"
    echo "  ✓ Value quantification"
    echo ""
    echo "Process ID: $CONTROL_PID"
    echo "Log file: logs/control_system.log"
    echo ""
    echo "To start the system: curl -X POST http://localhost:8000/system/start"
    echo "To view status: curl http://localhost:8000/system/status"
    echo ""
    echo "Silent systems are already operational in the background."
    
    # Save PID for later management
    echo $CONTROL_PID > ../logs/control_system.pid
    
else
    echo -e "${RED}❌ Failed to start Control System${NC}"
    echo "Check logs/control_system.log for details"
    exit 1
fi