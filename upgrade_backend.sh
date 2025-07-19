#!/bin/bash
# SutazAI Backend Upgrade Script
# Safely transitions from current backend to enhanced enterprise backend

echo "🚀 SutazAI Backend Upgrade Process"
echo "=================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Please run as root${NC}"
    exit 1
fi

# Function to check service status
check_service() {
    if pgrep -f "$1" > /dev/null; then
        echo -e "${GREEN}✓ $2 is running${NC}"
        return 0
    else
        echo -e "${RED}✗ $2 is not running${NC}"
        return 1
    fi
}

# Function to stop process gracefully
stop_process() {
    echo -e "${YELLOW}Stopping $1...${NC}"
    pkill -SIGTERM -f "$2" 2>/dev/null
    sleep 2
    if pgrep -f "$2" > /dev/null; then
        echo -e "${YELLOW}Force stopping $1...${NC}"
        pkill -SIGKILL -f "$2" 2>/dev/null
    fi
}

# 1. Check current system status
echo -e "\n${YELLOW}1. Checking current system status${NC}"
check_service "intelligent_backend_final.py" "Current Backend"
CURRENT_BACKEND_RUNNING=$?

check_service "simple_backend_api.py" "Simple Backend API"
check_service "ollama serve" "Ollama"
check_service "postgres" "PostgreSQL"
check_service "redis-server" "Redis"

# 2. Install additional Python dependencies
echo -e "\n${YELLOW}2. Installing required Python packages${NC}"
pip3 install -q aiofiles prometheus-client websockets python-multipart

# 3. Create necessary directories
echo -e "\n${YELLOW}3. Creating required directories${NC}"
mkdir -p /opt/sutazaiapp/uploads
mkdir -p /opt/sutazaiapp/logs
mkdir -p /opt/sutazaiapp/data/uploads
mkdir -p /opt/sutazaiapp/data/documents
chmod 755 /opt/sutazaiapp/uploads /opt/sutazaiapp/logs /opt/sutazaiapp/data

# 4. Backup current configuration
echo -e "\n${YELLOW}4. Backing up current configuration${NC}"
BACKUP_DIR="/opt/sutazaiapp/backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp intelligent_backend_final.py "$BACKUP_DIR/" 2>/dev/null

# 5. Stop current backend if running
if [ $CURRENT_BACKEND_RUNNING -eq 0 ]; then
    echo -e "\n${YELLOW}5. Stopping current backend${NC}"
    stop_process "intelligent_backend_final.py" "intelligent_backend_final.py"
    sleep 3
fi

# 6. Start enhanced backend
echo -e "\n${YELLOW}6. Starting enhanced enterprise backend${NC}"
cd /opt/sutazaiapp

# Check if enterprise backend exists
if [ ! -f "intelligent_backend_enterprise.py" ]; then
    echo -e "${RED}Error: intelligent_backend_enterprise.py not found${NC}"
    echo -e "${YELLOW}Restarting original backend...${NC}"
    nohup python3 intelligent_backend_final.py > logs/backend.log 2>&1 &
    exit 1
fi

# Start the enhanced backend
nohup python3 intelligent_backend_enterprise.py > logs/backend_enterprise.log 2>&1 &
BACKEND_PID=$!

# Wait for startup
echo -e "${YELLOW}Waiting for backend to start...${NC}"
sleep 5

# 7. Verify new backend is running
echo -e "\n${YELLOW}7. Verifying enhanced backend${NC}"
if check_service "intelligent_backend_enterprise.py" "Enhanced Backend"; then
    # Test API endpoints
    echo -e "${YELLOW}Testing API endpoints...${NC}"
    
    # Test health endpoint
    HEALTH_RESPONSE=$(curl -s -w "\n%{http_code}" http://localhost:8000/health 2>/dev/null | tail -n1)
    if [ "$HEALTH_RESPONSE" = "200" ]; then
        echo -e "${GREEN}✓ Health endpoint responding${NC}"
    else
        echo -e "${RED}✗ Health endpoint not responding${NC}"
    fi
    
    # Test models endpoint
    MODELS_RESPONSE=$(curl -s -w "\n%{http_code}" http://localhost:8000/api/models 2>/dev/null | tail -n1)
    if [ "$MODELS_RESPONSE" = "200" ]; then
        echo -e "${GREEN}✓ Models endpoint responding${NC}"
    else
        echo -e "${RED}✗ Models endpoint not responding${NC}"
    fi
    
    echo -e "\n${GREEN}✅ Backend upgrade successful!${NC}"
    echo -e "${GREEN}Enhanced backend is running on port 8000${NC}"
    echo -e "\n${YELLOW}New features available:${NC}"
    echo "  - WebSocket support at ws://localhost:8000/ws/{client_id}"
    echo "  - Enhanced caching for better performance"
    echo "  - Multi-agent system for complex tasks"
    echo "  - Batch processing endpoint"
    echo "  - File upload support"
    echo "  - Comprehensive monitoring and metrics"
    echo "  - Real-time alerts and notifications"
    
    echo -e "\n${YELLOW}API Documentation:${NC}"
    echo "  - Swagger UI: http://localhost:8000/api/docs"
    echo "  - ReDoc: http://localhost:8000/api/redoc"
    
else
    echo -e "${RED}✗ Enhanced backend failed to start${NC}"
    echo -e "${YELLOW}Rolling back to original backend...${NC}"
    
    # Stop failed backend
    stop_process "intelligent_backend_enterprise.py" "intelligent_backend_enterprise.py"
    
    # Restart original backend
    nohup python3 intelligent_backend_final.py > logs/backend.log 2>&1 &
    sleep 3
    
    if check_service "intelligent_backend_final.py" "Original Backend"; then
        echo -e "${GREEN}✓ Successfully rolled back to original backend${NC}"
    else
        echo -e "${RED}✗ Rollback failed! Manual intervention required${NC}"
        exit 1
    fi
fi

# 8. Show logs location
echo -e "\n${YELLOW}Log files:${NC}"
echo "  - Current log: /opt/sutazaiapp/logs/backend_enterprise.log"
echo "  - Backup location: $BACKUP_DIR"

echo -e "\n${YELLOW}To monitor logs:${NC}"
echo "  tail -f /opt/sutazaiapp/logs/backend_enterprise.log"

echo -e "\n${GREEN}Upgrade process complete!${NC}"