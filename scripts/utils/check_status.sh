#!/bin/bash
# Check the status of all SutazaiApp components

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}==========================================${NC}"
echo -e "${BLUE}     SutazaiApp Status Check Tool        ${NC}"
echo -e "${BLUE}==========================================${NC}"
echo ""

# Check if services are running
echo -e "${YELLOW}Checking systemd services...${NC}"
echo -e "${YELLOW}-------------------------${NC}"
if systemctl is-active --quiet sutazaiapp; then
    echo -e "SutazaiApp API Service: ${GREEN}RUNNING${NC}"
else
    echo -e "SutazaiApp API Service: ${RED}STOPPED${NC}"
fi

if systemctl is-active --quiet sutazai-orchestrator; then
    echo -e "SutazaiApp Orchestrator Service: ${GREEN}RUNNING${NC}"
else
    echo -e "SutazaiApp Orchestrator Service: ${RED}STOPPED${NC}"
fi
echo ""

# Check if the API is responding
echo -e "${YELLOW}Checking API health...${NC}"
echo -e "${YELLOW}-------------------${NC}"
HEALTH_RESPONSE=$(curl -s http://localhost:8000/health)
if [[ $HEALTH_RESPONSE == *"healthy"* ]]; then
    echo -e "API Health Check: ${GREEN}OK${NC}"
    echo -e "Response: $HEALTH_RESPONSE"
else
    echo -e "API Health Check: ${RED}FAILED${NC}"
    echo -e "Response: $HEALTH_RESPONSE"
fi
echo ""

# Check Python version
echo -e "${YELLOW}Checking Python Version...${NC}"
echo -e "${YELLOW}------------------------${NC}"
PYTHON_VERSION=$(source /opt/sutazaiapp/venv/bin/activate && python --version)
if [[ $PYTHON_VERSION == *"3.11"* ]]; then
    echo -e "Python Version: ${GREEN}$PYTHON_VERSION${NC} (correct version)"
else
    echo -e "Python Version: ${RED}$PYTHON_VERSION${NC} (expected 3.11.x)"
fi
echo ""

# Check for log files
echo -e "${YELLOW}Checking log files...${NC}"
echo -e "${YELLOW}-------------------${NC}"
for log_file in "backend.log" "orchestrator.log" "access.log" "error.log"; do
    if [ -f "/opt/sutazaiapp/logs/$log_file" ]; then
        LOG_SIZE=$(du -h "/opt/sutazaiapp/logs/$log_file" | cut -f1)
        echo -e "$log_file: ${GREEN}EXISTS${NC} (size: $LOG_SIZE)"
    else
        echo -e "$log_file: ${RED}MISSING${NC}"
    fi
done
echo ""

# Check directory structure
echo -e "${YELLOW}Checking directory permissions...${NC}"
echo -e "${YELLOW}-----------------------------${NC}"
for dir in "/opt/sutazaiapp/logs" "/opt/sutazaiapp/data" "/opt/sutazaiapp/run" "/opt/sutazaiapp/tmp"; do
    if [ -d "$dir" ]; then
        PERMISSIONS=$(stat -c "%a" "$dir")
        if [ "$PERMISSIONS" -ge "755" ]; then
            echo -e "$dir: ${GREEN}OK${NC} (permissions: $PERMISSIONS)"
        else
            echo -e "$dir: ${YELLOW}WARNING${NC} (permissions: $PERMISSIONS, should be at least 755)"
        fi
    else
        echo -e "$dir: ${RED}MISSING${NC}"
    fi
done
echo ""

echo -e "${BLUE}==========================================${NC}"
echo -e "${BLUE}     SutazaiApp Status Check Complete    ${NC}"
echo -e "${BLUE}==========================================${NC}" 