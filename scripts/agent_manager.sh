#\!/bin/bash
# SutazAI Agent Manager - Manage all AI agents

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

show_menu() {
    clear
    echo "╔═══════════════════════════════════════════════════════════════════╗"
    echo "║                 SutazAI Agent Manager                             ║"
    echo "╚═══════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "1. Show All Agents Status"
    echo "2. Start All Agents" 
    echo "3. Stop All Agents"
    echo "4. Restart Specific Agent"
    echo "5. Show Agent Logs"
    echo "6. Show System Dashboard"
    echo "7. Register Agents with API"
    echo "8. Show Agent Categories"
    echo "9. Export Agent List"
    echo "0. Exit"
    echo ""
    echo -n "Select option: "
}

show_all_agents() {
    echo -e "\n${BLUE}All AI Agents Status:${NC}\n"
    docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "sutazai-.*agent|developer|engineer|specialist|coordinator|manager|optimizer|architect|improver|debugger|gpt|ai|crewai|aider|letta|devika|babyagi" | sort
    echo -e "\nPress Enter to continue..."
    read
}

start_all_agents() {
    echo -e "\n${YELLOW}Starting all agents...${NC}\n"
    
    # Start extended agents
    docker-compose -f docker-compose.agents-extended.yml up -d
    
    # Start remaining agents
    docker-compose -f docker-compose.agents-remaining.yml up -d
    
    echo -e "\n${GREEN}All agents started\!${NC}"
    echo -e "Press Enter to continue..."
    read
}

stop_all_agents() {
    echo -e "\n${YELLOW}Stopping all agents...${NC}\n"
    
    docker ps --format "{{.Names}}" | grep -E "agent|developer|engineer|specialist|coordinator|manager|optimizer|architect|improver|debugger|gpt|ai|crewai|aider|letta|devika|babyagi" | while read container; do
        echo "Stopping $container..."
        docker stop $container
    done
    
    echo -e "\n${GREEN}All agents stopped\!${NC}"
    echo -e "Press Enter to continue..."
    read
}

restart_agent() {
    echo -e "\n${BLUE}Available Agents:${NC}\n"
    docker ps --format "{{.Names}}" | grep -E "agent|developer|engineer|specialist|coordinator|manager|optimizer|architect|improver|debugger|gpt|ai|crewai|aider|letta|devika|babyagi" | sort | nl
    
    echo -n -e "\nEnter agent number to restart (0 to cancel): "
    read agent_num
    
    if [ "$agent_num" \!= "0" ]; then
        agent_name=$(docker ps --format "{{.Names}}" | grep -E "agent|developer|engineer|specialist|coordinator|manager|optimizer|architect|improver|debugger|gpt|ai|crewai|aider|letta|devika|babyagi" | sort | sed -n "${agent_num}p")
        if [ -n "$agent_name" ]; then
            echo -e "\n${YELLOW}Restarting $agent_name...${NC}"
            docker restart $agent_name
            echo -e "${GREEN}Done\!${NC}"
        fi
    fi
    
    echo -e "Press Enter to continue..."
    read
}

show_agent_logs() {
    echo -e "\n${BLUE}Available Agents:${NC}\n"
    docker ps --format "{{.Names}}" | grep -E "agent|developer|engineer|specialist|coordinator|manager|optimizer|architect|improver|debugger|gpt|ai|crewai|aider|letta|devika|babyagi" | sort | nl
    
    echo -n -e "\nEnter agent number to view logs (0 to cancel): "
    read agent_num
    
    if [ "$agent_num" \!= "0" ]; then
        agent_name=$(docker ps --format "{{.Names}}" | grep -E "agent|developer|engineer|specialist|coordinator|manager|optimizer|architect|improver|debugger|gpt|ai|crewai|aider|letta|devika|babyagi" | sort | sed -n "${agent_num}p")
        if [ -n "$agent_name" ]; then
            echo -e "\n${BLUE}Logs for $agent_name:${NC}\n"
            docker logs --tail 50 $agent_name
        fi
    fi
    
    echo -e "\nPress Enter to continue..."
    read
}

register_agents() {
    echo -e "\n${YELLOW}Registering agents with API...${NC}\n"
    
    # Get all running agents
    docker ps --format "{{.Names}}" | grep -E "agent|developer|engineer|specialist|coordinator|manager|optimizer|architect|improver|debugger|gpt|ai|crewai|aider|letta|devika|babyagi" | while read container; do
        agent_id=${container#sutazai-}
        agent_name=$(echo $agent_id | tr '-' ' ' | sed 's/\b\(.\)/\u\1/g')
        
        echo -n "Registering $agent_name... "
        
        curl -s -X POST http://localhost:8000/api/v1/orchestration/agents \
            -H "Content-Type: application/json" \
            -d "{\"agent_type\": \"$agent_id\", \"name\": \"$agent_name\", \"config\": {}}" \
            > /dev/null 2>&1
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓${NC}"
        else
            echo -e "${RED}✗${NC}"
        fi
    done
    
    echo -e "\nPress Enter to continue..."
    read
}

show_categories() {
    echo -e "\n${BLUE}AI Agents by Category:${NC}\n"
    
    echo -e "${YELLOW}Task Automation ($(docker ps --format "{{.Names}}" | grep -E "autogpt|agentgpt|crewai|agi|autonomous|task|coordinator|babyagi|letta" | wc -l) agents):${NC}"
    docker ps --format "{{.Names}}" | grep -E "autogpt|agentgpt|crewai|agi|autonomous|task|coordinator|babyagi|letta" | sed 's/sutazai-/  - /'
    
    echo -e "\n${YELLOW}Code Generation ($(docker ps --format "{{.Names}}" | grep -E "code|developer|engineer|aider|gpt-engineer|devin|devika" | wc -l) agents):${NC}"
    docker ps --format "{{.Names}}" | grep -E "code|developer|engineer|aider|gpt-engineer|devin|devika" | sed 's/sutazai-/  - /'
    
    echo -e "\n${YELLOW}Data Analysis ($(docker ps --format "{{.Names}}" | grep -E "data|analysis|pipeline|analyst" | wc -l) agents):${NC}"
    docker ps --format "{{.Names}}" | grep -E "data|analysis|pipeline|analyst" | sed 's/sutazai-/  - /'
    
    echo -e "\n${YELLOW}ML/AI ($(docker ps --format "{{.Names}}" | grep -E "model|training|learning|neural|quantum|federated" | wc -l) agents):${NC}"
    docker ps --format "{{.Names}}" | grep -E "model|training|learning|neural|quantum|federated" | sed 's/sutazai-/  - /'
    
    echo -e "\n${YELLOW}Security ($(docker ps --format "{{.Names}}" | grep -E "security|pentest|semgrep|kali|shellgpt" | wc -l) agents):${NC}"
    docker ps --format "{{.Names}}" | grep -E "security|pentest|semgrep|kali|shellgpt" | sed 's/sutazai-/  - /'
    
    echo -e "\nPress Enter to continue..."
    read
}

export_agent_list() {
    timestamp=$(date +%Y%m%d_%H%M%S)
    filename="agent_list_$timestamp.txt"
    
    echo "SutazAI Agent List - Generated $(date)" > $filename
    echo "=======================================" >> $filename
    echo "" >> $filename
    
    echo "Total Agents: $(docker ps --format "{{.Names}}" | grep -E "agent|developer|engineer|specialist|coordinator|manager|optimizer|architect|improver|debugger|gpt|ai|crewai|aider|letta|devika|babyagi" | wc -l)" >> $filename
    echo "" >> $filename
    
    docker ps --format "{{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "agent|developer|engineer|specialist|coordinator|manager|optimizer|architect|improver|debugger|gpt|ai|crewai|aider|letta|devika|babyagi" | sort >> $filename
    
    echo -e "\n${GREEN}Agent list exported to $filename${NC}"
    echo -e "Press Enter to continue..."
    read
}

# Main loop
while true; do
    show_menu
    read choice
    
    case $choice in
        1) show_all_agents ;;
        2) start_all_agents ;;
        3) stop_all_agents ;;
        4) restart_agent ;;
        5) show_agent_logs ;;
        6) /opt/sutazaiapp/scripts/show_system_dashboard.sh; echo -e "\nPress Enter to continue..."; read ;;
        7) register_agents ;;
        8) show_categories ;;
        9) export_agent_list ;;
        0) echo "Exiting..."; exit 0 ;;
        *) echo "Invalid option. Press Enter to continue..."; read ;;
    esac
done
