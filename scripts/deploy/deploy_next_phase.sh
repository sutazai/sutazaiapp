#!/bin/bash

# Deploy Next Phase Helper Script
# Simplifies deployment of optimization phases

set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Function to show usage
show_usage() {
    cat << EOF
${BLUE}SutazAI Next Phase Deployment Helper${NC}

Usage: $0 [PHASE]

Phases:
  vector     - Deploy vector databases (ChromaDB, Qdrant)
  monitor    - Deploy monitoring stack (Prometheus, Grafana, Loki)
  ai         - Deploy additional AI services (LangFlow, Flowise, n8n)
  all        - Deploy all remaining services
  status     - Show current deployment status

Examples:
  $0 vector    # Deploy vector databases
  $0 monitor   # Deploy monitoring stack
  $0 status    # Check what's deployed

EOF
}

# Function to check current status
check_status() {
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}Current Deployment Status${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    
    echo -e "\n${YELLOW}Core Services:${NC}"
    for service in postgres redis ollama backend frontend; do
        if docker ps --format '{{.Names}}' | grep -q "sutazai-$service"; then
            echo -e "  ${GREEN}✅ $service${NC}"
        else
            echo -e "  ${RED}❌ $service${NC}"
        fi
    done
    
    echo -e "\n${YELLOW}Vector Databases:${NC}"
    for service in chromadb qdrant neo4j; do
        if docker ps --format '{{.Names}}' | grep -q "sutazai-$service"; then
            echo -e "  ${GREEN}✅ $service${NC}"
        else
            echo -e "  ${RED}❌ $service (not deployed)${NC}"
        fi
    done
    
    echo -e "\n${YELLOW}Monitoring Stack:${NC}"
    for service in prometheus grafana loki promtail; do
        if docker ps --format '{{.Names}}' | grep -q "sutazai-$service"; then
            echo -e "  ${GREEN}✅ $service${NC}"
        else
            echo -e "  ${RED}❌ $service (not deployed)${NC}"
        fi
    done
    
    echo -e "\n${YELLOW}AI Services:${NC}"
    for service in langflow flowise n8n dify; do
        if docker ps --format '{{.Names}}' | grep -q "sutazai-$service"; then
            echo -e "  ${GREEN}✅ $service${NC}"
        else
            echo -e "  ${RED}❌ $service (not deployed)${NC}"
        fi
    done
    
    echo -e "\n${YELLOW}System Resources:${NC}"
    free -h | grep Mem | awk '{printf "  Memory: %s / %s (%.0f%%)\n", $3, $2, ($3/$2)*100}'
    echo -e "  CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}')% used"
    echo -e "  Containers: $(docker ps -q | wc -l) running"
}

# Function to deploy vector databases
deploy_vector() {
    echo -e "${BLUE}Deploying Vector Databases...${NC}"
    echo -e "${YELLOW}This will deploy: ChromaDB, Qdrant${NC}"
    read -p "Continue? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd /opt/sutazaiapp
        ./scripts/deploy_complete_system.sh deploy --services chromadb,qdrant
        echo -e "${GREEN}✅ Vector databases deployment initiated${NC}"
    else
        echo -e "${RED}Deployment cancelled${NC}"
    fi
}

# Function to deploy monitoring
deploy_monitor() {
    echo -e "${BLUE}Deploying Monitoring Stack...${NC}"
    echo -e "${YELLOW}This will deploy: Prometheus, Grafana, Loki, Promtail${NC}"
    read -p "Continue? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd /opt/sutazaiapp
        ./scripts/deploy_complete_system.sh deploy --services prometheus,grafana,loki,promtail
        echo -e "${GREEN}✅ Monitoring stack deployment initiated${NC}"
        echo -e "${YELLOW}📊 Grafana will be available at: http://localhost:3000${NC}"
        echo -e "${YELLOW}   Default login: admin/admin${NC}"
    else
        echo -e "${RED}Deployment cancelled${NC}"
    fi
}

# Function to deploy AI services
deploy_ai() {
    echo -e "${BLUE}Deploying Additional AI Services...${NC}"
    echo -e "${YELLOW}This will deploy: LangFlow, Flowise, n8n, Dify${NC}"
    read -p "Continue? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd /opt/sutazaiapp
        ./scripts/deploy_complete_system.sh deploy --services langflow,flowise,n8n,dify
        echo -e "${GREEN}✅ AI services deployment initiated${NC}"
    else
        echo -e "${RED}Deployment cancelled${NC}"
    fi
}

# Function to deploy everything
deploy_all() {
    echo -e "${BLUE}Deploying All Remaining Services...${NC}"
    echo -e "${YELLOW}This will deploy all services not currently running${NC}"
    echo -e "${RED}Warning: This requires significant resources!${NC}"
    read -p "Continue? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd /opt/sutazaiapp
        ./scripts/deploy_complete_system.sh deploy --profile full
        echo -e "${GREEN}✅ Full deployment initiated${NC}"
    else
        echo -e "${RED}Deployment cancelled${NC}"
    fi
}

# Main script
case "${1:-}" in
    vector)
        deploy_vector
        ;;
    monitor)
        deploy_monitor
        ;;
    ai)
        deploy_ai
        ;;
    all)
        deploy_all
        ;;
    status)
        check_status
        ;;
    *)
        show_usage
        exit 1
        ;;
esac

# Show post-deployment status
if [[ "${1:-}" =~ ^(vector|monitor|ai|all)$ ]]; then
    echo -e "\n${YELLOW}Waiting for services to start...${NC}"
    sleep 10
    echo
    check_status
    
    echo -e "\n${BLUE}💡 Next Steps:${NC}"
    echo -e "  1. Check service health: ${GREEN}./scripts/run_deployment_verification.sh${NC}"
    echo -e "  2. View logs: ${GREEN}./scripts/live_logs.sh${NC}"
    echo -e "  3. Access services at their respective ports"
fi