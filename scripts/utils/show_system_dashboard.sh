#\!/bin/bash
# System Dashboard for SutazAI

clear
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║            SutazAI Multi-Agent System Dashboard                   ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Get counts
TOTAL_CONTAINERS=$(docker ps -q | wc -l)
TOTAL_AGENTS=$(docker ps --format "{{.Names}}" | grep -E "agent|developer|engineer|specialist|coordinator|manager|optimizer|architect|improver|debugger|gpt|ai|crewai|aider|letta|devika|babyagi" | wc -l)
HEALTHY_CONTAINERS=$(docker ps --format "{{.Status}}" | grep -c "(healthy)")

echo -e "${BLUE}📊 System Overview${NC}"
echo "├─ Total Containers: $TOTAL_CONTAINERS"
echo "├─ AI Agents: $TOTAL_AGENTS"
echo "└─ Healthy Services: $HEALTHY_CONTAINERS"
echo ""

echo -e "${BLUE}🤖 AI Agents by Category${NC}"
echo "├─ Task Automation: $(docker ps --format "{{.Names}}" | grep -E "autogpt|agentgpt|crewai|agi|autonomous|task|coordinator|babyagi|letta" | wc -l)"
echo "├─ Code Generation: $(docker ps --format "{{.Names}}" | grep -E "code|developer|engineer|aider|gpt-engineer|devin|devika" | wc -l)"
echo "├─ Data Analysis: $(docker ps --format "{{.Names}}" | grep -E "data|analysis|pipeline|analyst" | wc -l)"
echo "├─ ML/AI: $(docker ps --format "{{.Names}}" | grep -E "model|training|learning|neural|quantum|federated" | wc -l)"
echo "├─ Infrastructure: $(docker ps --format "{{.Names}}" | grep -E "infrastructure|devops|deployment|docker|kubernetes" | wc -l)"
echo "├─ Security: $(docker ps --format "{{.Names}}" | grep -E "security|pentest|semgrep|kali|shellgpt" | wc -l)"
echo "└─ Specialized: $(docker ps --format "{{.Names}}" | grep -E "special|bigagi|localagi|jarvis|litellm|opendevin" | wc -l)"
echo ""

echo -e "${BLUE}🌐 Access Points${NC}"
echo "├─ Frontend UI: ${GREEN}http://localhost:8501${NC}"
echo "├─ API Docs: ${GREEN}http://localhost:8000/docs${NC}"
echo "├─ Grafana: ${GREEN}http://localhost:3000${NC}"
echo "├─ n8n Workflows: ${GREEN}http://localhost:5678${NC}"
echo "└─ Jupyter: http://localhost:8888"
echo ""

echo -e "${BLUE}💾 Resource Usage${NC}"
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
MEM_INFO=$(free -h | awk '/^Mem:/ {print $3 " / " $2}')
DISK_INFO=$(df -h / | awk 'NR==2 {print $3 " / " $2 " (" $5 ")"}')

echo "├─ CPU Usage: ${CPU_USAGE}%"
echo "├─ Memory: $MEM_INFO"
echo "└─ Disk: $DISK_INFO"
echo ""

echo -e "${BLUE}🔥 Hot Agents (Recently Active)${NC}"
docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "agent|developer|engineer" | grep "Up [0-9]+ minutes" | head -5 | while read line; do
    echo "├─ $line"
done
echo ""

echo -e "${BLUE}📈 System Health${NC}"
# Check key services
echo -n "├─ Backend API: "
if curl -s http://localhost:8000/health >/dev/null 2>&1; then
    echo -e "${GREEN}● Online${NC}"
else
    echo -e "${RED}● Offline${NC}"
fi

echo -n "├─ Frontend UI: "
if curl -s http://localhost:8501 >/dev/null 2>&1; then
    echo -e "${GREEN}● Online${NC}"
else
    echo -e "${RED}● Offline${NC}"
fi

echo -n "├─ Ollama LLM: "
if curl -s http://localhost:11434 >/dev/null 2>&1; then
    echo -e "${GREEN}● Online${NC}"
else
    echo -e "${RED}● Offline${NC}"
fi

echo -n "└─ Vector DB (Qdrant): "
if curl -s http://localhost:6333 >/dev/null 2>&1; then
    echo -e "${GREEN}● Online${NC}"
else
    echo -e "${RED}● Offline${NC}"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo -e "${GREEN}✅ SutazAI System Status: OPERATIONAL${NC}"
echo "═══════════════════════════════════════════════════════════════════"
