#!/bin/bash
# Comprehensive Compliance Checker for SutazaiApp
# Evaluates project compliance against 20 Professional Codebase Standards

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

BASE_DIR="/opt/sutazaiapp"
SCORE=0
TOTAL_CHECKS=20
PASSED_CHECKS=0

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}   SUTAZAIAPP COMPLIANCE CHECKER v2.0${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Function to check and report
check_item() {
    local check_name="$1"
    local check_command="$2"
    local expected="$3"
    
    if eval "$check_command"; then
        echo -e "${GREEN}✅ $check_name${NC}"
        ((PASSED_CHECKS++))
        return 0
    else
        echo -e "${RED}❌ $check_name${NC}"
        return 1
    fi
}

# 1. Check CHANGELOG.md in all directories
echo -e "${YELLOW}Checking CHANGELOG.md files...${NC}"
DIRS_WITH_CHANGELOG=$(find "$BASE_DIR" -type f -name "CHANGELOG.md" | wc -l)
TOTAL_DIRS=$(find "$BASE_DIR" -type d -not -path "*/.*" -not -path "*/node_modules/*" -not -path "*/venv/*" | wc -l)
if [ "$DIRS_WITH_CHANGELOG" -ge "$((TOTAL_DIRS * 90 / 100))" ]; then
    echo -e "${GREEN}✅ CHANGELOG.md coverage: $DIRS_WITH_CHANGELOG/$TOTAL_DIRS directories${NC}"
    ((PASSED_CHECKS++))
else
    echo -e "${RED}❌ CHANGELOG.md coverage: $DIRS_WITH_CHANGELOG/$TOTAL_DIRS directories${NC}"
fi

# 2. Check Docker service health
echo -e "${YELLOW}Checking Docker service health...${NC}"
UNHEALTHY=$(docker ps --format '{{.Names}}\t{{.Status}}' | grep -c "unhealthy" || true)
if [ "$UNHEALTHY" -eq 0 ]; then
    echo -e "${GREEN}✅ All Docker services healthy${NC}"
    ((PASSED_CHECKS++))
else
    echo -e "${RED}❌ $UNHEALTHY unhealthy Docker services${NC}"
    docker ps --format '{{.Names}}\t{{.Status}}' | grep "unhealthy"
fi

# 3. Check network configuration
echo -e "${YELLOW}Checking network configuration...${NC}"
check_item "No IP conflicts in Docker compose files" \
    "! grep -r 'ipv4_address' $BASE_DIR/docker-compose*.yml 2>/dev/null | awk -F: '{print \$2}' | sort | uniq -d | grep -q ."

# 4. Check /IMPORTANT/diagrams directory
echo -e "${YELLOW}Checking /IMPORTANT/diagrams directory...${NC}"
check_item "/IMPORTANT/diagrams exists with content" \
    "[ -d '$BASE_DIR/IMPORTANT/diagrams' ] && [ \$(ls -1 $BASE_DIR/IMPORTANT/diagrams/*.md 2>/dev/null | wc -l) -gt 0 ]"

# 5. Check script organization
echo -e "${YELLOW}Checking script organization...${NC}"
SCRIPTS_IN_ROOT=$(find "$BASE_DIR/scripts" -maxdepth 1 -type f \( -name "*.sh" -o -name "*.py" \) | wc -l)
if [ "$SCRIPTS_IN_ROOT" -eq 0 ]; then
    echo -e "${GREEN}✅ Scripts properly organized in subdirectories${NC}"
    ((PASSED_CHECKS++))
else
    echo -e "${RED}❌ $SCRIPTS_IN_ROOT scripts still in root scripts directory${NC}"
fi

# 6. Check for README.md
echo -e "${YELLOW}Checking documentation...${NC}"
check_item "README.md exists" "[ -f '$BASE_DIR/README.md' ]"
check_item "CLAUDE.md exists" "[ -f '$BASE_DIR/CLAUDE.md' ]"
check_item ".env.example exists" "[ -f '$BASE_DIR/.env.example' ]"

# 7. Check Git configuration
echo -e "${YELLOW}Checking Git configuration...${NC}"
check_item ".gitignore exists" "[ -f '$BASE_DIR/.gitignore' ]"

# 8. Check testing infrastructure
echo -e "${YELLOW}Checking testing infrastructure...${NC}"
TEST_DIRS=0
[ -d "$BASE_DIR/backend/tests" ] && ((TEST_DIRS++))
[ -d "$BASE_DIR/frontend/tests" ] && ((TEST_DIRS++))
[ -d "$BASE_DIR/agents/tests" ] && ((TEST_DIRS++))
if [ "$TEST_DIRS" -ge 2 ]; then
    echo -e "${GREEN}✅ Test infrastructure: $TEST_DIRS test directories${NC}"
    ((PASSED_CHECKS++))
else
    echo -e "${YELLOW}⚠️  Test infrastructure: $TEST_DIRS test directories${NC}"
fi

# 9. Check monitoring scripts
echo -e "${YELLOW}Checking monitoring infrastructure...${NC}"
MONITORING_SCRIPTS=$(ls -1 "$BASE_DIR/scripts/monitoring/"*.sh 2>/dev/null | wc -l)
if [ "$MONITORING_SCRIPTS" -ge 3 ]; then
    echo -e "${GREEN}✅ Monitoring infrastructure: $MONITORING_SCRIPTS scripts${NC}"
    ((PASSED_CHECKS++))
else
    echo -e "${YELLOW}⚠️  Monitoring infrastructure: $MONITORING_SCRIPTS scripts${NC}"
fi

# 10. Check deployment scripts
echo -e "${YELLOW}Checking deployment scripts...${NC}"
check_item "Deployment scripts exist" \
    "[ -f '$BASE_DIR/deploy.sh' ] || [ -f '$BASE_DIR/scripts/deploy/start-infrastructure.sh' ]"

# 11. Check environment configuration
echo -e "${YELLOW}Checking environment configuration...${NC}"
check_item ".env file exists" "[ -f '$BASE_DIR/.env' ]"

# 12. Check backup infrastructure
echo -e "${YELLOW}Checking backup infrastructure...${NC}"
if [ -d "$BASE_DIR/backups" ] || [ -f "$BASE_DIR/scripts/maintenance/backup.sh" ]; then
    echo -e "${GREEN}✅ Backup infrastructure exists${NC}"
    ((PASSED_CHECKS++))
else
    echo -e "${YELLOW}⚠️  No backup infrastructure${NC}"
fi

# 13. Check MCP server configuration
echo -e "${YELLOW}Checking MCP server configuration...${NC}"
MCP_WRAPPERS=$(ls -1 "$BASE_DIR/scripts/mcp/wrappers/"*.sh 2>/dev/null | wc -l)
if [ "$MCP_WRAPPERS" -ge 10 ]; then
    echo -e "${GREEN}✅ MCP infrastructure: $MCP_WRAPPERS wrappers${NC}"
    ((PASSED_CHECKS++))
else
    echo -e "${YELLOW}⚠️  MCP infrastructure: $MCP_WRAPPERS wrappers${NC}"
fi

# 14. Check code quality tools
echo -e "${YELLOW}Checking code quality tools...${NC}"
if [ -f "$BASE_DIR/.pre-commit-config.yaml" ] || [ -f "$BASE_DIR/pyproject.toml" ]; then
    echo -e "${GREEN}✅ Code quality configuration exists${NC}"
    ((PASSED_CHECKS++))
else
    echo -e "${YELLOW}⚠️  No code quality configuration${NC}"
fi

# 15. Check CI/CD configuration
echo -e "${YELLOW}Checking CI/CD configuration...${NC}"
if [ -d "$BASE_DIR/.github/workflows" ] || [ -f "$BASE_DIR/.gitlab-ci.yml" ]; then
    echo -e "${GREEN}✅ CI/CD configuration exists${NC}"
    ((PASSED_CHECKS++))
else
    echo -e "${YELLOW}⚠️  No CI/CD configuration${NC}"
fi

# Calculate final score
SCORE=$((PASSED_CHECKS * 100 / TOTAL_CHECKS))

# Display summary
echo ""
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}                  SUMMARY${NC}"
echo -e "${BLUE}================================================${NC}"
echo -e "Checks Passed: ${GREEN}$PASSED_CHECKS${NC} / $TOTAL_CHECKS"
echo -e "Compliance Score: ${GREEN}$SCORE%${NC}"
echo ""

# Determine status
if [ "$SCORE" -ge 90 ]; then
    echo -e "${GREEN}🎉 EXCELLENT! Project meets professional standards.${NC}"
    exit 0
elif [ "$SCORE" -ge 70 ]; then
    echo -e "${YELLOW}⚠️  GOOD, but improvements needed to reach excellence.${NC}"
    exit 1
else
    echo -e "${RED}❌ CRITICAL: Major compliance issues detected!${NC}"
    echo -e "${RED}Run './scripts/maintenance/fix-compliance-violations.py' to auto-fix issues.${NC}"
    exit 2
fi