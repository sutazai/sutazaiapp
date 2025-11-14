#!/bin/bash
# Quick verification before Portainer migration

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║       SUTAZAI PLATFORM - PRE-MIGRATION VERIFICATION       ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

CHECKS_PASSED=0
CHECKS_FAILED=0

check() {
    if eval "$2" > /dev/null 2>&1; then
        echo "✅ $1"
        ((CHECKS_PASSED++))
        return 0
    else
        echo "❌ $1"
        ((CHECKS_FAILED++))
        return 1
    fi
}

echo "🔍 Checking prerequisites..."
echo ""

check "Docker installed" "command -v docker"
check "Portainer running" "sudo docker ps | grep -q portainer"
check "Network exists" "sudo docker network inspect sutazaiapp_sutazai-network"
check "Ollama running" "curl -s http://localhost:11434/api/version"

echo ""
echo "📦 Checking containers (expecting 11)..."
echo ""

CONTAINER_COUNT=$(sudo docker ps --filter "name=sutazai-" --format "{{.Names}}" | wc -l)
check "Container count ($CONTAINER_COUNT/11)" "[ $CONTAINER_COUNT -eq 11 ]"

HEALTHY_COUNT=$(sudo docker ps --filter "name=sutazai-" --filter "health=healthy" --format "{{.Names}}" | wc -l)
check "Healthy containers ($HEALTHY_COUNT)" "[ $HEALTHY_COUNT -ge 9 ]"

echo ""
echo "🌐 Checking service endpoints..."
echo ""

check "Frontend (11000)" "curl -s http://localhost:11000/_stcore/health"
check "Backend (10200)" "curl -s http://localhost:10200/health"
check "PostgreSQL (10000)" "sudo docker exec sutazai-postgres pg_isready -U jarvis"
check "Redis (10001)" "sudo docker exec sutazai-redis redis-cli ping"
check "Neo4j (10002)" "curl -s http://localhost:10002"
check "RabbitMQ (10005)" "curl -s http://localhost:10005"
check "Consul (10006)" "curl -s http://localhost:10006/v1/status/leader"
check "Kong (10009)" "curl -s http://localhost:10009/status"

echo ""
echo "📄 Checking migration files..."
echo ""

check "Migration script" "[ -f /opt/sutazaiapp/migrate-to-portainer.sh ]"
check "Stack config" "[ -f /opt/sutazaiapp/docker-compose-portainer.yml ]"
check "Quick start guide" "[ -f /opt/sutazaiapp/PORTAINER_QUICKSTART.md ]"
check "Deployment guide" "[ -f /opt/sutazaiapp/PORTAINER_DEPLOYMENT_GUIDE.md ]"
check "Validation report" "[ -f /opt/sutazaiapp/PRODUCTION_VALIDATION_REPORT.md ]"
check "Migration summary" "[ -f /opt/sutazaiapp/PORTAINER_MIGRATION_SUMMARY.md ]"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ $CHECKS_FAILED -eq 0 ]; then
    echo "✅ ALL CHECKS PASSED ($CHECKS_PASSED/$CHECKS_PASSED)"
    echo ""
    echo "🚀 SYSTEM IS READY FOR PORTAINER MIGRATION"
    echo ""
    echo "Next steps:"
    echo "  1. Run: ./migrate-to-portainer.sh"
    echo "  2. Or: See PORTAINER_QUICKSTART.md for manual steps"
    echo ""
    exit 0
else
    echo "⚠️  SOME CHECKS FAILED ($CHECKS_FAILED failures, $CHECKS_PASSED passed)"
    echo ""
    echo "Please resolve the failed checks before proceeding with migration."
    echo ""
    exit 1
fi
