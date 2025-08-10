#!/bin/bash
# Pre-Consolidation System Validation
# Ensures we know exactly what's working before making changes

set -euo pipefail

echo "=== PRE-CONSOLIDATION SYSTEM VALIDATION ==="
echo "Date: $(date)"
echo ""

# Check critical services
echo "=== CRITICAL SERVICE CHECK ==="
services=(
    "Backend:10010:/health"
    "Frontend:10011:/"
    "Ollama:10104:/api/tags"
    "PostgreSQL:10000:"
    "Redis:10001:"
)

working_services=0
total_services=0

for service in "${services[@]}"; do
    IFS=':' read -r name port endpoint <<< "$service"
    total_services=$((total_services + 1))
    
    if [ -n "$endpoint" ]; then
        if timeout 2 curl -s "http://localhost:$port$endpoint" > /dev/null 2>&1; then
            echo "✓ $name is operational"
            working_services=$((working_services + 1))
        else
            echo "✗ $name is not responding"
        fi
    else
        if timeout 2 nc -z localhost "$port" 2>/dev/null; then
            echo "✓ $name port is open"
            working_services=$((working_services + 1))
        else
            echo "✗ $name port is closed"
        fi
    fi
done

echo ""
echo "=== SUMMARY ==="
echo "Working services: $working_services/$total_services"

# Count scripts
echo ""
echo "=== SCRIPT COUNT ==="
echo "Shell scripts: $(find /opt/sutazaiapp -name "*.sh" | grep -v node_modules | grep -v archive | wc -l)"
echo "Python scripts: $(find /opt/sutazaiapp -name "*.py" | grep -v node_modules | grep -v archive | grep -v __pycache__ | wc -l)"
echo "JavaScript files: $(find /opt/sutazaiapp -name "*.js" | grep -v node_modules | grep -v archive | wc -l)"

echo ""
echo "=== RECOMMENDATION ==="
if [ "$working_services" -ge 3 ]; then
    echo "✓ System is stable enough for consolidation"
    echo "  Proceed with: python3 /opt/sutazaiapp/scripts/maintenance/ultra-script-consolidation.py --dry-run"
else
    echo "⚠ System has issues - fix before consolidation"
    echo "  Only $working_services/$total_services services are working"
fi