#!/bin/bash
# Configuration Usage Analysis Script
# Helps identify which configuration files are actually being used

set -euo pipefail

echo "=== Configuration Usage Analysis ==="
echo "Date: $(date)"
echo ""

echo "=== Currently Running Containers ==="
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Image}}"
echo ""

echo "=== Docker Compose Files in Use ==="
# Check which docker-compose file was used to start containers
if docker ps -q | head -1 > /dev/null 2>&1; then
    container_id=$(docker ps -q | head -1)
    if [ ! -z "$container_id" ]; then
        echo "Checking container labels for compose file info..."
        docker inspect $container_id | grep -A5 "com.docker.compose" | grep -E "project.config_files|project.working_dir" || echo "No compose labels found"
    fi
fi
echo ""

echo "=== Agent Configuration Files ==="
echo "Checking which agent configs are referenced in running backend..."
if docker ps | grep -q backend; then
    echo "Backend container is running"
    # Check backend logs for config loading
    docker logs sutazai-backend 2>&1 | grep -i "registry\|config" | head -10 || echo "No config loading messages found"
else
    echo "Backend container not running"
fi
echo ""

echo "=== File Access Times (Recently Used Configs) ==="
echo "Agent configs accessed in last 24 hours:"
find /opt/sutazaiapp -name "*.json" -path "*/config/*" -o -path "*/agents/*" -atime -1 2>/dev/null | head -20
echo ""

echo "=== Active Backend Configuration ==="
if [ -f /opt/sutazaiapp/backend/app/core/config.py ]; then
    echo "Main config file exists: /opt/sutazaiapp/backend/app/core/config.py"
    echo "Checking for config imports in backend..."
    grep -r "from.*config import\|import.*config" /opt/sutazaiapp/backend/app --include="*.py" | grep -v __pycache__ | head -10
fi
echo ""

echo "=== Configuration File Sizes (Empty Files) ==="
echo "Empty or nearly empty config files:"
find /opt/sutazaiapp -name "*.json" -o -name "*.yaml" -o -name "*.yml" | while read f; do
    size=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f" 2>/dev/null || echo "unknown")
    if [ "$size" -lt 10 ]; then
        echo "  EMPTY: $f (${size} bytes)"
    fi
done
echo ""

echo "=== Duplicate Config Names ==="
echo "Files with same name in different locations:"
find /opt/sutazaiapp -type f \( -name "*.json" -o -name "*.yaml" -o -name "*.yml" \) | xargs -I {} basename {} | sort | uniq -d | while read name; do
    echo "  Duplicate: $name"
    find /opt/sutazaiapp -name "$name" -type f | sed 's/^/    - /'
done
echo ""

echo "=== Summary ==="
echo "Total JSON configs: $(find /opt/sutazaiapp -name "*.json" | wc -l)"
echo "Total YAML configs: $(find /opt/sutazaiapp -name "*.yaml" -o -name "*.yml" | wc -l)"
echo "Total docker-compose files: $(find /opt/sutazaiapp -name "docker-compose*.yml" | wc -l)"
echo "Total requirements.txt: $(find /opt/sutazaiapp -name "requirements*.txt" | wc -l)"
echo ""
echo "Analysis complete. Review output above to identify unused configurations."