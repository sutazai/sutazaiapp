#!/bin/bash
# Comprehensive Resource Usage Diagnosis Tool
# Created: 2025-08-20
# Purpose: Identify and analyze high CPU/RAM usage in sutazaiapp system

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}SUTAZAIAPP RESOURCE USAGE DIAGNOSIS${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S UTC')"
echo ""

# System Overview
echo -e "${YELLOW}[1] SYSTEM OVERVIEW${NC}"
echo "----------------------------------------"
echo "Uptime and Load Average:"
uptime
echo ""
echo "Memory Usage:"
free -h
echo ""
echo "CPU Count: $(nproc)"
echo ""

# Memory Analysis
echo -e "${YELLOW}[2] TOP MEMORY CONSUMERS${NC}"
echo "----------------------------------------"
echo "Top 10 processes by memory usage:"
ps aux --sort=-%mem | head -11
echo ""

# Docker Container Analysis
echo -e "${YELLOW}[3] DOCKER CONTAINER RESOURCES${NC}"
echo "----------------------------------------"
docker stats --no-stream --format "table {{.Container}}\t{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" | head -20
echo ""

# Zombie Process Analysis
echo -e "${YELLOW}[4] ZOMBIE PROCESS ANALYSIS${NC}"
echo "----------------------------------------"
ZOMBIE_COUNT=$(ps aux | grep defunct | wc -l)
echo "Total zombie processes: $ZOMBIE_COUNT"
if [ $ZOMBIE_COUNT -gt 0 ]; then
    echo -e "${RED}WARNING: High number of zombie processes detected!${NC}"
    echo "Zombie process details:"
    ps aux | grep defunct | head -10
fi
echo ""

# MCP and Claude Process Analysis
echo -e "${YELLOW}[5] MCP AND CLAUDE PROCESSES${NC}"
echo "----------------------------------------"
echo "Active MCP processes:"
ps aux | grep -E "mcp" | grep -v grep | wc -l
ps aux | grep -E "mcp" | grep -v grep | head -20
echo ""
echo "Active Claude processes:"
ps aux | grep -E "claude" | grep -v grep | wc -l
ps aux | grep -E "claude" | grep -v grep | head -10
echo ""

# Duplicate Process Detection
echo -e "${YELLOW}[6] DUPLICATE PROCESS DETECTION${NC}"
echo "----------------------------------------"
echo "Checking for duplicate MCP servers..."
ps aux | grep -E "mcp-server|mcp-.*server" | grep -v grep | awk '{print $11}' | sort | uniq -c | sort -rn | head -10
echo ""

# Port Usage Analysis
echo -e "${YELLOW}[7] PORT USAGE ANALYSIS${NC}"
echo "----------------------------------------"
echo "Active listening ports:"
ss -tulpn | grep LISTEN | head -20
echo ""

# Disk I/O Analysis
echo -e "${YELLOW}[8] DISK I/O ANALYSIS${NC}"
echo "----------------------------------------"
if command -v iostat &> /dev/null; then
    iostat -x 1 2 | tail -n +4
else
    echo "iostat not available. Install sysstat package for disk I/O statistics."
fi
echo ""

# Container Health Check
echo -e "${YELLOW}[9] CONTAINER HEALTH STATUS${NC}"
echo "----------------------------------------"
docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "(unhealthy|starting|restarting)" || echo "All containers healthy"
echo ""

# Resource Recommendations
echo -e "${YELLOW}[10] IMMEDIATE RECOMMENDATIONS${NC}"
echo "----------------------------------------"
if [ $ZOMBIE_COUNT -gt 10 ]; then
    echo -e "${RED}• Critical: Clean up zombie processes immediately${NC}"
fi

# Check for high memory containers
HIGH_MEM_CONTAINERS=$(docker stats --no-stream --format "{{.Name}}\t{{.MemPerc}}" | awk '$2 > 10 {print $1}' | wc -l)
if [ $HIGH_MEM_CONTAINERS -gt 0 ]; then
    echo -e "${RED}• Warning: Containers using >10% memory detected${NC}"
    docker stats --no-stream --format "{{.Name}}\t{{.MemPerc}}" | awk '$2 > 10'
fi

# Check for duplicate MCP processes
MCP_COUNT=$(ps aux | grep -E "mcp-server" | grep -v grep | wc -l)
if [ $MCP_COUNT -gt 20 ]; then
    echo -e "${RED}• Warning: Too many MCP server processes ($MCP_COUNT)${NC}"
fi

echo ""
echo -e "${GREEN}Diagnosis complete. Check logs for detailed analysis.${NC}"