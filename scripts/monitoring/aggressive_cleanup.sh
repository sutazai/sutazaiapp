#!/bin/bash
# Aggressive Resource Cleanup Script
# Created: 2025-08-20
# Purpose: Forcefully clean up zombie processes and excessive MCP instances

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}AGGRESSIVE RESOURCE CLEANUP${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S UTC')"
echo ""

# 1. Kill all Claude processes except the current one
echo -e "${YELLOW}[1] Cleaning up excessive Claude instances${NC}"
CURRENT_CLAUDE_PID=$(pgrep -f "claude" | head -1)
CLAUDE_PIDS=$(pgrep -f "claude" | grep -v "^$CURRENT_CLAUDE_PID$" || true)
if [ -n "$CLAUDE_PIDS" ]; then
    echo "Killing duplicate Claude processes..."
    echo "$CLAUDE_PIDS" | while read cpid; do
        if [ -n "$cpid" ] && [ "$cpid" != "$CURRENT_CLAUDE_PID" ]; then
            echo "  Killing Claude process: $cpid"
            kill -TERM "$cpid" 2>/dev/null || true
        fi
    done
else
    echo "No duplicate Claude processes found"
fi
echo ""

# 2. Clean all zombie processes forcefully
echo -e "${YELLOW}[2] Forcefully cleaning zombie processes${NC}"
# Get all zombie process PIDs and their parents
zombies=$(ps aux | grep defunct | awk '{print $2":"$3}' | grep -v "PID:PPID")
zombie_count=$(echo "$zombies" | wc -l)

if [ "$zombie_count" -gt 0 ]; then
    echo "Found $zombie_count zombie processes"
    
    # Kill parent processes of zombies
    echo "$zombies" | while IFS=: read zpid zppid; do
        if [ -n "$zppid" ] && [ "$zppid" != "0" ] && [ "$zppid" != "0.0" ]; then
            # Try to clean up the parent
            echo "  Sending SIGCHLD to parent $zppid of zombie $zpid"
            kill -SIGCHLD "$zppid" 2>/dev/null || true
        fi
    done
    
    sleep 2
    
    # If zombies still exist, kill their parents more aggressively
    remaining=$(ps aux | grep defunct | wc -l)
    if [ "$remaining" -gt 0 ]; then
        echo "Still $remaining zombies, killing parent processes..."
        ps aux | grep defunct | awk '{print $3}' | sort -u | while read parent; do
            if [ "$parent" != "PPID" ] && [ "$parent" != "0" ] && [ "$parent" != "0.0" ] && [ -n "$parent" ]; then
                # Check if parent is a critical system process
                parent_name=$(ps -p "$parent" -o comm= 2>/dev/null || echo "unknown")
                if [[ "$parent_name" != "systemd" ]] && [[ "$parent_name" != "init" ]] && [[ "$parent_name" != "kernel" ]]; then
                    echo "  Killing parent process $parent ($parent_name)"
                    kill -TERM "$parent" 2>/dev/null || true
                fi
            fi
        done
    fi
else
    echo "No zombie processes found"
fi
echo ""

# 3. Clean up all MCP processes except essential ones
echo -e "${YELLOW}[3] Cleaning up excessive MCP processes${NC}"
# Keep only one instance of each MCP service type
declare -A mcp_services
mcp_services["mcp-server-filesystem"]=0
mcp_services["mcp-server-github"]=0
mcp_services["mcp-knowledge-graph"]=0
mcp_services["mcp-compass"]=0
mcp_services["extended_memory_mcp"]=0

ps aux | grep -E "mcp" | grep -v grep | grep -v "cleanup_containers" | while read line; do
    proc_pid=$(echo $line | awk '{print $2}')
    proc_cmd=$(echo $line | awk '{for(i=11;i<=NF;i++) printf "%s ", $i}')
    
    # Identify the service type
    service_type=""
    for service in "${!mcp_services[@]}"; do
        if echo "$proc_cmd" | grep -q "$service"; then
            service_type=$service
            break
        fi
    done
    
    if [ -n "$service_type" ]; then
        if [ "${mcp_services[$service_type]}" -eq 0 ]; then
            echo "  Keeping $service_type (PID: $proc_pid)"
            mcp_services[$service_type]=1
        else
            echo "  Killing duplicate $service_type (PID: $proc_pid)"
            kill -TERM "$proc_pid" 2>/dev/null || true
        fi
    else
        # Unknown MCP process, check if it's old
        proc_age=$(ps -o etimes= -p "$proc_pid" 2>/dev/null | tr -d ' ')
        if [ -n "$proc_age" ] && [ "$proc_age" -gt 3600 ]; then
            echo "  Killing old MCP process (PID: $proc_pid, age: ${proc_age}s)"
            kill -TERM "$proc_pid" 2>/dev/null || true
        fi
    fi
done
echo ""

# 4. Clean up Docker containers
echo -e "${YELLOW}[4] Optimizing Docker containers${NC}"
# Remove stopped containers
docker container prune -f 2>/dev/null || true
echo "Removed stopped containers"

# Clean up volumes
docker volume prune -f 2>/dev/null || true
echo "Removed unused volumes"

# Clean up images
docker image prune -f 2>/dev/null || true
echo "Removed unused images"

# Restart unhealthy containers
unhealthy=$(docker ps --format "{{.Names}}" --filter "health=unhealthy")
if [ -n "$unhealthy" ]; then
    echo "Restarting unhealthy containers:"
    echo "$unhealthy" | while read container; do
        echo "  Restarting $container"
        docker restart "$container" 2>/dev/null || true
    done
fi
echo ""

# 5. Clear system caches
echo -e "${YELLOW}[5] Clearing system caches${NC}"
sync
echo 3 > /proc/sys/vm/drop_caches 2>/dev/null && echo "System caches cleared" || echo "Could not clear caches (requires root)"
echo ""

# 6. Show results
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}CLEANUP COMPLETE${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Current Status:"
echo "Memory: $(free -h | grep Mem | awk '{print "Used: " $3 " / Total: " $2}')"
echo "Swap: $(free -h | grep Swap | awk '{print "Used: " $3 " / Total: " $2}')"
echo "Load Average: $(uptime | awk -F'load average:' '{print $2}')"
echo "Process Count: $(ps aux | wc -l)"
echo "MCP Processes: $(ps aux | grep mcp | grep -v grep | wc -l)"
echo "Zombie Processes: $(ps aux | grep defunct | wc -l)"
echo "Docker Containers: $(docker ps -q | wc -l) running"
echo ""
echo -e "${BLUE}Recommendations:${NC}"
echo "1. Monitor: watch -n 5 'free -h; echo; ps aux | grep defunct | wc -l'"
echo "2. Check logs: journalctl -xe | tail -50"
echo "3. Review container logs: docker logs sutazai-mcp-orchestrator --tail 50"
echo "4. Set up automated monitoring: systemctl enable /opt/sutazaiapp/scripts/monitoring/sutazaiapp-monitor.service"