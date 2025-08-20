#!/bin/bash
# Resource Usage Fix Script
# Created: 2025-08-20
# Purpose: Fix identified resource usage issues in sutazaiapp

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

LOG_FILE="/var/log/sutazaiapp_resource_fix_$(date +%Y%m%d_%H%M%S).log"

# Logging function
log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

log "${BLUE}========================================${NC}"
log "${BLUE}SUTAZAIAPP RESOURCE ISSUE FIXES${NC}"
log "${BLUE}========================================${NC}"
log "Timestamp: $(date '+%Y-%m-%d %H:%M:%S UTC')"
log "Log file: $LOG_FILE"
log ""

# Function to clean zombie processes
clean_zombies() {
    log "${YELLOW}[1] CLEANING ZOMBIE PROCESSES${NC}"
    log "----------------------------------------"
    
    # Get parent PIDs of zombie processes
    ZOMBIE_PARENTS=$(ps aux | grep defunct | awk '{print $3}' | sort -u)
    
    if [ -z "$ZOMBIE_PARENTS" ]; then
        log "No zombie processes found."
    else
        for PARENT_PID in $ZOMBIE_PARENTS; do
            if [ "$PARENT_PID" != "PPID" ] && [ "$PARENT_PID" -gt 1 ]; then
                log "Sending SIGCHLD to parent process $PARENT_PID"
                kill -SIGCHLD "$PARENT_PID" 2>/dev/null || true
            fi
        done
        
        sleep 2
        
        # Check if zombies still exist
        REMAINING_ZOMBIES=$(ps aux | grep defunct | wc -l)
        if [ "$REMAINING_ZOMBIES" -gt 0 ]; then
            log "${YELLOW}Warning: $REMAINING_ZOMBIES zombie processes remain. May need to restart parent processes.${NC}"
            
            # Identify specific MCP zombie processes
            ps aux | grep defunct | grep mcp | while read line; do
                ZPID=$(echo $line | awk '{print $2}')
                ZPPID=$(echo $line | awk '{print $3}')
                ZNAME=$(echo $line | awk '{print $11}')
                log "Zombie: $ZNAME (PID: $ZPID, Parent: $ZPPID)"
            done
        else
            log "${GREEN}All zombie processes cleaned successfully.${NC}"
        fi
    fi
    log ""
}

# Function to deduplicate MCP processes
deduplicate_mcp() {
    log "${YELLOW}[2] DEDUPLICATING MCP PROCESSES${NC}"
    log "----------------------------------------"
    
    # Find duplicate MCP server processes
    MCP_PROCESSES=$(ps aux | grep -E "mcp-server|mcp-.*server|mcp-compass|mcp-knowledge" | grep -v grep | grep -v "cleanup_containers")
    
    # Group by command and keep only one
    declare -A seen_commands
    echo "$MCP_PROCESSES" | while read line; do
        if [ -n "$line" ]; then
            PROC_PID=$(echo $line | awk '{print $2}')
            CMD=$(echo $line | awk '{for(i=11;i<=NF;i++) printf "%s ", $i}')
            
            # Normalize command for comparison
            CMD_KEY=$(echo "$CMD" | sed 's/[0-9]*//g' | tr -d ' ')
            
            if [ -z "${seen_commands[$CMD_KEY]:-}" ]; then
                seen_commands[$CMD_KEY]=$PROC_PID
                log "Keeping MCP process: PID $PROC_PID"
            else
                log "Killing duplicate MCP process: PID $PROC_PID"
                kill -TERM "$PROC_PID" 2>/dev/null || true
            fi
        fi
    done
    log ""
}

# Function to optimize Docker containers
optimize_containers() {
    log "${YELLOW}[3] OPTIMIZING DOCKER CONTAINERS${NC}"
    log "----------------------------------------"
    
    # Apply memory limits to high-usage containers
    log "Applying resource limits to containers..."
    
    # MCP Orchestrator - Main issue (using 458MB)
    if docker ps | grep -q sutazai-mcp-orchestrator; then
        log "Setting memory limit for sutazai-mcp-orchestrator to 256MB"
        docker update --memory="256m" --memory-swap="256m" sutazai-mcp-orchestrator 2>/dev/null || \
            log "${YELLOW}Warning: Could not update mcp-orchestrator limits${NC}"
    fi
    
    # Neo4j - Using 600MB
    if docker ps | grep -q sutazai-neo4j; then
        log "Optimizing Neo4j memory settings"
        docker update --memory="512m" --memory-swap="512m" sutazai-neo4j 2>/dev/null || \
            log "${YELLOW}Warning: Could not update neo4j limits${NC}"
    fi
    
    # Ollama - Using 662MB
    if docker ps | grep -q sutazai-ollama; then
        log "Setting memory limit for Ollama to 512MB"
        docker update --memory="512m" --memory-swap="512m" sutazai-ollama 2>/dev/null || \
            log "${YELLOW}Warning: Could not update ollama limits${NC}"
    fi
    
    # Clean up stopped containers
    log "Removing stopped containers..."
    docker container prune -f 2>/dev/null || true
    
    # Clean up unused images
    log "Removing unused Docker images..."
    docker image prune -f 2>/dev/null || true
    
    # Clean up unused volumes
    log "Removing unused Docker volumes..."
    docker volume prune -f 2>/dev/null || true
    
    log ""
}

# Function to restart unhealthy containers
fix_unhealthy_containers() {
    log "${YELLOW}[4] FIXING UNHEALTHY CONTAINERS${NC}"
    log "----------------------------------------"
    
    UNHEALTHY=$(docker ps --format "{{.Names}}" --filter "health=unhealthy")
    
    if [ -z "$UNHEALTHY" ]; then
        log "No unhealthy containers found."
    else
        for CONTAINER in $UNHEALTHY; do
            log "Restarting unhealthy container: $CONTAINER"
            docker restart "$CONTAINER" 2>/dev/null || log "${YELLOW}Warning: Could not restart $CONTAINER${NC}"
        done
    fi
    log ""
}

# Function to clean up orphaned processes
cleanup_orphaned() {
    log "${YELLOW}[5] CLEANING ORPHANED PROCESSES${NC}"
    log "----------------------------------------"
    
    # Kill orphaned npm processes
    ORPHANED_NPM=$(ps aux | grep "npm exec" | grep -E "defunct|<defunct>" | awk '{print $2}')
    if [ -n "$ORPHANED_NPM" ]; then
        for ORPHAN_PID in $ORPHANED_NPM; do
            log "Killing orphaned npm process: $ORPHAN_PID"
            kill -9 "$ORPHAN_PID" 2>/dev/null || true
        done
    fi
    
    # Clean up old MCP wrapper processes
    OLD_MCP=$(ps aux | grep "/scripts/mcp/wrappers" | grep -v grep | awk '{print $2}')
    if [ -n "$OLD_MCP" ]; then
        for OLD_PID in $OLD_MCP; do
            UPTIME=$(ps -o etimes= -p "$OLD_PID" 2>/dev/null | tr -d ' ')
            if [ -n "$UPTIME" ] && [ "$UPTIME" -gt 86400 ]; then  # Older than 24 hours
                log "Killing old MCP wrapper process: $OLD_PID (age: ${UPTIME}s)"
                kill -TERM "$OLD_PID" 2>/dev/null || true
            fi
        done
    fi
    log ""
}

# Function to apply system optimizations
system_optimizations() {
    log "${YELLOW}[6] APPLYING SYSTEM OPTIMIZATIONS${NC}"
    log "----------------------------------------"
    
    # Clear system caches
    log "Clearing system caches..."
    sync
    echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || log "${YELLOW}Warning: Could not clear caches (requires root)${NC}"
    
    # Optimize swappiness for container workloads
    log "Setting swappiness to 10 (optimized for containers)..."
    echo 10 > /proc/sys/vm/swappiness 2>/dev/null || log "${YELLOW}Warning: Could not set swappiness (requires root)${NC}"
    
    # Set OOM killer adjustments for critical services
    for SERVICE in dockerd containerd systemd; do
        SERVICE_PID=$(pgrep -x "$SERVICE" | head -1)
        if [ -n "$SERVICE_PID" ]; then
            echo -1000 > /proc/$SERVICE_PID/oom_score_adj 2>/dev/null || true
            log "Protected $SERVICE from OOM killer"
        fi
    done
    
    log ""
}

# Function to create monitoring configuration
create_monitoring_config() {
    log "${YELLOW}[7] CREATING MONITORING CONFIGURATION${NC}"
    log "----------------------------------------"
    
    cat > /opt/sutazaiapp/config/resource_limits.yaml << 'EOF'
# Resource Limits Configuration
# Generated: 2025-08-20

services:
  mcp_orchestrator:
    memory_limit: 256m
    cpu_limit: 0.5
    restart_policy: unless-stopped
    healthcheck:
      interval: 30s
      timeout: 10s
      retries: 3

  neo4j:
    memory_limit: 512m
    heap_size: 256m
    pagecache_size: 128m
    
  ollama:
    memory_limit: 512m
    model_memory: 256m
    
  prometheus:
    retention_days: 7
    scrape_interval: 30s
    
  grafana:
    memory_limit: 128m
    
monitoring:
  alerts:
    memory_threshold: 80
    cpu_threshold: 70
    zombie_threshold: 10
    
cleanup:
  zombie_check_interval: 300  # 5 minutes
  container_prune_interval: 3600  # 1 hour
  log_rotation_days: 7
EOF
    
    log "Created resource limits configuration at /opt/sutazaiapp/config/resource_limits.yaml"
    log ""
}

# Main execution
main() {
    log "${BLUE}Starting resource optimization...${NC}"
    log ""
    
    # Record initial state
    INITIAL_MEM=$(free -m | awk '/^Mem:/{print $3}')
    INITIAL_ZOMBIES=$(ps aux | grep defunct | wc -l)
    
    # Execute fixes
    clean_zombies
    deduplicate_mcp
    optimize_containers
    fix_unhealthy_containers
    cleanup_orphaned
    system_optimizations
    create_monitoring_config
    
    # Record final state
    FINAL_MEM=$(free -m | awk '/^Mem:/{print $3}')
    FINAL_ZOMBIES=$(ps aux | grep defunct | wc -l)
    
    # Calculate improvements
    MEM_SAVED=$((INITIAL_MEM - FINAL_MEM))
    ZOMBIES_CLEANED=$((INITIAL_ZOMBIES - FINAL_ZOMBIES))
    
    log "${GREEN}========================================${NC}"
    log "${GREEN}OPTIMIZATION COMPLETE${NC}"
    log "${GREEN}========================================${NC}"
    log "Memory saved: ${MEM_SAVED}MB"
    log "Zombies cleaned: ${ZOMBIES_CLEANED}"
    log "Final memory usage: ${FINAL_MEM}MB"
    log "Final zombie count: ${FINAL_ZOMBIES}"
    log ""
    log "Full log saved to: $LOG_FILE"
    
    # Provide next steps
    log ""
    log "${BLUE}RECOMMENDED NEXT STEPS:${NC}"
    log "1. Monitor resource usage: watch -n 5 'docker stats --no-stream'"
    log "2. Check for new zombies: ps aux | grep defunct"
    log "3. Review container logs: docker logs sutazai-mcp-orchestrator"
    log "4. Set up automated monitoring: crontab -e"
    log "   Add: */5 * * * * /opt/sutazaiapp/scripts/monitoring/diagnose_resource_usage.sh"
}

# Run main function
main