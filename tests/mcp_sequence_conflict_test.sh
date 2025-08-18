#!/usr/bin/env bash
set -Eeuo pipefail

# MCP Sequential Startup Conflict Detection
# Starts MCPs one by one to identify specific conflict pairs

ROOT="/opt/sutazaiapp"
LOG_DIR="$ROOT/logs/mcp_sequence_test"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SEQUENCE_LOG="$LOG_DIR/sequence_test_$TIMESTAMP.log"

echo "=== MCP Sequential Conflict Testing Started at $(date) ===" | tee "$SEQUENCE_LOG"

MCP_SERVERS=(
    "files"
    "context7"
    "http_fetch"
    "ddg"
    "sequentialthinking"
    "nx-mcp"
    "extended-memory"
    "mcp_ssh"
    "postgres"
    "playwright-mcp"
    "memory-bank-mcp"
    "puppeteer-mcp (no longer in use)"
    "knowledge-graph-mcp"
    "compass-mcp"
    "github"
    "http"
    "language-server"
    "claude-flow"
    "ruv-swarm"
    "claude-task-runner"
)

STARTED_MCPS=()
FAILED_MCPS=()

cleanup() {
    echo "Cleaning up all started MCPs..." | tee -a "$SEQUENCE_LOG"
    for pid in "${STARTED_MCPS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    pkill -f "mcp-" 2>/dev/null || true
    pkill -f "claude-flow" 2>/dev/null || true
    pkill -f "ruv-swarm" 2>/dev/null || true
    docker stop $(docker ps -q --filter "ancestor=mcp/") 2>/dev/null || true
}

start_mcp() {
    local server="$1"
    local attempt="$2"
    
    echo "[$attempt/${#MCP_SERVERS[@]}] Starting $server..." | tee -a "$SEQUENCE_LOG"
    
    case "$server" in
        "claude-flow")
            timeout 15 npx claude-flow@alpha mcp start 2>&1 &
            ;;
        "ruv-swarm")
            timeout 15 npx ruv-swarm@latest mcp start 2>&1 &
            ;;
        *)
            timeout 15 "/opt/sutazaiapp/scripts/mcp/wrappers/${server}.sh" 2>&1 &
            ;;
    esac
    
    local pid=$!
    
    # Wait briefly for startup
    sleep 2
    
    if kill -0 "$pid" 2>/dev/null; then
        echo "✅ $server started successfully (PID: $pid)" | tee -a "$SEQUENCE_LOG"
        STARTED_MCPS+=("$pid")
        
        # Check system state after each startup
        echo "System state after $server:" | tee -a "$SEQUENCE_LOG"
        echo "  Processes: $(ps aux --no-headers | wc -l)" | tee -a "$SEQUENCE_LOG"
        echo "  Memory: $(free -m | awk 'NR==2{printf "%.1f%%", $3*100/$2}')" | tee -a "$SEQUENCE_LOG"
        echo "  Open FDs: $(lsof | wc -l)" | tee -a "$SEQUENCE_LOG"
        
        # Check for port conflicts
        ports_used=$(ss -tulpn | grep -c ":111" || echo "0")
        if [[ $ports_used -gt 1 ]]; then
            echo "⚠️  Port conflict detected after starting $server" | tee -a "$SEQUENCE_LOG"
            ss -tulpn | grep ":111" | tee -a "$SEQUENCE_LOG"
        fi
        
        return 0
    else
        echo "❌ $server failed to start" | tee -a "$SEQUENCE_LOG"
        FAILED_MCPS+=("$server")
        return 1
    fi
}

test_interaction() {
    local new_server="$1"
    
    echo "Testing interaction of $new_server with running MCPs..." | tee -a "$SEQUENCE_LOG"
    
    # Check for stdio conflicts
    echo "Checking stdio stream usage..." | tee -a "$SEQUENCE_LOG"
    local stdio_conflicts=0
    for pid in "${STARTED_MCPS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            local stdio_count=$(lsof -p "$pid" 2>/dev/null | grep -c -E " [012]u " || echo "0")
            if [[ $stdio_count -gt 3 ]]; then
                echo "⚠️  PID $pid has $stdio_count stdio handles" | tee -a "$SEQUENCE_LOG"
                ((stdio_conflicts++))
            fi
        fi
    done
    
    # Check for memory pressure
    local mem_usage=$(free | awk 'NR==2{printf "%.1f", $3*100/$2}')
    if (( $(echo "$mem_usage > 80.0" | bc -l) )); then
        echo "⚠️  High memory usage: ${mem_usage}%" | tee -a "$SEQUENCE_LOG"
    fi
    
    # Check for process interference
    local zombie_count=$(ps aux | awk '$8=="Z" {print $0}' | wc -l)
    if [[ $zombie_count -gt 0 ]]; then
        echo "⚠️  $zombie_count zombie processes detected" | tee -a "$SEQUENCE_LOG"
    fi
    
    return $stdio_conflicts
}

# Main test execution
main() {
    trap cleanup EXIT
    
    for i in "${!MCP_SERVERS[@]}"; do
        server="${MCP_SERVERS[$i]}"
        attempt=$((i + 1))
        
        echo "" | tee -a "$SEQUENCE_LOG"
        echo "=== Testing MCP $attempt/${#MCP_SERVERS[@]}: $server ===" | tee -a "$SEQUENCE_LOG"
        
        if start_mcp "$server" "$attempt"; then
            test_interaction "$server"
            
            # Wait for stabilization
            sleep 1
        else
            echo "Skipping interaction test for failed MCP: $server" | tee -a "$SEQUENCE_LOG"
        fi
        
        # Check if we should continue
        if [[ ${#FAILED_MCPS[@]} -gt 5 ]]; then
            echo "Too many failures, stopping test" | tee -a "$SEQUENCE_LOG"
            break
        fi
    done
    
    echo "" | tee -a "$SEQUENCE_LOG"
    echo "=== FINAL RESULTS ===" | tee -a "$SEQUENCE_LOG"
    echo "Successfully started: $((${#MCP_SERVERS[@]} - ${#FAILED_MCPS[@]}))" | tee -a "$SEQUENCE_LOG"
    echo "Failed to start: ${#FAILED_MCPS[@]}" | tee -a "$SEQUENCE_LOG"
    
    if [[ ${#FAILED_MCPS[@]} -gt 0 ]]; then
        echo "Failed MCPs: ${FAILED_MCPS[*]}" | tee -a "$SEQUENCE_LOG"
    fi
    
    echo "Running MCPs at end of test:" | tee -a "$SEQUENCE_LOG"
    ps aux | grep -E "mcp|claude-flow|ruv-swarm" | grep -v grep | tee -a "$SEQUENCE_LOG" || echo "No MCP processes running"
    
    echo "Full log: $SEQUENCE_LOG"
}

main "$@"