#!/usr/bin/env bash
set -Eeuo pipefail

# Comprehensive MCP Runtime Conflict Testing Script
# Tests simultaneous startup of all 21 MCPs with real-time monitoring

ROOT="/opt/sutazaiapp"
LOG_DIR="$ROOT/logs/mcp_conflict_test"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TEST_LOG="$LOG_DIR/mcp_conflict_test_$TIMESTAMP.log"
RESOURCE_LOG="$LOG_DIR/resource_monitor_$TIMESTAMP.log"
ERROR_LOG="$LOG_DIR/errors_$TIMESTAMP.log"
PROCESS_LOG="$LOG_DIR/processes_$TIMESTAMP.log"

echo "=== MCP Runtime Conflict Testing Started at $(date) ===" | tee "$TEST_LOG"

# MCP Servers from .mcp.json
MCP_SERVERS=(
    "claude-flow"
    "ruv-swarm" 
    "files"
    "context7"
    "http_fetch"
    "ddg"
    "sequentialthinking"
    "nx-mcp"
    "extended-memory"
    "mcp_ssh"
    "ultimatecoder"
    "postgres"
    "playwright-mcp"
    "memory-bank-mcp"
    "puppeteer-mcp (no longer in use)"
    "knowledge-graph-mcp"
    "compass-mcp"
    "github"
    "http"
    "language-server"
    "claude-task-runner"
)

# Function to monitor system resources
monitor_resources() {
    while true; do
        echo "$(date): $(ps aux --no-headers | wc -l) processes, $(free -m | awk 'NR==2{printf "%.2f%%", $3*100/$2}') memory used" >> "$RESOURCE_LOG"
        echo "$(date): Open file descriptors: $(lsof | wc -l)" >> "$RESOURCE_LOG"
        echo "$(date): TCP connections: $(ss -t state connected | wc -l)" >> "$RESOURCE_LOG"
        sleep 1
    done &
    MONITOR_PID=$!
}

# Function to kill all MCP processes
cleanup_mcps() {
    echo "Cleaning up existing MCP processes..." | tee -a "$TEST_LOG"
    pkill -f "claude-flow" 2>/dev/null || true
    pkill -f "ruv-swarm" 2>/dev/null || true
    pkill -f "mcp-server" 2>/dev/null || true
    pkill -f "mcp-" 2>/dev/null || true
    pkill -f "nx-mcp" 2>/dev/null || true
    pkill -f "extended_memory" 2>/dev/null || true
    pkill -f "memory-bank-mcp" 2>/dev/null || true
    pkill -f "knowledge-graph" 2>/dev/null || true
    pkill -f "puppeteer-mcp (no longer in use)" 2>/dev/null || true
    pkill -f "context7-mcp" 2>/dev/null || true
    
    # Stop any docker containers
    docker stop $(docker ps -q --filter "ancestor=mcp/") 2>/dev/null || true
    
    sleep 3
    echo "Cleanup complete. Remaining MCP processes:" | tee -a "$TEST_LOG"
    ps aux | grep -E "mcp|claude-flow|ruv-swarm" | grep -v grep | tee -a "$PROCESS_LOG" || echo "No MCP processes found"
}

# Function to start all MCPs simultaneously
start_all_mcps_simultaneously() {
    echo "Starting all ${#MCP_SERVERS[@]} MCPs simultaneously..." | tee -a "$TEST_LOG"
    
    # Start resource monitoring
    monitor_resources
    
    # Record initial state
    echo "Initial system state:" | tee -a "$TEST_LOG"
    ps aux --no-headers | wc -l | xargs echo "Initial process count:" | tee -a "$TEST_LOG"
    free -m | tee -a "$TEST_LOG"
    
    # Start all MCPs in background
    for server in "${MCP_SERVERS[@]}"; do
        echo "Starting $server..." | tee -a "$TEST_LOG"
        case "$server" in
            "claude-flow")
                timeout 30 npx claude-flow@alpha mcp start 2>>"$ERROR_LOG" &
                ;;
            "ruv-swarm")
                timeout 30 npx ruv-swarm@latest mcp start 2>>"$ERROR_LOG" &
                ;;
            *)
                timeout 30 "/opt/sutazaiapp/scripts/mcp/wrappers/${server}.sh" 2>>"$ERROR_LOG" &
                ;;
        esac
        MCP_PIDS+=($!)
        echo "Started $server with PID ${MCP_PIDS[-1]}" | tee -a "$TEST_LOG"
        
        # Brief pause to detect immediate conflicts
        sleep 0.1
    done
    
    echo "All MCPs started. Monitoring for conflicts..." | tee -a "$TEST_LOG"
}

# Function to analyze conflicts
analyze_conflicts() {
    echo "=== CONFLICT ANALYSIS ===" | tee -a "$TEST_LOG"
    
    # Check for failed processes
    failed_count=0
    for i in "${!MCP_PIDS[@]}"; do
        pid=${MCP_PIDS[$i]}
        server=${MCP_SERVERS[$i]}
        
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "CONFLICT: $server (PID $pid) failed to start or crashed" | tee -a "$TEST_LOG"
            ((failed_count++))
        else
            echo "OK: $server (PID $pid) is running" | tee -a "$TEST_LOG"
        fi
    done
    
    # Check for port conflicts
    echo "Checking port usage..." | tee -a "$TEST_LOG"
    ss -tulpn | grep -E ":111|:8000|:3000|:5000|:11112|:8080" | tee -a "$TEST_LOG" || echo "No common ports in use"
    
    # Check for resource exhaustion
    echo "Resource usage during test:" | tee -a "$TEST_LOG"
    tail -5 "$RESOURCE_LOG" | tee -a "$TEST_LOG"
    
    # Check error log for specific conflicts
    echo "Errors detected:" | tee -a "$TEST_LOG"
    if [[ -s "$ERROR_LOG" ]]; then
        echo "=== ERROR DETAILS ===" | tee -a "$TEST_LOG"
        cat "$ERROR_LOG" | tee -a "$TEST_LOG"
    else
        echo "No errors in error log" | tee -a "$TEST_LOG"
    fi
    
    return $failed_count
}

# Function to test concurrent operations
test_concurrent_operations() {
    echo "=== TESTING CONCURRENT OPERATIONS ===" | tee -a "$TEST_LOG"
    
    # Test multiple clients accessing same MCP
    echo "Testing multiple client access..." | tee -a "$TEST_LOG"
    
    # Wait for MCPs to stabilize
    sleep 5
    
    # Try to connect to multiple MCPs simultaneously
    echo "Testing simultaneous MCP connections..." | tee -a "$TEST_LOG"
    
    # Check stdio streams
    echo "Checking stdio stream conflicts..." | tee -a "$TEST_LOG"
    lsof -p $(pgrep -f "mcp-server" | head -5 | tr '\n' ',' | sed 's/,$//') 2>/dev/null | grep -E "0u|1u|2u" | tee -a "$TEST_LOG" || echo "No stdio conflicts detected"
    
    # Check file descriptor usage
    echo "File descriptor usage per MCP:" | tee -a "$TEST_LOG"
    for pid in $(pgrep -f "mcp-"); do
        if kill -0 "$pid" 2>/dev/null; then
            fd_count=$(lsof -p "$pid" 2>/dev/null | wc -l)
            echo "PID $pid: $fd_count file descriptors" | tee -a "$TEST_LOG"
        fi
    done
}

# Main execution
main() {
    # Initialize
    MCP_PIDS=()
    
    # Cleanup existing processes
    cleanup_mcps
    
    # Start simultaneous test
    start_all_mcps_simultaneously
    
    # Wait for startup
    echo "Waiting 10 seconds for startup completion..." | tee -a "$TEST_LOG"
    sleep 10
    
    # Test concurrent operations
    test_concurrent_operations
    
    # Analyze results
    analyze_conflicts
    conflict_count=$?
    
    # Stop monitoring
    if [[ -n "${MONITOR_PID:-}" ]]; then
        kill "$MONITOR_PID" 2>/dev/null || true
    fi
    
    # Final process snapshot
    echo "Final process state:" | tee -a "$TEST_LOG"
    ps aux | grep -E "mcp|claude-flow|ruv-swarm" | grep -v grep | tee -a "$PROCESS_LOG" || echo "No MCP processes found"
    
    # Cleanup
    echo "Cleaning up test processes..." | tee -a "$TEST_LOG"
    for pid in "${MCP_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    
    cleanup_mcps
    
    echo "=== TEST COMPLETED at $(date) ===" | tee -a "$TEST_LOG"
    echo "Total conflicts detected: $conflict_count" | tee -a "$TEST_LOG"
    echo "Full log: $TEST_LOG"
    echo "Resource log: $RESOURCE_LOG" 
    echo "Error log: $ERROR_LOG"
    echo "Process log: $PROCESS_LOG"
    
    return $conflict_count
}

# Run the test
main "$@"