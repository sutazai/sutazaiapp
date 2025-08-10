#!/bin/bash

# Automated Health Check Monitoring Script for SutazAI System
# Monitors all services and generates health reports
# Author: System Administrator
# Date: 2025-08-09

set -euo pipefail

# Colors for output

# Signal handlers for graceful shutdown
cleanup_and_exit() {
    local exit_code="${1:-0}"
    echo "Script interrupted, cleaning up..." >&2
    # Clean up any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    exit "$exit_code"
}

trap 'cleanup_and_exit 130' INT
trap 'cleanup_and_exit 143' TERM
trap 'cleanup_and_exit 1' ERR

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs/monitoring"
REPORT_FILE="${LOG_DIR}/health_report_$(date +%Y%m%d_%H%M%S).json"
ALERT_THRESHOLD=80  # CPU/Memory threshold for alerts

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Initialize report
declare -A health_report
health_report[timestamp]=$(date -Iseconds)
health_report[total_containers]=0
health_report[healthy_containers]=0
health_report[unhealthy_containers]=0
health_report[warnings]=0

# Function to print colored output
print_color() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to check container health
check_container_health() {
    local container_name=$1
    local health_status=$(docker inspect --format='{{.State.Health.Status}}' "$container_name" 2>/dev/null || echo "none")
    local running_status=$(docker inspect --format='{{.State.Running}}' "$container_name" 2>/dev/null || echo "false")
    
    if [[ "$running_status" == "false" ]]; then
        echo "stopped"
    elif [[ "$health_status" == "healthy" ]]; then
        echo "healthy"
    elif [[ "$health_status" == "unhealthy" ]]; then
        echo "unhealthy"
    elif [[ "$health_status" == "starting" ]]; then
        echo "starting"
    elif [[ "$health_status" == "none" ]] && [[ "$running_status" == "true" ]]; then
        echo "running"
    else
        echo "unknown"
    fi
}

# Function to get container resource usage
get_container_stats() {
    local container_name=$1
    local stats=$(docker stats --no-stream --format "{{json .}}" "$container_name" 2>/dev/null || echo "{}")
    echo "$stats"
}

# Function to check service endpoint
check_endpoint() {
    local url=$1
    local timeout=${2:-5}
    
    if curl -sf -m "$timeout" "$url" > /dev/null 2>&1; then
        echo "reachable"
    else
        echo "unreachable"
    fi
}

# Function to check port availability
check_port() {
    local port=$1
    if nc -z localhost "$port" 2>/dev/null; then
        echo "open"
    else
        echo "closed"
    fi
}

# Main monitoring function
monitor_services() {
    print_color "$BLUE" "========================================="
    print_color "$BLUE" "SutazAI System Health Monitor"
    print_color "$BLUE" "Time: $(date)"
    print_color "$BLUE" "========================================="
    echo ""
    
    # Core Infrastructure Services
    print_color "$BLUE" "Core Infrastructure Services:"
    print_color "$BLUE" "-----------------------------"
    
    declare -A services=(
        ["PostgreSQL"]="sutazai-postgres:10000:/health"
        ["Redis"]="sutazai-redis:10001:"
        ["Neo4j"]="sutazai-neo4j:10002:/browser"
        ["Ollama"]="sutazai-ollama:10104:/api/tags"
        ["RabbitMQ"]="sutazai-rabbitmq:10008:/api/health/checks/virtual-hosts"
    )
    
    for service in "${!services[@]}"; do
        IFS=':' read -r container port endpoint <<< "${services[$service]}"
        
        health=$(check_container_health "$container")
        port_status=$(check_port "$port")
        
        if [[ "$health" == "healthy" ]] || [[ "$health" == "running" ]]; then
            print_color "$GREEN" "  ✓ $service ($container)"
            echo "    Health: $health | Port $port: $port_status"
            ((health_report[healthy_containers]++))
        else
            print_color "$RED" "  ✗ $service ($container)"
            echo "    Health: $health | Port $port: $port_status"
            ((health_report[unhealthy_containers]++))
        fi
        ((health_report[total_containers]++))
    done
    
    echo ""
    
    # Vector Databases
    print_color "$BLUE" "Vector Databases:"
    print_color "$BLUE" "-----------------"
    
    declare -A vector_dbs=(
        ["ChromaDB"]="sutazai-chromadb:10100"
        ["Qdrant"]="sutazai-qdrant:10101"
    )
    
    for db in "${!vector_dbs[@]}"; do
        IFS=':' read -r container port <<< "${vector_dbs[$db]}"
        
        health=$(check_container_health "$container")
        port_status=$(check_port "$port")
        
        if [[ "$health" == "healthy" ]] || [[ "$health" == "running" ]]; then
            print_color "$GREEN" "  ✓ $db ($container)"
            echo "    Health: $health | Port $port: $port_status"
            ((health_report[healthy_containers]++))
        else
            print_color "$RED" "  ✗ $db ($container)"
            echo "    Health: $health | Port $port: $port_status"
            ((health_report[unhealthy_containers]++))
        fi
        ((health_report[total_containers]++))
    done
    
    echo ""
    
    # Agent Services
    print_color "$BLUE" "Agent Services:"
    print_color "$BLUE" "---------------"
    
    declare -A agents=(
        ["AI Agent Orchestrator"]="sutazai-ai-agent-orchestrator:8589"
        ["Ollama Integration"]="sutazai-ollama-integration:8090"
        ["Hardware Optimizer"]="sutazai-hardware-resource-optimizer:11110"
        ["Jarvis Hardware"]="sutazai-jarvis-hardware-resource-optimizer:11104"
        ["Jarvis Automation"]="sutazai-jarvis-automation-agent:11102"
    )
    
    for agent in "${!agents[@]}"; do
        IFS=':' read -r container port <<< "${agents[$agent]}"
        
        if docker ps --format "{{.Names}}" | grep -q "^$container$"; then
            health=$(check_container_health "$container")
            endpoint_status=$(check_endpoint "http://localhost:$port/health")
            
            if [[ "$health" == "healthy" ]] && [[ "$endpoint_status" == "reachable" ]]; then
                print_color "$GREEN" "  ✓ $agent"
                echo "    Container: $health | Endpoint: $endpoint_status"
                ((health_report[healthy_containers]++))
            else
                print_color "$YELLOW" "  ⚠ $agent"
                echo "    Container: $health | Endpoint: $endpoint_status"
                ((health_report[warnings]++))
            fi
        else
            print_color "$RED" "  ✗ $agent (not running)"
            ((health_report[unhealthy_containers]++))
        fi
        ((health_report[total_containers]++))
    done
    
    echo ""
    
    # Monitoring Stack
    print_color "$BLUE" "Monitoring Stack:"
    print_color "$BLUE" "-----------------"
    
    declare -A monitoring=(
        ["Prometheus"]="sutazai-prometheus:10200"
        ["Grafana"]="sutazai-grafana:10201"
        ["Loki"]="sutazai-loki:10202"
    )
    
    for service in "${!monitoring[@]}"; do
        IFS=':' read -r container port <<< "${monitoring[$service]}"
        
        health=$(check_container_health "$container")
        port_status=$(check_port "$port")
        
        if [[ "$health" == "healthy" ]] || [[ "$health" == "running" ]]; then
            print_color "$GREEN" "  ✓ $service"
            echo "    Health: $health | Port $port: $port_status"
            ((health_report[healthy_containers]++))
        else
            print_color "$RED" "  ✗ $service"
            echo "    Health: $health | Port $port: $port_status"
            ((health_report[unhealthy_containers]++))
        fi
        ((health_report[total_containers]++))
    done
    
    echo ""
    
    # System Resources
    print_color "$BLUE" "System Resources:"
    print_color "$BLUE" "-----------------"
    
    # CPU and Memory usage
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
    mem_usage=$(free -m | awk 'NR==2{printf "%.1f", $3*100/$2}')
    disk_usage=$(df -h / | awk 'NR==2{print $5}' | sed 's/%//')
    
    echo "  CPU Usage: ${cpu_usage}%"
    echo "  Memory Usage: ${mem_usage}%"
    echo "  Disk Usage: ${disk_usage}%"
    
    # Alert on high resource usage
    if (( $(echo "$cpu_usage > $ALERT_THRESHOLD" | bc -l) )); then
        print_color "$YELLOW" "  ⚠ WARNING: High CPU usage detected!"
        ((health_report[warnings]++))
    fi
    
    if (( $(echo "$mem_usage > $ALERT_THRESHOLD" | bc -l) )); then
        print_color "$YELLOW" "  ⚠ WARNING: High memory usage detected!"
        ((health_report[warnings]++))
    fi
    
    echo ""
    
    # Summary
    print_color "$BLUE" "========================================="
    print_color "$BLUE" "Summary:"
    print_color "$BLUE" "========================================="
    
    local health_percentage=$(( health_report[healthy_containers] * 100 / health_report[total_containers] ))
    
    echo "Total Containers: ${health_report[total_containers]}"
    print_color "$GREEN" "Healthy: ${health_report[healthy_containers]}"
    
    if [[ ${health_report[unhealthy_containers]} -gt 0 ]]; then
        print_color "$RED" "Unhealthy: ${health_report[unhealthy_containers]}"
    fi
    
    if [[ ${health_report[warnings]} -gt 0 ]]; then
        print_color "$YELLOW" "Warnings: ${health_report[warnings]}"
    fi
    
    echo ""
    echo "System Health Score: ${health_percentage}%"
    
    if [[ $health_percentage -ge 90 ]]; then
        print_color "$GREEN" "Status: EXCELLENT"
    elif [[ $health_percentage -ge 70 ]]; then
        print_color "$YELLOW" "Status: GOOD (with minor issues)"
    elif [[ $health_percentage -ge 50 ]]; then
        print_color "$YELLOW" "Status: DEGRADED"
    else
        print_color "$RED" "Status: CRITICAL"
    fi
    
    # Save report to JSON
    cat > "$REPORT_FILE" <<EOF
{
  "timestamp": "${health_report[timestamp]}",
  "total_containers": ${health_report[total_containers]},
  "healthy_containers": ${health_report[healthy_containers]},
  "unhealthy_containers": ${health_report[unhealthy_containers]},
  "warnings": ${health_report[warnings]},
  "health_percentage": $health_percentage,
  "cpu_usage": "$cpu_usage",
  "memory_usage": "$mem_usage",
  "disk_usage": "$disk_usage"
}
EOF
    
    echo ""
    print_color "$GREEN" "Report saved to: $REPORT_FILE"
}

# Continuous monitoring mode
continuous_monitor() {
    local interval=${1:-60}  # Default 60 seconds
    
    print_color "$BLUE" "Starting continuous monitoring (interval: ${interval}s)"
    print_color "$BLUE" "Press Ctrl+C to stop"
    echo ""
    
    # Timeout mechanism to prevent infinite loops
    LOOP_TIMEOUT=${LOOP_TIMEOUT:-300}  # 5 minute default timeout
    loop_start=$(date +%s)
    while true; do
        clear
        monitor_services
        sleep "$interval"
        # Check for timeout
        current_time=$(date +%s)
        if [[ $((current_time - loop_start)) -gt $LOOP_TIMEOUT ]]; then
            echo 'Loop timeout reached after ${LOOP_TIMEOUT}s, exiting...' >&2
            break
        fi

    done
}

# Main script logic
main() {
    case "${1:-once}" in
        once)
            monitor_services
            ;;
        continuous)
            continuous_monitor "${2:-60}"
            ;;
        help|--help|-h)
            echo "Usage: $0 [once|continuous [interval]]"
            echo ""
            echo "Options:"
            echo "  once       - Run health check once (default)"
            echo "  continuous - Run continuously with specified interval (default: 60s)"
            echo ""
            echo "Examples:"
            echo "  $0                  # Run once"
            echo "  $0 continuous       # Run every 60 seconds"
            echo "  $0 continuous 30    # Run every 30 seconds"
            ;;
        *)
            print_color "$RED" "Invalid option: $1"
            echo "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"