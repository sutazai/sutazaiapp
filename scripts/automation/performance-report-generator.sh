#!/bin/bash
# Purpose: Generate comprehensive performance reports for SutazAI system
# Usage: ./performance-report-generator.sh [--format json|html|both] [--period daily|weekly|monthly]
# Requires: Docker, jq, bc

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="/opt/sutazaiapp"
LOG_DIR="$BASE_DIR/logs"
REPORT_DIR="$BASE_DIR/reports/performance"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# Configuration
REPORT_FORMAT="both"    # json, html, or both
REPORT_PERIOD="daily"   # daily, weekly, monthly
RETENTION_DAYS=90       # Keep performance reports for this many days

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --format)
            REPORT_FORMAT="$2"
            shift 2
            ;;
        --period)
            REPORT_PERIOD="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--format json|html|both] [--period daily|weekly|monthly]"
            exit 1
            ;;
    esac
done

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local log_file="$LOG_DIR/performance_report_$TIMESTAMP.log"
    
    echo "[$timestamp] $level: $message" >> "$log_file"
    
    case $level in
        ERROR) echo -e "${RED}[$timestamp] ERROR: $message${NC}" ;;
        WARN) echo -e "${YELLOW}[$timestamp] WARN: $message${NC}" ;;
        INFO) echo -e "${BLUE}[$timestamp] INFO: $message${NC}" ;;
        SUCCESS) echo -e "${GREEN}[$timestamp] SUCCESS: $message${NC}" ;;
    esac
}

# Setup report directory
setup_report_directory() {
    log "INFO" "Setting up performance report directory..."
    mkdir -p "$REPORT_DIR" "$LOG_DIR"
}

# Collect system performance metrics
collect_system_metrics() {
    log "INFO" "Collecting system performance metrics..."
    
    local metrics="{}"
    
    # CPU metrics
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//')
    local load_avg=$(uptime | awk -F'load average:' '{print $2}' | xargs)
    local cpu_cores=$(nproc)
    
    # Memory metrics
    local mem_info=$(free -b | grep "Mem:")
    local mem_total=$(echo $mem_info | awk '{print $2}')
    local mem_used=$(echo $mem_info | awk '{print $3}')
    local mem_free=$(echo $mem_info | awk '{print $4}')
    local mem_usage_percent=$(echo "scale=2; $mem_used * 100 / $mem_total" | bc -l)
    
    # Disk metrics
    local disk_info=$(df / | tail -1)
    local disk_total=$(echo $disk_info | awk '{print $2}')
    local disk_used=$(echo $disk_info | awk '{print $3}')
    local disk_usage_percent=$(echo $disk_info | awk '{print $5}' | sed 's/%//')
    
    # Network metrics (if available)
    local network_rx_bytes=0
    local network_tx_bytes=0
    if [[ -f /proc/net/dev ]]; then
        network_rx_bytes=$(awk '/eth0|ens|enp/ {rx += $2} END {print rx+0}' /proc/net/dev)
        network_tx_bytes=$(awk '/eth0|ens|enp/ {tx += $10} END {print tx+0}' /proc/net/dev)
    fi
    
    # Create metrics JSON
    metrics=$(jq -n \
        --arg cpu_usage "$cpu_usage" \
        --arg load_avg "$load_avg" \
        --arg cpu_cores "$cpu_cores" \
        --arg mem_total "$mem_total" \
        --arg mem_used "$mem_used" \
        --arg mem_free "$mem_free" \
        --arg mem_usage_percent "$mem_usage_percent" \
        --arg disk_total "$disk_total" \
        --arg disk_used "$disk_used" \
        --arg disk_usage_percent "$disk_usage_percent" \
        --arg network_rx_bytes "$network_rx_bytes" \
        --arg network_tx_bytes "$network_tx_bytes" \
        '{
            "cpu": {
                "usage_percent": $cpu_usage,
                "load_average": $load_avg,
                "cores": ($cpu_cores | tonumber)
            },
            "memory": {
                "total_bytes": ($mem_total | tonumber),
                "used_bytes": ($mem_used | tonumber),
                "free_bytes": ($mem_free | tonumber),
                "usage_percent": ($mem_usage_percent | tonumber)
            },
            "disk": {
                "total_kb": ($disk_total | tonumber),
                "used_kb": ($disk_used | tonumber),
                "usage_percent": ($disk_usage_percent | tonumber)
            },
            "network": {
                "rx_bytes": ($network_rx_bytes | tonumber),
                "tx_bytes": ($network_tx_bytes | tonumber)
            }
        }')
    
    echo "$metrics"
}

# Collect Docker container metrics
collect_container_metrics() {
    log "INFO" "Collecting Docker container performance metrics..."
    
    local containers_metrics="[]"
    
    # Get all SutazAI containers
    while IFS= read -r container; do
        if [[ -n "$container" ]]; then
            local stats=$(docker stats --no-stream --format "{{.CPUPerc}},{{.MemUsage}},{{.MemPerc}},{{.NetIO}},{{.BlockIO}}" "$container" 2>/dev/null || echo "0.00%,0B / 0B,0.00%,0B / 0B,0B / 0B")
            
            IFS=',' read -r cpu_perc mem_usage mem_perc net_io block_io <<< "$stats"
            
            # Parse memory usage
            local mem_used=$(echo "$mem_usage" | awk '{print $1}' | sed 's/[^0-9.]//g')
            local mem_total=$(echo "$mem_usage" | awk '{print $3}' | sed 's/[^0-9.]//g')
            
            # Parse network I/O
            local net_rx=$(echo "$net_io" | awk '{print $1}' | sed 's/[^0-9.]//g')
            local net_tx=$(echo "$net_io" | awk '{print $3}' | sed 's/[^0-9.]//g')
            
            # Parse block I/O
            local block_read=$(echo "$block_io" | awk '{print $1}' | sed 's/[^0-9.]//g')
            local block_write=$(echo "$block_io" | awk '{print $3}' | sed 's/[^0-9.]//g')
            
            # Add container metrics
            local container_metric=$(jq -n \
                --arg name "$container" \
                --arg cpu_perc "$cpu_perc" \
                --arg mem_used "$mem_used" \
                --arg mem_total "$mem_total" \
                --arg mem_perc "$mem_perc" \
                --arg net_rx "$net_rx" \
                --arg net_tx "$net_tx" \
                --arg block_read "$block_read" \
                --arg block_write "$block_write" \
                '{
                    "name": $name,
                    "cpu_percent": $cpu_perc,
                    "memory": {
                        "used": $mem_used,
                        "total": $mem_total,
                        "percent": $mem_perc
                    },
                    "network": {
                        "rx": $net_rx,
                        "tx": $net_tx
                    },
                    "block_io": {
                        "read": $block_read,
                        "write": $block_write
                    }
                }')
            
            containers_metrics=$(echo "$containers_metrics" | jq ". += [$container_metric]")
        fi
    done < <(docker ps --format "{{.Names}}" | grep "^sutazai-" || true)
    
    echo "$containers_metrics"
}

# Collect AI agent performance metrics
collect_agent_metrics() {
    log "INFO" "Collecting AI agent performance metrics..."
    
    local agents_metrics="[]"
    local expected_agents=(
        "sutazai-senior-ai-engineer:8001"
        "sutazai-infrastructure-devops-manager:8002"
        "sutazai-testing-qa-validator:8003"
        "sutazai-agent-orchestrator:8004"
        "sutazai-ai-system-architect:8005"
    )
    
    for agent_config in "${expected_agents[@]}"; do
        IFS=':' read -r agent_name port <<< "$agent_config"
        
        # Check if agent is responding
        local response_time=0
        local is_healthy=false
        local status_code=0
        
        if docker ps --format "{{.Names}}" | grep -q "^${agent_name}$"; then
            local health_check=$(curl -s -m 10 -w "%{http_code}:%{time_total}" "http://localhost:${port}/health" 2>/dev/null || echo "000:0")
            IFS=':' read -r status_code response_time <<< "$health_check"
            
            if [[ "$status_code" == "200" ]]; then
                is_healthy=true
            fi
        fi
        
        # Get agent-specific metrics if available
        local total_requests=0
        local avg_response_time=0
        local error_rate=0
        
        # Try to get metrics from agent's metrics endpoint
        if [[ "$is_healthy" == "true" ]]; then
            local metrics_response=$(curl -s -m 5 "http://localhost:${port}/metrics" 2>/dev/null || echo "{}")
            total_requests=$(echo "$metrics_response" | jq -r '.total_requests // 0' 2>/dev/null || echo 0)
            avg_response_time=$(echo "$metrics_response" | jq -r '.avg_response_time // 0' 2>/dev/null || echo 0)
            error_rate=$(echo "$metrics_response" | jq -r '.error_rate // 0' 2>/dev/null || echo 0)
        fi
        
        # Create agent metrics
        local agent_metric=$(jq -n \
            --arg name "$agent_name" \
            --arg port "$port" \
            --argjson is_healthy "$is_healthy" \
            --arg status_code "$status_code" \
            --arg response_time "$response_time" \
            --arg total_requests "$total_requests" \
            --arg avg_response_time "$avg_response_time" \
            --arg error_rate "$error_rate" \
            '{
                "name": $name,
                "port": ($port | tonumber),
                "is_healthy": $is_healthy,
                "health_check": {
                    "status_code": ($status_code | tonumber),
                    "response_time": ($response_time | tonumber)
                },
                "performance": {
                    "total_requests": ($total_requests | tonumber),
                    "avg_response_time": ($avg_response_time | tonumber),
                    "error_rate": ($error_rate | tonumber)
                }
            }')
        
        agents_metrics=$(echo "$agents_metrics" | jq ". += [$agent_metric]")
    done
    
    echo "$agents_metrics"
}

# Collect database performance metrics
collect_database_metrics() {
    log "INFO" "Collecting database performance metrics..."
    
    local db_metrics="{}"
    
    # PostgreSQL metrics
    if docker ps --format "{{.Names}}" | grep -q "sutazai-postgres-minimal"; then
        local pg_db_size=$(docker exec sutazai-postgres-minimal psql -U sutazai -d sutazai -t -c "SELECT pg_database_size('sutazai');" 2>/dev/null | xargs || echo "0")
        local pg_connections=$(docker exec sutazai-postgres-minimal psql -U sutazai -d sutazai -t -c "SELECT count(*) FROM pg_stat_activity;" 2>/dev/null | xargs || echo "0")
        local pg_max_connections=$(docker exec sutazai-postgres-minimal psql -U sutazai -d sutazai -t -c "SHOW max_connections;" 2>/dev/null | xargs || echo "100")
        local pg_cache_hit_ratio=$(docker exec sutazai-postgres-minimal psql -U sutazai -d sutazai -t -c "SELECT round(sum(blks_hit)*100.0/sum(blks_hit+blks_read), 2) FROM pg_stat_database WHERE datname = 'sutazai';" 2>/dev/null | xargs || echo "0")
        local pg_slow_queries=$(docker exec sutazai-postgres-minimal psql -U sutazai -d sutazai -t -c "SELECT count(*) FROM pg_stat_statements WHERE mean_time > 1000;" 2>/dev/null | xargs || echo "0")
        
        # Add PostgreSQL metrics
        db_metrics=$(echo "$db_metrics" | jq \
            --arg size "$pg_db_size" \
            --arg connections "$pg_connections" \
            --arg max_connections "$pg_max_connections" \
            --arg cache_hit_ratio "$pg_cache_hit_ratio" \
            --arg slow_queries "$pg_slow_queries" \
            '.postgresql = {
                "database_size_bytes": ($size | tonumber),
                "active_connections": ($connections | tonumber),
                "max_connections": ($max_connections | tonumber),
                "connection_usage_percent": (($connections | tonumber) * 100 / ($max_connections | tonumber)),
                "cache_hit_ratio": ($cache_hit_ratio | tonumber),
                "slow_queries": ($slow_queries | tonumber)
            }')
    fi
    
    # Redis metrics
    if docker ps --format "{{.Names}}" | grep -q "sutazai-redis-minimal"; then
        local redis_memory=$(docker exec sutazai-redis-minimal redis-cli info memory | grep used_memory: | cut -d: -f2 | tr -d '\r' || echo "0")
        local redis_keys=$(docker exec sutazai-redis-minimal redis-cli dbsize 2>/dev/null || echo "0")
        local redis_hits=$(docker exec sutazai-redis-minimal redis-cli info stats | grep keyspace_hits | cut -d: -f2 | tr -d '\r' || echo "0")
        local redis_misses=$(docker exec sutazai-redis-minimal redis-cli info stats | grep keyspace_misses | cut -d: -f2 | tr -d '\r' || echo "0")
        local redis_hit_rate=0
        
        if [[ $redis_hits -gt 0 || $redis_misses -gt 0 ]]; then
            redis_hit_rate=$(echo "scale=2; $redis_hits * 100 / ($redis_hits + $redis_misses)" | bc -l 2>/dev/null || echo "0")
        fi
        
        # Add Redis metrics
        db_metrics=$(echo "$db_metrics" | jq \
            --arg memory "$redis_memory" \
            --arg keys "$redis_keys" \
            --arg hits "$redis_hits" \
            --arg misses "$redis_misses" \
            --arg hit_rate "$redis_hit_rate" \
            '.redis = {
                "memory_usage_bytes": ($memory | tonumber),
                "total_keys": ($keys | tonumber),
                "keyspace_hits": ($hits | tonumber),
                "keyspace_misses": ($misses | tonumber),
                "hit_rate_percent": ($hit_rate | tonumber)
            }')
    fi
    
    echo "$db_metrics"
}

# Collect Ollama performance metrics  
collect_ollama_metrics() {
    log "INFO" "Collecting Ollama performance metrics..."
    
    local ollama_metrics="{}"
    
    if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        # Test model performance
        local start_time=$(date +%s.%N)
        local test_response=$(curl -s -X POST http://localhost:11434/api/generate \
            -d '{"model": "tinyllama", "prompt": "Hello world", "stream": false}' \
            --max-time 30 2>/dev/null | jq -r '.response // empty' 2>/dev/null)
        local end_time=$(date +%s.%N)
        local response_time=$(echo "$end_time - $start_time" | bc -l)
        
        local model_available=false
        local response_length=0
        
        if [[ -n "$test_response" && "$test_response" != "null" ]]; then
            model_available=true
            response_length=${#test_response}
        fi
        
        # Get model information
        local model_info=$(curl -s http://localhost:11434/api/show -d '{"name": "tinyllama"}' 2>/dev/null || echo "{}")
        local model_size=$(echo "$model_info" | jq -r '.details.parameter_size // "unknown"' 2>/dev/null || echo "unknown")
        
        ollama_metrics=$(jq -n \
            --argjson model_available "$model_available" \
            --arg response_time "$response_time" \
            --arg response_length "$response_length" \
            --arg model_size "$model_size" \
            '{
                "service_available": true,
                "model_available": $model_available,
                "test_response_time": ($response_time | tonumber),
                "test_response_length": ($response_length | tonumber),
                "model_info": {
                    "name": "tinyllama",
                    "parameter_size": $model_size
                }
            }')
    else
        ollama_metrics='{"service_available": false}'
    fi
    
    echo "$ollama_metrics"
}

# Generate performance summary and recommendations
generate_performance_summary() {
    local system_metrics="$1"
    local container_metrics="$2"
    local agent_metrics="$3"
    local db_metrics="$4"
    local ollama_metrics="$5"
    
    log "INFO" "Generating performance summary and recommendations..."
    
    local summary="{}"
    local recommendations="[]"
    local performance_score=100
    
    # Analyze system metrics
    local mem_usage=$(echo "$system_metrics" | jq -r '.memory.usage_percent')
    local disk_usage=$(echo "$system_metrics" | jq -r '.disk.usage_percent')
    local cpu_cores=$(echo "$system_metrics" | jq -r '.cpu.cores')
    local load_avg=$(echo "$system_metrics" | jq -r '.cpu.load_average' | awk '{print $1}')
    
    # Memory analysis
    if (( $(echo "$mem_usage > 80" | bc -l) )); then
        performance_score=$((performance_score - 15))
        recommendations=$(echo "$recommendations" | jq '. += ["High memory usage detected (' + "\"$mem_usage\"" + '%). Consider adding more RAM or optimizing memory-intensive processes."]')
    elif (( $(echo "$mem_usage > 60" | bc -l) )); then
        performance_score=$((performance_score - 5))
        recommendations=$(echo "$recommendations" | jq '. += ["Moderate memory usage (' + "\"$mem_usage\"" + '%). Monitor for memory leaks."]')
    fi
    
    # Disk analysis
    if (( disk_usage > 85 )); then
        performance_score=$((performance_score - 20))
        recommendations=$(echo "$recommendations" | jq '. += ["Critical disk usage (' + "\"$disk_usage\"" + '%). Clean up logs and temporary files immediately."]')
    elif (( disk_usage > 70 )); then
        performance_score=$((performance_score - 10))
        recommendations=$(echo "$recommendations" | jq '. += ["High disk usage (' + "\"$disk_usage\"" + '%). Schedule regular cleanup tasks."]')
    fi
    
    # Load average analysis
    local load_threshold=$(echo "$cpu_cores * 1.5" | bc -l)
    if (( $(echo "$load_avg > $load_threshold" | bc -l) )); then
        performance_score=$((performance_score - 15))
        recommendations=$(echo "$recommendations" | jq '. += ["High system load (' + "\"$load_avg\"" + ' on ' + "\"$cpu_cores\"" + ' cores). Consider load balancing or scaling."]')
    fi
    
    # Agent health analysis
    local healthy_agents=$(echo "$agent_metrics" | jq '[.[] | select(.is_healthy == true)] | length')
    local total_agents=$(echo "$agent_metrics" | jq 'length')
    local agent_health_percent=$((healthy_agents * 100 / total_agents))
    
    if (( agent_health_percent < 80 )); then
        performance_score=$((performance_score - 25))
        recommendations=$(echo "$recommendations" | jq '. += ["Only ' + "\"$agent_health_percent\"" + '% of AI agents are healthy. Check agent logs and restart failed agents."]')
    elif (( agent_health_percent < 100 )); then
        performance_score=$((performance_score - 10))
        recommendations=$(echo "$recommendations" | jq '. += ["Some AI agents are unhealthy (' + "\"$agent_health_percent\"" + '% healthy). Monitor and investigate."]')
    fi
    
    # Database performance analysis
    if echo "$db_metrics" | jq -e '.postgresql' >/dev/null; then
        local pg_cache_hit=$(echo "$db_metrics" | jq -r '.postgresql.cache_hit_ratio')
        local pg_conn_usage=$(echo "$db_metrics" | jq -r '.postgresql.connection_usage_percent')
        
        if (( $(echo "$pg_cache_hit < 90" | bc -l) )); then
            performance_score=$((performance_score - 10))
            recommendations=$(echo "$recommendations" | jq '. += ["PostgreSQL cache hit ratio is low (' + "\"$pg_cache_hit\"" + '%). Consider increasing shared_buffers."]')
        fi
        
        if (( $(echo "$pg_conn_usage > 80" | bc -l) )); then
            performance_score=$((performance_score - 15))
            recommendations=$(echo "$recommendations" | jq '. += ["PostgreSQL connection usage is high (' + "\"$pg_conn_usage\"" + '%). Consider connection pooling."]')
        fi
    fi
    
    # Ollama performance analysis
    if echo "$ollama_metrics" | jq -e '.service_available' >/dev/null; then
        local ollama_available=$(echo "$ollama_metrics" | jq -r '.service_available')
        local model_available=$(echo "$ollama_metrics" | jq -r '.model_available // false')
        
        if [[ "$ollama_available" != "true" ]]; then
            performance_score=$((performance_score - 30))
            recommendations=$(echo "$recommendations" | jq '. += ["Ollama service is not available. Check service status and configuration."]')
        elif [[ "$model_available" != "true" ]]; then
            performance_score=$((performance_score - 20))
            recommendations=$(echo "$recommendations" | jq '. += ["Ollama model is not responding. Check model loading and configuration."]')
        fi
    fi
    
    # Determine overall performance status
    local performance_status="excellent"
    if (( performance_score < 60 )); then
        performance_status="poor"
    elif (( performance_score < 75 )); then
        performance_status="fair"
    elif (( performance_score < 90 )); then
        performance_status="good"
    fi
    
    summary=$(jq -n \
        --arg status "$performance_status" \
        --arg score "$performance_score" \
        --arg healthy_agents "$healthy_agents" \
        --arg total_agents "$total_agents" \
        --arg mem_usage "$mem_usage" \
        --arg disk_usage "$disk_usage" \
        --argjson recommendations "$recommendations" \
        '{
            "overall_status": $status,
            "performance_score": ($score | tonumber),
            "agent_health": {
                "healthy": ($healthy_agents | tonumber),
                "total": ($total_agents | tonumber),
                "percentage": (($healthy_agents | tonumber) * 100 / ($total_agents | tonumber))
            },
            "resource_usage": {
                "memory_percent": ($mem_usage | tonumber),
                "disk_percent": ($disk_usage | tonumber)
            },
            "recommendations": $recommendations
        }')
    
    echo "$summary"
}

# Generate JSON report
generate_json_report() {
    local system_metrics="$1"
    local container_metrics="$2"
    local agent_metrics="$3"
    local db_metrics="$4"
    local ollama_metrics="$5"
    local summary="$6"
    
    local json_report_file="$REPORT_DIR/performance_report_${REPORT_PERIOD}_$TIMESTAMP.json"
    
    log "INFO" "Generating JSON performance report..."
    
    jq -n \
        --arg timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        --arg hostname "$(hostname)" \
        --arg period "$REPORT_PERIOD" \
        --argjson system "$system_metrics" \
        --argjson containers "$container_metrics" \
        --argjson agents "$agent_metrics" \
        --argjson databases "$db_metrics" \
        --argjson ollama "$ollama_metrics" \
        --argjson summary "$summary" \
        '{
            "report_info": {
                "timestamp": $timestamp,
                "hostname": $hostname,
                "period": $period
            },
            "system_metrics": $system,
            "container_metrics": $containers,
            "agent_metrics": $agents,
            "database_metrics": $databases,
            "ollama_metrics": $ollama,
            "summary": $summary
        }' > "$json_report_file"
    
    log "SUCCESS" "JSON report generated: $json_report_file"
    
    # Create symlink to latest report
    ln -sf "$json_report_file" "$REPORT_DIR/latest_performance_report.json"
    
    echo "$json_report_file"
}

# Generate HTML report
generate_html_report() {
    local json_report_file="$1"
    local html_report_file="${json_report_file%.json}.html"
    
    log "INFO" "Generating HTML performance report..."
    
    # Read JSON data
    local report_data=$(cat "$json_report_file")
    local timestamp=$(echo "$report_data" | jq -r '.report_info.timestamp')
    local hostname=$(echo "$report_data" | jq -r '.report_info.hostname')
    local period=$(echo "$report_data" | jq -r '.report_info.period')
    local summary=$(echo "$report_data" | jq -r '.summary')
    
    # Generate HTML
    cat > "$html_report_file" << EOF
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SutazAI Performance Report - $period</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; }
        .status-excellent { color: #28a745; }
        .status-good { color: #17a2b8; }
        .status-fair { color: #ffc107; }
        .status-poor { color: #dc3545; }
        .metric-card { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #007bff; }
        .metric-title { font-weight: bold; font-size: 1.1em; margin-bottom: 10px; }
        .metric-value { font-size: 1.5em; font-weight: bold; }
        .recommendations { background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .recommendations ul { margin: 10px 0; padding-left: 20px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .status-healthy { color: #28a745; }
        .status-unhealthy { color: #dc3545; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>SutazAI Performance Report</h1>
            <p><strong>Period:</strong> $(echo $period | tr '[:lower:]' '[:upper:]') | <strong>Host:</strong> $hostname</p>
            <p><strong>Generated:</strong> $timestamp</p>
        </div>

        <div class="metric-card">
            <div class="metric-title">Overall Performance Status</div>
            <div class="metric-value status-$(echo "$summary" | jq -r '.overall_status')">
                $(echo "$summary" | jq -r '.overall_status | ascii_upcase') ($(echo "$summary" | jq -r '.performance_score')/100)
            </div>
        </div>

        <div class="grid">
            <div class="metric-card">
                <div class="metric-title">System Resources</div>
                <p><strong>Memory Usage:</strong> $(echo "$report_data" | jq -r '.system_metrics.memory.usage_percent')%</p>
                <p><strong>Disk Usage:</strong> $(echo "$report_data" | jq -r '.system_metrics.disk.usage_percent')%</p>
                <p><strong>CPU Cores:</strong> $(echo "$report_data" | jq -r '.system_metrics.cpu.cores')</p>
                <p><strong>Load Average:</strong> $(echo "$report_data" | jq -r '.system_metrics.cpu.load_average')</p>
            </div>

            <div class="metric-card">
                <div class="metric-title">AI Agents Status</div>
                <p><strong>Healthy Agents:</strong> $(echo "$summary" | jq -r '.agent_health.healthy')/$(echo "$summary" | jq -r '.agent_health.total') ($(echo "$summary" | jq -r '.agent_health.percentage')%)</p>
            </div>
        </div>

        <div class="metric-card">
            <div class="metric-title">AI Agents Details</div>
            <table>
                <thead>
                    <tr>
                        <th>Agent Name</th>
                        <th>Port</th>
                        <th>Status</th>
                        <th>Response Time</th>
                    </tr>
                </thead>
                <tbody>
EOF

    # Add agent rows
    echo "$report_data" | jq -r '.agent_metrics[] | 
        "<tr><td>" + .name + "</td><td>" + (.port | tostring) + "</td><td class=\"status-" + 
        (if .is_healthy then "healthy\">Healthy" else "unhealthy\">Unhealthy" end) + 
        "</td><td>" + (.health_check.response_time | tostring) + "s</td></tr>"' >> "$html_report_file"

    cat >> "$html_report_file" << EOF
                </tbody>
            </table>
        </div>

        <div class="recommendations">
            <h3>Recommendations</h3>
EOF

    # Add recommendations
    local recommendations_count=$(echo "$summary" | jq '.recommendations | length')
    if [[ $recommendations_count -gt 0 ]]; then
        echo "<ul>" >> "$html_report_file"
        echo "$summary" | jq -r '.recommendations[] | "<li>" + . + "</li>"' >> "$html_report_file"
        echo "</ul>" >> "$html_report_file"
    else
        echo "<p>No specific recommendations at this time. System is performing well.</p>" >> "$html_report_file"
    fi

    cat >> "$html_report_file" << EOF
        </div>

        <div class="metric-card">
            <div class="metric-title">Raw Data (JSON)</div>
            <pre style="background: #f8f9fa; padding: 10px; border-radius: 4px; overflow-x: auto; font-size: 0.9em;">$(echo "$report_data" | jq .)</pre>
        </div>
    </div>
</body>
</html>
EOF

    log "SUCCESS" "HTML report generated: $html_report_file"
    
    # Create symlink to latest HTML report
    ln -sf "$html_report_file" "$REPORT_DIR/latest_performance_report.html"
    
    echo "$html_report_file"
}

# Clean old performance reports
clean_old_reports() {
    log "INFO" "Cleaning old performance reports (older than $RETENTION_DAYS days)..."
    
    if [[ ! -d "$REPORT_DIR" ]]; then
        log "INFO" "Report directory does not exist, skipping cleanup"
        return 0
    fi
    
    local deleted_count=0
    
    while IFS= read -r -d '' report; do
        local basename=$(basename "$report")
        
        log "INFO" "Deleting old report: $basename"
        rm "$report"
        ((deleted_count++))
    done < <(find "$REPORT_DIR" -name "performance_report_*.json" -o -name "performance_report_*.html" -type f -mtime +$RETENTION_DAYS -print0 2>/dev/null)
    
    if [[ $deleted_count -gt 0 ]]; then
        log "SUCCESS" "Deleted $deleted_count old performance reports"
    else
        log "INFO" "No old performance reports found for deletion"
    fi
}

# Main execution
main() {
    log "INFO" "Starting performance report generation for SutazAI system"
    log "INFO" "Report format: $REPORT_FORMAT, Period: $REPORT_PERIOD"
    
    # Setup directories
    setup_report_directory
    
    # Collect all performance metrics
    log "INFO" "Collecting performance metrics..."
    local system_metrics=$(collect_system_metrics)
    local container_metrics=$(collect_container_metrics)
    local agent_metrics=$(collect_agent_metrics)
    local db_metrics=$(collect_database_metrics)
    local ollama_metrics=$(collect_ollama_metrics)
    
    # Generate performance summary
    local summary=$(generate_performance_summary "$system_metrics" "$container_metrics" "$agent_metrics" "$db_metrics" "$ollama_metrics")
    
    # Generate reports based on format
    local json_report=""
    local html_report=""
    
    if [[ "$REPORT_FORMAT" == "json" || "$REPORT_FORMAT" == "both" ]]; then
        json_report=$(generate_json_report "$system_metrics" "$container_metrics" "$agent_metrics" "$db_metrics" "$ollama_metrics" "$summary")
    fi
    
    if [[ "$REPORT_FORMAT" == "html" || "$REPORT_FORMAT" == "both" ]]; then
        if [[ -z "$json_report" ]]; then
            json_report=$(generate_json_report "$system_metrics" "$container_metrics" "$agent_metrics" "$db_metrics" "$ollama_metrics" "$summary")
        fi
        html_report=$(generate_html_report "$json_report")
    fi
    
    # Clean old reports
    clean_old_reports
    
    log "SUCCESS" "Performance report generation completed"
    
    # Show summary
    echo
    echo "============================================"
    echo "       PERFORMANCE REPORT SUMMARY"
    echo "============================================"
    echo "Period: $REPORT_PERIOD"
    echo "Format: $REPORT_FORMAT"
    echo "Overall Status: $(echo "$summary" | jq -r '.overall_status | ascii_upcase')"
    echo "Performance Score: $(echo "$summary" | jq -r '.performance_score')/100"
    echo "Agent Health: $(echo "$summary" | jq -r '.agent_health.healthy')/$(echo "$summary" | jq -r '.agent_health.total') ($(echo "$summary" | jq -r '.agent_health.percentage')%)"
    if [[ -n "$json_report" ]]; then
        echo "JSON Report: $json_report"
    fi
    if [[ -n "$html_report" ]]; then
        echo "HTML Report: $html_report"
    fi
    echo "Timestamp: $(date)"
    echo "============================================"
}

# Run main function
main "$@"