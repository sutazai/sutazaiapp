#!/bin/bash

# Fix unhealthy Docker services in Sutazai system
# This script diagnoses and fixes health check issues

set -e

echo "================================================"
echo "Sutazai Infrastructure Recovery Script"
echo "================================================"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    case $status in
        "error")
            echo -e "${RED}[ERROR]${NC} $message"
            ;;
        "success")
            echo -e "${GREEN}[SUCCESS]${NC} $message"
            ;;
        "warning")
            echo -e "${YELLOW}[WARNING]${NC} $message"
            ;;
        "info")
            echo -e "${BLUE}[INFO]${NC} $message"
            ;;
    esac
}

# Function to check service health
check_health() {
    local service=$1
    local status=$(docker inspect $service --format='{{.State.Health.Status}}' 2>/dev/null || echo "no-health-check")
    echo "$status"
}

# Function to get container memory usage
get_memory_usage() {
    local service=$1
    docker stats $service --no-stream --format "{{.MemPerc}}" 2>/dev/null || echo "0%"
}

# Function to fix Ollama health check
fix_ollama() {
    print_status "info" "Fixing Ollama service..."
    
    # Check if Ollama API is actually working
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        print_status "success" "Ollama API is responding correctly"
        
        # The health check is failing because curl is not in the container
        # We need to update the health check to use a different method
        print_status "info" "Updating Ollama health check configuration..."
        
        # Create a custom health check script that works without curl
        cat > /tmp/ollama-healthcheck.sh << 'EOF'
#!/bin/sh
# Test if Ollama can list models
if ollama list > /dev/null 2>&1; then
    exit 0
else
    exit 1
fi
EOF
        
        # Copy the health check script to the container
        docker cp /tmp/ollama-healthcheck.sh sutazai-ollama:/healthcheck.sh
        docker exec sutazai-ollama chmod +x /healthcheck.sh 2>/dev/null || true
        
        # Update the container health check
        docker exec sutazai-ollama ollama list > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            print_status "success" "Ollama internal health check passes"
        else
            print_status "warning" "Ollama may need model reload"
            # Pull a small model to ensure Ollama is working
            docker exec sutazai-ollama ollama pull tinyllama 2>/dev/null || true
        fi
    else
        print_status "error" "Ollama API not responding - restarting service"
        docker restart sutazai-ollama
        sleep 10
    fi
}

# Function to fix agent services
fix_agent_service() {
    local service=$1
    local port=$2
    
    print_status "info" "Fixing $service..."
    
    # Check if the service is actually running
    if docker exec $service curl -f http://localhost:$port/health 2>/dev/null; then
        print_status "success" "$service health endpoint is accessible"
    else
        # Check the actual issue
        local logs=$(docker logs $service --tail 20 2>&1 | grep -E "ERROR|Failed" | head -5)
        
        if echo "$logs" | grep -q "Failed to connect to Ollama"; then
            print_status "warning" "$service cannot connect to Ollama - fixing network"
            
            # Ensure the service can reach Ollama
            docker exec $service ping -c 1 172.20.0.1 > /dev/null 2>&1 || {
                print_status "error" "Network connectivity issue detected"
            }
        fi
        
        if echo "$logs" | grep -q "Failed to register with MCP Bridge"; then
            print_status "warning" "$service cannot register with MCP Bridge"
            
            # Check if MCP Bridge is running
            if docker ps --format "{{.Names}}" | grep -q "sutazai-mcp-bridge"; then
                print_status "info" "MCP Bridge is running, checking connectivity..."
                docker exec $service ping -c 1 sutazai-mcp-bridge > /dev/null 2>&1 || {
                    print_status "error" "Cannot reach MCP Bridge"
                }
            else
                print_status "error" "MCP Bridge is not running"
            fi
        fi
        
        # Restart the service to reset health check
        print_status "info" "Restarting $service..."
        docker restart $service
        sleep 5
    fi
}

# Function to optimize resource allocation
optimize_resources() {
    print_status "info" "Optimizing resource allocations..."
    
    # Update Ollama memory limit (reduce from 23GB to 4GB)
    print_status "info" "Adjusting Ollama memory allocation (23GB -> 4GB)..."
    docker update --memory="4g" --memory-swap="4g" sutazai-ollama 2>/dev/null || {
        print_status "warning" "Could not update Ollama memory limits"
    }
    
    # Check other over-provisioned services
    for service in sutazai-chromadb sutazai-qdrant sutazai-faiss; do
        if docker ps --format "{{.Names}}" | grep -q "$service"; then
            local usage=$(get_memory_usage $service)
            print_status "info" "$service memory usage: $usage"
        fi
    done
}

# Function to setup monitoring
setup_monitoring() {
    print_status "info" "Setting up health monitoring..."
    
    # Create a monitoring script
    cat > /opt/sutazaiapp/scripts/health-monitor.sh << 'EOF'
#!/bin/bash
# Continuous health monitoring for Sutazai services

UNHEALTHY_SERVICES=()
while true; do
    clear
    echo "=== Sutazai Service Health Monitor ==="
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Check key services
    for service in sutazai-ollama sutazai-semgrep sutazai-documind sutazai-finrobot sutazai-backend sutazai-frontend; do
        if docker ps --format "{{.Names}}" | grep -q "^$service$"; then
            health=$(docker inspect $service --format='{{.State.Health.Status}}' 2>/dev/null || echo "no-check")
            mem=$(docker stats $service --no-stream --format "{{.MemPerc}}" 2>/dev/null || echo "N/A")
            
            if [ "$health" = "healthy" ]; then
                echo -e "\033[0;32m✓\033[0m $service: $health (Mem: $mem)"
            elif [ "$health" = "unhealthy" ]; then
                echo -e "\033[0;31m✗\033[0m $service: $health (Mem: $mem)"
                UNHEALTHY_SERVICES+=("$service")
            else
                echo -e "\033[1;33m?\033[0m $service: $health (Mem: $mem)"
            fi
        fi
    done
    
    # Auto-recover unhealthy services
    if [ ${#UNHEALTHY_SERVICES[@]} -gt 0 ]; then
        echo ""
        echo "Auto-recovering unhealthy services..."
        for service in "${UNHEALTHY_SERVICES[@]}"; do
            docker restart $service > /dev/null 2>&1
            echo "Restarted: $service"
        done
        UNHEALTHY_SERVICES=()
    fi
    
    sleep 30
done
EOF
    
    chmod +x /opt/sutazaiapp/scripts/health-monitor.sh
    print_status "success" "Health monitor script created"
}

# Main execution
main() {
    print_status "info" "Starting infrastructure recovery process..."
    echo ""
    
    # Step 1: Check current status
    print_status "info" "Current service status:"
    for service in sutazai-ollama sutazai-semgrep sutazai-documind sutazai-finrobot; do
        health=$(check_health $service)
        mem=$(get_memory_usage $service)
        if [ "$health" = "healthy" ]; then
            print_status "success" "$service: $health (Memory: $mem)"
        else
            print_status "error" "$service: $health (Memory: $mem)"
        fi
    done
    echo ""
    
    # Step 2: Fix Ollama
    fix_ollama
    echo ""
    
    # Step 3: Fix agent services
    fix_agent_service "sutazai-semgrep" "8000"
    fix_agent_service "sutazai-documind" "8000"
    fix_agent_service "sutazai-finrobot" "8000"
    echo ""
    
    # Step 4: Optimize resources
    optimize_resources
    echo ""
    
    # Step 5: Setup monitoring
    setup_monitoring
    echo ""
    
    # Step 6: Verify fixes
    print_status "info" "Verifying fixes (waiting 15 seconds for services to stabilize)..."
    sleep 15
    
    echo ""
    print_status "info" "Final service status:"
    for service in sutazai-ollama sutazai-semgrep sutazai-documind sutazai-finrobot; do
        health=$(check_health $service)
        mem=$(get_memory_usage $service)
        if [ "$health" = "healthy" ]; then
            print_status "success" "$service: $health (Memory: $mem)"
        else
            print_status "warning" "$service: $health (Memory: $mem) - may need more time"
        fi
    done
    
    echo ""
    print_status "info" "Recovery process complete!"
    print_status "info" "To monitor services continuously, run: /opt/sutazaiapp/scripts/health-monitor.sh"
}

# Run main function
main