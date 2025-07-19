#!/bin/bash

###############################################################################
# Deploy Remaining AI Agents for SutazAI
# Builds and starts available agent containers
###############################################################################

set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }
warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
info() { echo -e "${BLUE}[INFO]${NC} $1"; }

cd /opt/sutazaiapp

log "Deploying remaining AI agents..."

# Check which agents have Dockerfiles
agents_to_deploy=()
for agent_dir in docker/*/; do
    if [ -f "$agent_dir/Dockerfile" ]; then
        agent_name=$(basename "$agent_dir")
        if ! docker ps | grep -q "sutazai-$agent_name"; then
            agents_to_deploy+=("$agent_name")
        fi
    fi
done

if [ ${#agents_to_deploy[@]} -eq 0 ]; then
    info "All available agents are already running"
    exit 0
fi

log "Found ${#agents_to_deploy[@]} agents to deploy: ${agents_to_deploy[*]}"

# Build and start each agent
for agent in "${agents_to_deploy[@]}"; do
    log "Building $agent..."
    
    # Skip if it's already in docker-compose
    if grep -q "  $agent:" docker-compose.yml 2>/dev/null; then
        log "Starting $agent from docker-compose..."
        docker-compose up -d "$agent" 2>/dev/null || warning "Could not start $agent from compose"
    else
        # Build and run manually
        if docker build -t "sutazai/$agent:latest" "./docker/$agent/" 2>/dev/null; then
            log "Starting $agent container..."
            
            # Default port assignment
            case "$agent" in
                "autogpt-real") port=8090 ;;
                "crewai") port=8091 ;;
                "agentgpt") port=8092 ;;
                "privategpt") port=8093 ;;
                "llamaindex") port=8094 ;;
                "flowise") port=8095 ;;
                *) port=$((8100 + RANDOM % 100)) ;;
            esac
            
            docker run -d \
                --name "sutazai-$agent" \
                --network sutazaiapp_sutazai-network \
                -p "$port:8080" \
                -v "$PWD/workspace:/workspace" \
                -e AGENT_NAME="$agent" \
                -e BACKEND_URL="http://sutazai-backend:8000" \
                "sutazai/$agent:latest" || warning "Failed to start $agent"
                
            info "$agent started on port $port"
        else
            warning "Failed to build $agent"
        fi
    fi
done

# Wait for agents to initialize
log "Waiting for agents to initialize..."
sleep 10

# Check deployed agents
log "Checking deployed agents..."
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep sutazai | sort

# Update agent count
total_agents=$(docker ps | grep -c sutazai || true)
log "Total running containers: $total_agents"

# Create agent status report
cat > AGENT_STATUS.md << EOF
# SutazAI Agent Status Report
Generated: $(date)

## Running Agents
$(docker ps --format "- {{.Names}} ({{.Status}})" | grep sutazai | sort)

## Total Agents: $total_agents

## Access Points
- Main UI: http://localhost:8501
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Agent Ports
$(docker ps --format "{{.Names}}: {{.Ports}}" | grep sutazai | grep -v -E "postgres|redis|ollama|chroma|qdrant" | sed 's/.*://g' | sed 's/->.*//g' | sort)
EOF

log "Agent deployment complete! Status report: AGENT_STATUS.md"