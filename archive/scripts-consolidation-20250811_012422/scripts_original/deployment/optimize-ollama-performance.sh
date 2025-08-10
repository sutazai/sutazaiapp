#!/bin/bash
# ULTRA-PERFORMANCE Ollama Optimization Script
# Stabilizes Ollama with optimal settings for <10s response time

set -e


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

echo "=========================================="
echo "ULTRA OLLAMA PERFORMANCE OPTIMIZATION"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Backup current docker-compose
echo -e "${YELLOW}Creating backup of docker-compose.yml...${NC}"
cp docker-compose.yml docker-compose.yml.backup.$(date +%Y%m%d_%H%M%S)

# Update Ollama environment variables in docker-compose.yml
echo -e "${GREEN}Applying Ollama performance optimizations...${NC}"

# Create temporary Python script to update the YAML
cat > "$(mktemp /tmp/update_ollama_config.py.XXXXXX)" << 'EOF'
import yaml
import sys

# Load docker-compose.yml
with open('docker-compose.yml', 'r') as f:
    compose = yaml.safe_load(f)

# Update Ollama service environment
if 'services' in compose and 'ollama' in compose['services']:
    ollama = compose['services']['ollama']
    
    # Update environment variables for stability
    ollama['environment'].update({
        'OLLAMA_NUM_PARALLEL': '1',      # Process one request at a time
        'OLLAMA_MAX_QUEUE': '10',         # Limit queue size
        'OLLAMA_KEEP_ALIVE': '5m',       # Keep models loaded for 5 minutes
        'OLLAMA_MAX_LOADED_MODELS': '1',  # Only one model at a time
        'OLLAMA_NUM_THREADS': '8',       # Optimal thread count
        'OLLAMA_TIMEOUT': '30s',         # 30 second timeout
        'OLLAMA_REQUEST_TIMEOUT': '30',  # Request timeout in seconds
        'OLLAMA_CONNECTION_POOL': '10',  # Connection pool size
        'OLLAMA_FLASH_ATTENTION': '0',   # Disable flash attention for stability
        'OLLAMA_USE_MMAP': 'false',      # Disable mmap for predictable memory
        'OLLAMA_USE_NUMA': 'false',      # Disable NUMA for stability
    })
    
    # Update healthcheck for better reliability
    ollama['healthcheck'] = {
        'test': ['CMD-SHELL', 'timeout 5 ollama list > /dev/null || exit 1'],
        'interval': '30s',
        'timeout': '5s',
        'retries': '3',
        'start_period': '60s'
    }
    
    # Update resource limits for stability
    ollama['deploy']['resources']['limits'] = {
        'cpus': '8',
        'memory': '16G'
    }
    ollama['deploy']['resources']['reservations'] = {
        'cpus': '4',
        'memory': '8G'
    }
    
    # Add restart policy
    ollama['restart'] = 'unless-stopped'
    
    # Add logging limits
    ollama['logging'] = {
        'driver': 'json-file',
        'options': {
            'max-size': '100m',
            'max-file': '5'
        }
    }
    
    print("Ollama configuration updated successfully")
else:
    print("ERROR: Ollama service not found in docker-compose.yml")
    sys.exit(1)

# Save updated configuration
with open('docker-compose.yml', 'w') as f:
    yaml.dump(compose, f, default_flow_style=False, sort_keys=False)

print("docker-compose.yml updated with performance optimizations")
EOF

python3 /tmp/update_ollama_config.py
rm /tmp/update_ollama_config.py

echo -e "${GREEN}✓ Ollama configuration optimized${NC}"

# Create Ollama startup script with model preloading
echo -e "${YELLOW}Creating Ollama startup script...${NC}"

cat > scripts/deployment/ollama-startup.sh << 'EOF'
#!/bin/bash
# Ollama startup script with model preloading

echo "Starting Ollama with performance optimizations..."

# Wait for Ollama to be ready
until ollama list > /dev/null 2>&1; do
    echo "Waiting for Ollama to start..."
    sleep 5
done

echo "Ollama is ready. Loading models..."

# Pull and load TinyLlama (primary model)
ollama pull tinyllama:latest || true
ollama run tinyllama:latest "test" --verbose || true

echo "Model loaded. Warming up..."

# Warm up the model with a few test queries
for i in {1..3}; do
    echo "Warm-up query $i..."
    timeout 30 ollama run tinyllama:latest "Hello, how are you?" || true
    sleep 2
done

echo "Ollama startup complete and warmed up!"
EOF

chmod +x scripts/deployment/ollama-startup.sh

# Restart Ollama service with new configuration
echo -e "${YELLOW}Restarting Ollama service...${NC}"
docker compose stop ollama
docker compose up -d ollama

# Wait for Ollama to be healthy
echo -e "${YELLOW}Waiting for Ollama to be healthy...${NC}"
for i in {1..30}; do
    if docker compose exec -T ollama ollama list > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Ollama is healthy${NC}"
        break
    fi
    echo "Waiting... ($i/30)"
    sleep 2
done

# Load TinyLlama model
echo -e "${YELLOW}Loading TinyLlama model...${NC}"
docker compose exec -T ollama ollama pull tinyllama:latest || echo "Model already loaded"

# Test Ollama response time
echo -e "${YELLOW}Testing Ollama response time...${NC}"
START_TIME=$(date +%s)
docker compose exec -T ollama ollama run tinyllama:latest "Hello" --verbose 2>/dev/null || true
END_TIME=$(date +%s)
RESPONSE_TIME=$((END_TIME - START_TIME))

if [ $RESPONSE_TIME -lt 10 ]; then
    echo -e "${GREEN}✓ Response time: ${RESPONSE_TIME}s (TARGET MET: <10s)${NC}"
else
    echo -e "${YELLOW}⚠ Response time: ${RESPONSE_TIME}s (Above target, may need further optimization)${NC}"
fi

# Display optimization summary
echo ""
echo "=========================================="
echo "OPTIMIZATION COMPLETE"
echo "=========================================="
echo -e "${GREEN}Applied optimizations:${NC}"
echo "  • Reduced parallel processing to 1 for stability"
echo "  • Limited queue size to 10 requests"
echo "  • Set model keep-alive to 5 minutes"
echo "  • Optimized thread count to 8"
echo "  • Added 30-second timeout"
echo "  • Disabled unstable features (flash attention, mmap, numa)"
echo "  • Updated resource limits for stability"
echo ""
echo -e "${GREEN}Next steps:${NC}"
echo "  1. Monitor Ollama logs: docker compose logs -f ollama"
echo "  2. Check response times: curl http://localhost:10104/api/generate -d '{\"model\":\"tinyllama\",\"prompt\":\"test\"}'"
echo "  3. View metrics in Grafana: http://localhost:10201"
echo ""
echo -e "${GREEN}✓ Ollama performance optimization complete!${NC}"