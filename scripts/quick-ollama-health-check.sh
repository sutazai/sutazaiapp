#!/bin/bash

# Quick Ollama Health Check for SutazAI
# Purpose: Rapid health assessment of Ollama optimization systems
# Usage: ./quick-ollama-health-check.sh

set -e

echo "SutazAI Ollama Health Check"
echo "========================="
echo "$(date)"
echo ""

# Check if Ollama is running
echo "1. Checking Ollama service..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "   ✓ Ollama service is running"
    MODELS=$(curl -s http://localhost:11434/api/tags | python3 -c "import sys, json; print(len(json.load(sys.stdin).get('models', [])))")
    echo "   ✓ $MODELS models available"
else
    echo "   ✗ Ollama service is not accessible"
    echo "   Please start Ollama: ollama serve"
fi

# Check Redis connectivity
echo ""
echo "2. Checking Redis service..."
if redis-cli -h localhost -p 10001 ping > /dev/null 2>&1; then
    echo "   ✓ Redis service is running on port 10001"
else
    echo "   ✗ Redis service is not accessible on port 10001"
    echo "   Please check Redis configuration"
fi

# Check optimization modules
echo ""
echo "3. Checking optimization modules..."
OPTIM_DIR="/opt/sutazaiapp/agents/core"
MODULES=("ollama_performance_optimizer.py" "ollama_batch_processor.py" "ollama_context_optimizer.py" "ollama_model_manager.py")

for module in "${MODULES[@]}"; do
    if [ -f "$OPTIM_DIR/$module" ]; then
        echo "   ✓ $module"
    else
        echo "   ✗ $module (missing)"
    fi
done

# Check configuration files
echo ""
echo "4. Checking configuration files..."
CONFIG_FILES=(
    "/opt/sutazaiapp/config/ollama.yaml"
    "/opt/sutazaiapp/config/ollama_optimization.yaml"
    "/opt/sutazaiapp/config/ollama_performance_optimization.yaml"
)

for config in "${CONFIG_FILES[@]}"; do
    if [ -f "$config" ]; then
        echo "   ✓ $(basename $config)"
    else
        echo "   ✗ $(basename $config) (missing)"
    fi
done

# Check log directory
echo ""
echo "5. Checking log directory..."
LOG_DIR="/opt/sutazaiapp/logs"
if [ -d "$LOG_DIR" ]; then
    echo "   ✓ Log directory exists"
    LOG_COUNT=$(find "$LOG_DIR" -name "*ollama*" -o -name "*optimization*" | wc -l)
    echo "   ✓ $LOG_COUNT optimization log files found"
else
    echo "   ✗ Log directory missing"
    mkdir -p "$LOG_DIR"
    echo "   ✓ Created log directory"
fi

# System resource check
echo ""
echo "6. System resources..."
MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
CPU_CORES=$(nproc)
echo "   ✓ Available RAM: ${MEMORY_GB}GB"
echo "   ✓ CPU cores: $CPU_CORES"

if [ "$MEMORY_GB" -lt 8 ]; then
    echo "   ⚠ Warning: Low memory for optimal AI performance"
fi

if [ "$CPU_CORES" -lt 4 ]; then
    echo "   ⚠ Warning: Limited CPU cores for concurrent processing"
fi

echo ""
echo "Health check completed!"
echo ""
echo "To run full optimization:"
echo "  python3 /opt/sutazaiapp/scripts/optimize-ollama-performance.py --full-optimization"
echo ""
echo "For quick performance test:"
echo "  python3 /opt/sutazaiapp/scripts/optimize-ollama-performance.py --health-check"
echo ""
