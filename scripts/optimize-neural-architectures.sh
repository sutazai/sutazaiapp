#!/bin/bash
# Purpose: Deploy neural architecture optimization for all AI agents
# Usage: ./optimize-neural-architectures.sh [--dry-run] [--benchmark-only]
# Requirements: Python 3.8+, required Python packages

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
OPTIMIZATION_DIR="/opt/sutazaiapp/models/optimization"
LOG_DIR="/opt/sutazaiapp/logs"
BACKUP_DIR="/opt/sutazaiapp/model_backups"
REPORT_DIR="/opt/sutazaiapp/reports"

# Parse arguments
DRY_RUN=false
BENCHMARK_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --benchmark-only)
            BENCHMARK_ONLY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--benchmark-only]"
            exit 1
            ;;
    esac
done

# Functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✓${NC} $1"
}

warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

error() {
    echo -e "${RED}✗${NC} $1"
}

check_requirements() {
    log "Checking requirements..."
    
    # Check Python version
    if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
        error "Python 3.8+ is required"
        exit 1
    fi
    
    # Check required directories
    mkdir -p "$LOG_DIR" "$BACKUP_DIR" "$REPORT_DIR"
    
    # Check Ollama service
    if ! systemctl is-active --quiet ollama; then
        warning "Ollama service is not running. Starting it..."
        sudo systemctl start ollama
    fi
    
    success "All requirements met"
}

backup_models() {
    log "Backing up current models..."
    
    local backup_timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_path="$BACKUP_DIR/models_$backup_timestamp"
    
    mkdir -p "$backup_path"
    
    # Backup model files
    if [ -d "/opt/sutazaiapp/models" ]; then
        cp -r /opt/sutazaiapp/models/* "$backup_path/" 2>/dev/null || true
        success "Models backed up to $backup_path"
    else
        warning "No models directory found to backup"
    fi
}

install_dependencies() {
    log "Installing Python dependencies..."
    
    # Create requirements file
    cat > /tmp/optimization_requirements.txt << EOF
numpy>=1.21.0
asyncio
psutil>=5.8.0
httpx>=0.24.0
matplotlib>=3.5.0
aioredis>=2.0.0
EOF
    
    # Install dependencies
    pip3 install -r /tmp/optimization_requirements.txt --quiet
    
    success "Dependencies installed"
}

run_benchmarks() {
    log "Running performance benchmarks..."
    
    cd "$OPTIMIZATION_DIR"
    
    # Run baseline benchmarks
    python3 performance_benchmark.py --baseline > "$LOG_DIR/benchmark_baseline.log" 2>&1
    
    success "Benchmarks completed"
}

optimize_models() {
    log "Starting neural architecture optimization..."
    
    cd "$OPTIMIZATION_DIR"
    
    if [ "$DRY_RUN" = true ]; then
        warning "DRY RUN MODE - No actual optimization will be performed"
        python3 optimization_orchestrator.py --dry-run
    else
        # Run the main optimization
        python3 optimization_orchestrator.py 2>&1 | tee "$LOG_DIR/optimization_$(date +%Y%m%d_%H%M%S).log"
    fi
    
    success "Optimization completed"
}

validate_optimization() {
    log "Validating optimized models..."
    
    cd "$OPTIMIZATION_DIR"
    
    # Run validation tests
    python3 << EOF
import asyncio
from performance_benchmark import PerformanceBenchmark, BenchmarkConfig

async def validate():
    benchmark = PerformanceBenchmark()
    
    # Test key models
    models = ['tinyllama', 'codellama-7b', 'mistral-7b', 'phi-2']
    
    for model in models:
        config = BenchmarkConfig(
            model_name=model,
            test_prompts=["Test validation prompt"],
            num_iterations=10,
            warmup_iterations=2
        )
        
        result = await benchmark.benchmark_model(
            f"/opt/sutazaiapp/models/{model}_optimized.onnx",
            config,
            optimization_type='validation'
        )
        
        print(f"{model}: Speedup={result.speedup_vs_baseline:.2f}x, "
              f"Quality={result.quality_preservation:.2%}")

asyncio.run(validate())
EOF
    
    success "Validation completed"
}

generate_report() {
    log "Generating optimization report..."
    
    local report_file="$REPORT_DIR/optimization_report_$(date +%Y%m%d_%H%M%S).json"
    
    # Copy the main report
    if [ -f "/opt/sutazaiapp/optimization_report.json" ]; then
        cp "/opt/sutazaiapp/optimization_report.json" "$report_file"
        
        # Generate summary
        python3 -c "
import json
with open('$report_file', 'r') as f:
    report = json.load(f)
    summary = report.get('summary', {})
    print('\n=== OPTIMIZATION SUMMARY ===')
    print(f\"Models Optimized: {summary.get('models_optimized', 0)}\")
    print(f\"Average Speedup: {summary.get('average_speedup', 0):.2f}x\")
    print(f\"Memory Saved: {summary.get('total_memory_saved_gb', 0):.1f} GB\")
    print(f\"Time Taken: {summary.get('optimization_time_hours', 0):.1f} hours\")
    print('=' * 28)
"
        
        success "Report saved to $report_file"
    else
        warning "No optimization report found"
    fi
}

update_agent_configs() {
    log "Updating agent configurations..."
    
    if [ "$DRY_RUN" = true ]; then
        warning "Skipping agent config updates (dry run)"
        return
    fi
    
    # Update agent configs to use optimized models
    python3 << 'EOF'
import json
import os
from pathlib import Path

config_dir = Path("/opt/sutazaiapp/agents")
updated = 0

# Map of optimized models
optimized_models = {
    "tinyllama": "tinyllama_optimized",
    "codellama-7b": "codellama_int8_optimized",
    "mistral-7b": "mistral_int8_optimized",
    "phi-2": "phi2_int4_optimized"
}

# Update each agent's config
for agent_dir in config_dir.iterdir():
    if agent_dir.is_dir():
        config_file = agent_dir / "config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                # Update model reference
                if 'model' in config:
                    original_model = config['model']
                    if original_model in optimized_models:
                        config['model'] = optimized_models[original_model]
                        config['optimized'] = True
                        
                        with open(config_file, 'w') as f:
                            json.dump(config, f, indent=2)
                        
                        updated += 1
            except Exception as e:
                print(f"Error updating {agent_dir.name}: {e}")

print(f"Updated {updated} agent configurations")
EOF
    
    success "Agent configurations updated"
}

restart_services() {
    log "Restarting AI services..."
    
    if [ "$DRY_RUN" = true ]; then
        warning "Skipping service restart (dry run)"
        return
    fi
    
    # Restart Ollama with optimized settings
    sudo systemctl restart ollama
    
    # Restart agent services
    for service in sutazai-agents sutazai-backend; do
        if systemctl is-active --quiet "$service"; then
            sudo systemctl restart "$service"
            success "Restarted $service"
        fi
    done
}

# Main execution
main() {
    echo -e "${GREEN}Neural Architecture Optimization for SutazAI${NC}"
    echo "=============================================="
    
    check_requirements
    
    if [ "$BENCHMARK_ONLY" = true ]; then
        run_benchmarks
        generate_report
        exit 0
    fi
    
    backup_models
    install_dependencies
    
    # Run optimization pipeline
    optimize_models
    validate_optimization
    generate_report
    
    if [ "$DRY_RUN" = false ]; then
        update_agent_configs
        restart_services
    fi
    
    echo
    success "Neural architecture optimization completed successfully!"
    echo
    echo "Next steps:"
    echo "1. Monitor agent performance for 24 hours"
    echo "2. Check logs in $LOG_DIR"
    echo "3. Review report in $REPORT_DIR"
    echo "4. Run benchmarks: $0 --benchmark-only"
    
    if [ "$DRY_RUN" = true ]; then
        echo
        warning "This was a dry run. To apply optimizations, run without --dry-run"
    fi
}

# Run main function
main