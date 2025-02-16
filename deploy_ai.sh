#!/bin/bash
set -euo pipefail

# Source utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/scripts/deploy_utils.sh"

log "INFO" "Starting AI service deployment"

# Verify virtual environment
verify_virtualenv || handle_error "Virtual environment verification failed"

# Start AI service
{
    log "DEBUG" "Loading AI models"
    python3 "$SCRIPT_DIR/load_models.py" || handle_error "Failed to load AI models"
    
    log "DEBUG" "Starting AI inference service"
    python3 "$SCRIPT_DIR/ai_service.py" || handle_error "Failed to start AI service"
    
    log "DEBUG" "Starting API server"
    uvicorn api:app --host 0.0.0.0 --port 8000 || handle_error "API server failed to start"
} | modern_progress_bar "AI Service" "Model Loading" "Initialization"

log "INFO" "AI service deployed successfully"