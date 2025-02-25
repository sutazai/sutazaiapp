#!/bin/bash

# Ensure script is run as root
if [ "$EUID" -ne 0 ]; then
    echo "❌ Please run as root"
    exit 1
fi

# Base directory
BASE_DIR="/root/sutazai/v1"

# Create base directory if it doesn't exist
if [ ! -d "$BASE_DIR" ]; then
    mkdir -p "$BASE_DIR"
    chmod 755 "$BASE_DIR"
    chown root:root "$BASE_DIR"
fi

# Create all subdirectories
DIRS=(
    "agents/architect"
    "agents/factory"
    "agents/loyalty"
    "agents/omnicoder"
    "agents/reality"
    "agents/research"
    "agents/self_evolution"
    "agents/self_improvement"
    "agents/Semgrep"
    "agents/supreme"
    "agents/supreme_agent"
    "agents/vision"
    "avatar/emotion"
    "avatar/ethnicity"
    "avatar/interface"
    "backend/alembic"
    "backend/config"
    "backend/models"
    "backend/processing"
    "frontend/assets"
    "frontend/components"
    "models/memory"
    "monitoring/grafana/dashboards"
    "security"
    "services"
    "system"
    "system_optimizer"
    "temporal"
    "tests"
    "ui"
    ".github/workflows"
)

for DIR in "${DIRS[@]}"; do
    FULL_PATH="$BASE_DIR/$DIR"
    if [ ! -d "$FULL_PATH" ]; then
        mkdir -p "$FULL_PATH"
        chmod 755 "$FULL_PATH"
        chown root:root "$FULL_PATH"
        echo "✅ Created directory: $FULL_PATH"
    else
        echo "ℹ️ Directory exists: $FULL_PATH"
    fi
done

echo "✅ Directory structure setup complete" 