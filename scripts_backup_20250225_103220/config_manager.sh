#!/bin/bash

validate_config() {
    local config_file=$1
    local required_keys=("AI_MODEL_PATH" "DATABASE_URL" "API_ENDPOINT")
    
    for key in "${required_keys[@]}"; do
        if ! grep -q "^$key=" "$config_file"; then
            handle_error "Missing required configuration: $key"
        fi
    done
} 