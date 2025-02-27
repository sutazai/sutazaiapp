#!/bin/bash

# Check network connectivity
check_network() {
    local endpoints=(
        "https://example.com"
        "https://api.example.com"
        "https://storage.example.com"
    )
    
    for endpoint in "${endpoints[@]}"; do
        if ! curl -Is "$endpoint" >/dev/null; then
            log_error "Cannot reach $endpoint"
            return 1
        fi
    done
} 