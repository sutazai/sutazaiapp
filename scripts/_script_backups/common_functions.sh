#!/bin/bash

# Common functions for system audit scripts

log_message() {
    local level="INFO"
    if [[ "$1" == "WARNING:"* ]] || [[ "$1" == "ERROR:"* ]]; then
        level=$(echo "$1" | cut -d: -f1)
    fi
    printf "[%s] [%-7s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$level" "$1" | tee -a $AUDIT_LOG
}

check_dependencies() {
    local scripts=("$@")
    local missing=0
    for script in "${scripts[@]}"; do
        if [ ! -f "$script" ]; then
            log_message "ERROR: Required script $script not found"
            missing=$((missing + 1))
        fi
    done
    
    if [ $missing -gt 0 ]; then
        log_message "FATAL: Missing $missing required scripts"
        exit 1
    fi
} 