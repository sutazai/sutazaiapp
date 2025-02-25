#!/bin/bash

# File Monitor Class
FileMonitor() {
    local engine=$1
    
    # File states
    declare -A FILE_STATES
    
    # Monitor files
    monitor_files() {
        while true; do
            check_config_files
            check_script_files
            sleep 5
        done
    }
    
    # Check config files
    check_config_files() {
        local config_files=(/etc/sutazai/*.conf)
        for file in "${config_files[@]}"; do
            local current_hash=$(md5sum "$file" | awk '{print $1}')
            if [[ "${FILE_STATES[$file]}" != "$current_hash" ]]; then
                FILE_STATES["$file"]="$current_hash"
                $engine trigger_event "file_change" "$file"
            fi
        done
    }
    
    # Return instance methods
    echo "monitor_files"
} 