#!/bin/bash

# Initialize logging system
init_logging() {
    LOG_DIR="/var/log/sutazai"
    mkdir -p "$LOG_DIR"
    LOG_FILE="$LOG_DIR/sutazai.log"
    
    # Create log file if it doesn't exist
    touch "$LOG_FILE"
    
    # Set log levels
    declare -gA LOG_LEVELS=(
        ["DEBUG"]=0
        ["INFO"]=1
        ["WARN"]=2
        ["ERROR"]=3
    )
    
    # Default log level
    CURRENT_LOG_LEVEL=${LOG_LEVELS["INFO"]}
}

# Log function
log() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Check if level exists
    if [[ -z "${LOG_LEVELS[$level]}" ]]; then
        echo "Invalid log level: $level" >&2
        return 1
    fi
    
    # Only log if level is >= current log level
    if [[ ${LOG_LEVELS[$level]} -ge $CURRENT_LOG_LEVEL ]]; then
        echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
    fi
}

# Set log level
set_log_level() {
    local level=$1
    if [[ -z "${LOG_LEVELS[$level]}" ]]; then
        log "ERROR" "Invalid log level: $level"
        return 1
    fi
    CURRENT_LOG_LEVEL=${LOG_LEVELS[$level]}
    log "INFO" "Log level set to $level"
}

# Export functions
export -f init_logging log set_log_level 

handle_error() {
    local message=$1
    log "ERROR" "$message"
    echo -e "${RED}Error: $message${RESET}"
    echo -e "${YELLOW}Please check the logs at /var/log/sutazai/sutazai.log${RESET}"
    exit 1
} 