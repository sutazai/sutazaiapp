#!/bin/bash
# ULTRA SCRIPT DISCOVERY BOOTSTRAP
# Flexible script path resolution for migration compatibility

# Function to find script regardless of current location
find_script() {
    local script_name="$1"
    local search_paths=(
        "scripts/"
        "scripts/testing/"
        "scripts/monitoring/"  
        "scripts/security/"
        "scripts/deployment/"
        "scripts/maintenance/"
        "scripts/documentation/"
        "scripts/utils/"
        "scripts/mcp/"
        "scripts/automation/"
        "scripts/emergency_fixes/"
        "scripts/sync/"
        "./"
    )
    
    for path in "${search_paths[@]}"; do
        if [ -f "${path}${script_name}" ]; then
            echo "${path}${script_name}"
            return 0
        fi
    done
    
    # If not found, try a recursive search
    local found=$(find scripts -name "$script_name" 2>/dev/null | head -1)
    if [ -n "$found" ]; then
        echo "$found"
        return 0
    fi
    
    echo "ERROR: Script $script_name not found" >&2
    return 1
}

# Function to execute script with discovery
exec_script() {
    local script_name="$1"
    shift # Remove script name from args
    
    local script_path=$(find_script "$script_name")
    if [ $? -eq 0 ]; then
        if [[ "$script_path" == *.py ]]; then
            python3 "$script_path" "$@"
        else
            bash "$script_path" "$@"
        fi
    else
        exit 1
    fi
}

# Export functions for use by other scripts
export -f find_script
export -f exec_script
