#!/bin/bash

# VS Code Sync Handler
analyze_code_patterns() {
    local file=$1
    
    # Check for duplicate code patterns
    local patterns_found=$(grep -E '^#DUPLICATE: |^#SIMILAR: ' "$file")
    if [[ -n "$patterns_found" ]]; then
        echo "⚠️  Code patterns detected in $file"
        trigger_event "code_pattern_detected" "$file"
    fi
}

process_file_change() {
    local changed_file=$1
    analyze_code_patterns "$changed_file"
    
    # Filter file types
    case "$changed_file" in
        *.sh|*.py|*.js|*.json)
            echo "Detected change in: $changed_file"
            ./scripts/auto_detection_engine.sh scan "$changed_file"
            ./deploy_all.sh sync
            ;;
        *)
            echo "Ignoring non-code file: $changed_file"
            ;;
    esac
}

# Main execution
process_file_change "$@" 