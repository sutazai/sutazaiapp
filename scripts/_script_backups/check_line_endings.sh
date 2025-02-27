#!/bin/bash

# Check for Windows line endings
check_line_endings() {
    local files=(
        .deployment_config
        scripts/*.sh
        *.sh
    )
    
    local has_errors=0
    
    for file in "${files[@]}"; do
        if grep -q $'\r' "$file"; then
            echo "ERROR: Windows line endings found in $file"
            has_errors=1
        fi
    done
    
    if (( has_errors )); then
        echo "To fix, run: find . -type f -name '*.sh' -exec dos2unix {} \;"
        return 1
    fi
    
    echo "All files have proper Unix line endings"
    return 0
}

# Main execution
if [[ "$1" == "--fix" ]]; then
    find . -type f -name '*.sh' -exec dos2unix {} \;
    dos2unix .deployment_config
    echo "Fixed line endings in all files"
else
    check_line_endings
fi 