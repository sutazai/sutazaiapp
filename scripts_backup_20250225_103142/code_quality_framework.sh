#!/bin/bash

# Configuration
CONFIG_FILE=".code_quality.conf"
WHITELIST_DIRS=("node_modules" ".git")
MIN_SIMILARITY_PERCENT=85
MAX_DUPLICATE_SIZE_KB=100

# Load user configuration if exists
[[ -f "$CONFIG_FILE" ]] && source "$CONFIG_FILE"

# Initialize results
declare -A ANALYSIS_RESULTS
declare -a FOUND_ISSUES

# Register a new analysis module
register_analysis_module() {
    local module_name=$1
    local check_function=$2
    ANALYSIS_RESULTS["$module_name"]=$check_function
}

# Check for duplicate files using checksum
check_duplicate_files() {
    echo "🔍 Scanning for duplicate files..."
    local duplicates=0
    
    # Generate checksum index
    declare -A checksum_index
    while IFS= read -r -d '' file; do
        [[ " ${WHITELIST_DIRS[@]} " =~ " $(basename "$file") " ]] && continue
        
        checksum=$(md5sum "$file" | awk '{print $1}')
        if [[ -n "${checksum_index[$checksum]}" ]]; then
            FOUND_ISSUES+=("DUPLICATE: ${checksum_index[$checksum]} <-> $file")
            ((duplicates++))
        else
            checksum_index["$checksum"]="$file"
        fi
    done < <(find . -type f -print0)
    
    [[ $duplicates -gt 0 ]] && return 1 || return 0
}

# Check for similar code blocks using jscpd
check_similar_code() {
    command -v jscpd >/dev/null || return 0
    echo "🧬 Checking code similarity..."
    
    local output_file="/tmp/jscpd_report.json"
    jscpd --languages sh,bash --output "$output_file" --format json --min-tokens 50 --reporters json .
    
    # Process results
    local clones=$(jq -r ".duplicates | length" "$output_file")
    for ((i=0; i<clones; i++)); do
        local file1=$(jq -r ".duplicates[$i].firstFile.name" "$output_file")
        local file2=$(jq -r ".duplicates[$i].secondFile.name" "$output_file")
        local similarity=$(jq -r ".duplicates[$i].percentage" "$output_file")
        
        if (( $(echo "$similarity >= $MIN_SIMILARITY_PERCENT" | bc -l) )); then
            FOUND_ISSUES+=("SIMILAR_CODE: $file1 ↔ $file2 (${similarity}% similar)")
        fi
    done
    
    [[ $clones -gt 0 ]] && return 1 || return 0
}

# Main analysis runner
run_quality_checks() {
    local exit_code=0
    
    # Run registered modules
    for module in "${!ANALYSIS_RESULTS[@]}"; do
        echo -e "\n=== Running $module check ==="
        if ! ${ANALYSIS_RESULTS[$module]}; then
            ((exit_code++))
        fi
    done

    # Print summary
    echo -e "\n🔎 Found ${#FOUND_ISSUES[@]} issues:"
    printf '%s\n' "${FOUND_ISSUES[@]}"
    
    return $exit_code
}

# Register default modules
register_analysis_module "FileDuplicates" check_duplicate_files
register_analysis_module "CodeSimilarity" check_similar_code

# Example usage:
# run_quality_checks
# exit $? 