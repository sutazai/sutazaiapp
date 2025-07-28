#!/bin/bash

# Extract just the function definition for testing
clean_duplicate_port_mappings_from_env() {
    if [ ! -f ".env" ]; then
        return 0
    fi
    
    # Create temporary file without duplicate port mapping sections
    local temp_env=$(mktemp)
    local in_sutazai_section=false
    
    while IFS= read -r line; do
        # Check if this line starts a SutazAI port mapping section
        if [[ "$line" =~ ^#.*SutazAI.*Dynamic.*Port.*Mappings ]]; then
            in_sutazai_section=true
            continue  # Skip the comment line
        fi
        
        # Check if we're in a port mapping section and this is a port mapping line
        if [ "$in_sutazai_section" = true ] && [[ "$line" =~ ^SUTAZAI_PORT_.*_MAPPED= ]]; then
            continue  # Skip duplicate port mapping lines
        fi
        
        # If we hit a non-port-mapping line, we're out of the section
        if [ "$in_sutazai_section" = true ] && [[ ! "$line" =~ ^SUTAZAI_PORT_.*_MAPPED= ]] && [[ ! "$line" =~ ^[[:space:]]*$ ]]; then
            in_sutazai_section=false
        fi
        
        # Keep all other lines
        echo "$line" >> "$temp_env"
    done < .env
    
    # Replace original .env with cleaned version
    mv "$temp_env" .env
    
    # Remove any trailing empty lines that might have been created
    sed -i -e :a -e '/^\s*$/N;ba' -e 's/\n\s*$//' .env
}

echo "Before cleanup:"
grep -n "SutazAI Dynamic Port Mappings" .env

echo -e "\nRunning cleanup function..."
clean_duplicate_port_mappings_from_env

echo -e "\nAfter cleanup:"
grep -n "SutazAI Dynamic Port Mappings" .env

echo -e "\nTotal lines in .env:"
wc -l .env