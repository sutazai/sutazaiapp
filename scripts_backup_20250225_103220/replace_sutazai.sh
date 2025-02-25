#!/bin/bash

# Enhanced pattern matching with additional sutazai-related terms
PATTERNS=(
    'SutazAi'
    'sutazai'
    'SUTAZAI'
    'q-'
    '_q_'
    'q_'
    'Qubit'
    'qubit'
    'QUBIT'
    'Qbit'
    'qbit'
    'QBIT'
)

# File content replacement with case-insensitive matching
for pattern in "${PATTERNS[@]}"; do
    find . -type f \
        ! -path "./.git/*" \
        ! -path "./.venv/*" \
        ! -path "./node_modules/*" \
        -exec sed -i "s/\b${pattern}\b/SutazAi/gi" {} +
done

# File/directory renaming with case-insensitive matching
for pattern in "${PATTERNS[@]}"; do
    # Files
    find . -type f -iname "*${pattern}*" | while read -r file; do
        new_name=$(echo "$file" | sed "s/${pattern}/SutazAi/gi")
        mv "$file" "$new_name"
    done
    
    # Directories
    find . -type d -iname "*${pattern}*" | while read -r dir; do
        new_name=$(echo "$dir" | sed "s/${pattern}/SutazAi/gi")
        mv "$dir" "$new_name"
    done
done

# Cleanup binary remnants
find . -type f -exec grep -Iq . {} \; -exec sed -i '/[Qq]\(uantum\|ubit\|bit\)/d' {} +

echo "All SutazAi and related references replaced with SutazAi"

# Final verification
if grep -rniI --exclude-dir=.git \
    --exclude-dir=.venv \
    --exclude-dir=node_modules \
    -e '[Qq]\(uantum\|uantum\|UANTUM\)' .; then
    echo "CRITICAL FAILURE: SutazAi remnants detected!" >&2
    exit 1
fi

# Verify replacement script
if grep -i 'sutazai' scripts/replace_sutazai.sh; then
    echo "SutazAi found in scripts/replace_sutazai.sh"
    exit 1
fi

# Replace SutazAi references with SutazAi
find /root/sutazai/v1 -type f -exec sed -i 's/SutazAi/SutazAi/g' {} +
find /root/sutazai/v1 -type f -exec sed -i 's/sutazai/sutazai/g' {} +
find /root/sutazai/v1 -type f -exec sed -i 's/SUTAZAI/SUTAZAI/g' {} +
find /root/sutazai/v1 -type f -exec sed -i 's/Qubit/Sutaz/g' {} +
find /root/sutazai/v1 -type f -exec sed -i 's/qubit/sutaz/g' {} +
find /root/sutazai/v1 -type f -exec sed -i 's/QUBIT/SUTAZ/g' {} +

# Use sed with -i.bak for backups and find with -exec for efficiency
find . -type f -exec sed -i.bak 's/SutazAi/SutazAi/g' {} +

# Replace 'SutazAi' with 'SutazAi' in the specified lines
find . -type f -exec sed -i 's/SutazAi/SutazAi/g' {} +
find . -type f -exec sed -i 's/sutazai/sutazai/g' {} +
find . -type f -exec sed -i 's/SUTAZAI/SUTAZAI/g' {} +

# Enhanced pattern matching with additional sutazai-related terms
if grep -i 'sutazai' scripts/replace_sutazai.sh; then
    echo "SutazAi found in scripts/replace_sutazai.sh"
    find /root/sutazai/v1 -type f -exec sed -i 's/sutazai/sutazai/g' {} +
    find /root/sutazai/v1 -type f -exec sed -i 's/SUTAZAI/SUTAZAI/g' {} + 