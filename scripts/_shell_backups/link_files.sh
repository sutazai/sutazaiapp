#!/bin/bash
set -euo pipefail

# Source the utility script
source "$(dirname "$0")/deploy_utils.sh"

# Define link mappings
declare -A LINK_MAPPINGS=(
    ["deploy_all.sh"]="/usr/local/bin/sutazai-deploy"
    ["scripts/deploy_utils.sh"]="/usr/local/lib/sutazai/deploy_utils.sh"
    ["scripts/link_files.sh"]="/usr/local/lib/sutazai/link_files.sh"
    ["deploy_ai.sh"]="/usr/local/lib/sutazai/deploy_ai.sh"
    ["deploy_sutazai.sh"]="/usr/local/lib/sutazai/deploy_sutazai.sh"
    ["requirements.txt"]="/usr/local/etc/sutazai/requirements.txt"
    ["ai-stack.yml"]="/usr/local/etc/sutazai/ai-stack.yml"
    ["scripts/"]="/usr/local/lib/sutazai/scripts"
)

# Create necessary directories
print_status "Creating directories..."
mkdir -p /usr/local/bin
mkdir -p /usr/local/lib/sutazai
mkdir -p /usr/local/etc/sutazai
mkdir -p /usr/local/lib/sutazai/scripts

# Create symlinks
print_status "Creating symlinks..."
for source in "${!LINK_MAPPINGS[@]}"; do
    target="${LINK_MAPPINGS[$source]}"
    if [[ -e "$target" ]]; then
        print_warning "Link already exists: $target"
    else
        ln -s "$(pwd)/$source" "$target"
        print_success "Created link: $source -> $target"
    fi
done

# Link entire scripts directory
if [[ ! -e "/usr/local/lib/sutazai/scripts" ]]; then
    ln -s "$(pwd)/scripts" "/usr/local/lib/sutazai/scripts"
    print_success "Created scripts directory link"
else
    print_warning "Scripts directory link already exists"
fi

# Verify links
print_status "Verifying links..."
for source in "${!LINK_MAPPINGS[@]}"; do
    target="${LINK_MAPPINGS[$source]}"
    if [[ ! -L "$target" ]]; then
        handle_error "Failed to create link: $target"
    fi
done

print_success "All files linked successfully" 