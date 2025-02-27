#!/bin/bash
# Script to reorganize the SutazaiApp project directory structure
# Usage: ./scripts/reorganize_project.sh

set -e

# Define the root directory
ROOT_DIR="/opt/sutazaiapp"
cd "$ROOT_DIR" || exit 1

echo "Reorganizing SutazaiApp project directory structure..."

# Ensure all required directories exist
echo "Ensuring all required directories exist..."
mkdir -p ai_agents model_management backend web_ui scripts packages/wheels logs doc_data docs

# Move any Python requirements files to the packages directory if they're not already there
if [ -f "requirements.txt" ] && [ ! -f "packages/requirements.txt" ]; then
  echo "Moving requirements.txt to packages directory..."
  mv requirements.txt packages/
fi

if [ -f "get-pip.py" ] && [ ! -f "packages/get-pip.py" ]; then
  echo "Moving get-pip.py to packages directory..."
  mv get-pip.py packages/
fi

# Set correct ownership and permissions
echo "Setting correct ownership and permissions..."
sudo chown -R sutazaiapp_dev:sutazaiapp_dev "$ROOT_DIR"
chmod -R 750 "$ROOT_DIR"

echo "Directory reorganization complete!"
echo "Run the following to verify the directory structure:"
echo "ls -la $ROOT_DIR"

# Print reminder to update documentation
echo -e "\nReminder: Update the documentation in docs/DIRECTORY_STRUCTURE.md if needed."
echo "Also run the audit tools as specified in the project plan." 