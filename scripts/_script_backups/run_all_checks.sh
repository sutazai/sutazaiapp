#!/bin/bash
set -euo pipefail

echo "Starting comprehensive system checks for SutazAI..."

# Run the Audit System to perform a detailed check of directories, files, dependencies, and syntax
echo "Step 1: Running Audit System..."
python3 scripts/audit_system.py

# Run the File Organization script to move unrecognized files to the 'misc' directory
echo "Step 2: Running File Organization..."
python3 scripts/organize_project.py

# Run the Auto-Fix script to create missing directories and initialize the virtual environment if needed
echo "Step 3: Running Auto-Fix..."
python3 scripts/auto_fix.py

echo "Completed all system checks. Please review the logs in the 'logs' directory for details." 