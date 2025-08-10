#!/bin/bash

# Strict error handling
set -euo pipefail

# SutazAI Archive Creation Script
# Creates a complete backup/archive of the SutazAI project
# Excludes temporary files, virtual environments, and large model files

# Navigate to the project root directory

# Signal handlers for graceful shutdown
cleanup_and_exit() {
    local exit_code="${1:-0}"
    echo "Script interrupted, cleaning up..." >&2
    # Clean up any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    exit "$exit_code"
}

trap 'cleanup_and_exit 130' INT
trap 'cleanup_and_exit 143' TERM
trap 'cleanup_and_exit 1' ERR

cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get timestamp for the archive name
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ARCHIVE_NAME="sutazai_backup_${TIMESTAMP}.tar.gz"
ARCHIVE_DIR="${PROJECT_ROOT}/backups"
FULL_PATH="${ARCHIVE_DIR}/${ARCHIVE_NAME}"

# Create backups directory if it doesn't exist
mkdir -p "$ARCHIVE_DIR"

# Show a spinner animation during archive creation
spinner() {
    local pid=$1
    local delay=0.1
    local spinstr='|/-\'
    while [ "$(ps a | awk '{print $1}' | grep $pid)" ]; do
        local temp=${spinstr#?}
        printf "[%c]  " "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\b\b\b\b\b"
    done
    printf "    \b\b\b\b"
}

echo "Creating SutazAI project archive..."
echo "Archive will be saved to: $FULL_PATH"
echo

# Ask user if they want to include model files (which can be large)
read -p "Include model files? These can be very large. (y/n): " INCLUDE_MODELS
if [[ "$INCLUDE_MODELS" =~ ^[Yy]$ ]]; then
    echo "Including model files in the archive."
    MODEL_EXCLUDE=""
else
    echo "Excluding model files from the archive."
    MODEL_EXCLUDE="--exclude='*.bin' --exclude='*.gguf' --exclude='*.onnx' --exclude='model_management/*/'"
fi

# Ask if user wants to include logs
read -p "Include log files? (y/n): " INCLUDE_LOGS
if [[ "$INCLUDE_LOGS" =~ ^[Yy]$ ]]; then
    echo "Including log files in the archive."
    LOG_EXCLUDE=""
else
    echo "Excluding log files from the archive."
    LOG_EXCLUDE="--exclude='logs/*'"
fi

# Create exclude patterns for temporary files and directories
EXCLUDES="--exclude='venv' --exclude='__pycache__' --exclude='*.pyc' \
          --exclude='.git' --exclude='.pytest_cache' --exclude='*.tmp' \
          --exclude='node_modules' --exclude='monitoring/data/*' \
          --exclude='.ipynb_checkpoints' --exclude='*.log' \
          --exclude='dist' --exclude='build' \
          --exclude='*.egg-info' $MODEL_EXCLUDE $LOG_EXCLUDE"

# Create a list of files to be included in the archive
echo "Creating file list..."
TEMP_FILELIST="${PROJECT_ROOT}/.archive_filelist.tmp"
find "$PROJECT_ROOT" -type f -not -path "*/\.*" -not -path "*/venv/*" \
     -not -path "*/__pycache__/*" -not -path "*/node_modules/*" \
     -not -path "*/monitoring/data/*" > "$TEMP_FILELIST"

# Show number of files to be archived
FILE_COUNT=$(wc -l < "$TEMP_FILELIST")
echo "Found $FILE_COUNT files to archive."

# Create the archive command
ARCHIVE_CMD="tar czf \"$FULL_PATH\" $EXCLUDES -C \"$PROJECT_ROOT\" ."

# Add manifest file
echo "# SutazAI Project Archive" > "${PROJECT_ROOT}/.archive_manifest.txt"
echo "# Created: $(date)" >> "${PROJECT_ROOT}/.archive_manifest.txt"
echo "# Files: $FILE_COUNT" >> "${PROJECT_ROOT}/.archive_manifest.txt"
echo "# Excluded: venv, __pycache__, node_modules, monitoring/data" >> "${PROJECT_ROOT}/.archive_manifest.txt"
if [[ -n "$MODEL_EXCLUDE" ]]; then
    echo "# Model files excluded" >> "${PROJECT_ROOT}/.archive_manifest.txt"
fi
if [[ -n "$LOG_EXCLUDE" ]]; then
    echo "# Log files excluded" >> "${PROJECT_ROOT}/.archive_manifest.txt"
fi
echo "" >> "${PROJECT_ROOT}/.archive_manifest.txt"
echo "## Project Structure" >> "${PROJECT_ROOT}/.archive_manifest.txt"
find "$PROJECT_ROOT" -type d -not -path "*/\.*" -not -path "*/venv/*" \
     -not -path "*/__pycache__/*" -not -path "*/node_modules/*" \
     -not -path "*/monitoring/data/*" | sed "s|$PROJECT_ROOT||" | sort >> "${PROJECT_ROOT}/.archive_manifest.txt"

# Start the archiving process
echo "Creating archive... (this may take a while)"
# SECURITY FIX: eval replaced
# Original: eval $ARCHIVE_CMD
"${ARCHIVE_CMD}" &
ARCHIVE_PID=$!
spinner $ARCHIVE_PID

wait $ARCHIVE_PID
ARCHIVE_EXIT_CODE=$?

# Clean up temporary files
rm -f "$TEMP_FILELIST" "${PROJECT_ROOT}/.archive_manifest.txt"

# Check if archive was created successfully
if [ $ARCHIVE_EXIT_CODE -eq 0 ] && [ -f "$FULL_PATH" ]; then
    # Get archive size
    ARCHIVE_SIZE=$(du -h "$FULL_PATH" | cut -f1)
    
    echo -e "${GREEN}Archive created successfully!${NC}"
    echo "Location: $FULL_PATH"
    echo "Size: $ARCHIVE_SIZE"
    echo 
    echo "To extract this archive, run:"
    echo "  mkdir sutazai_extracted"
    echo "  tar -xzf $ARCHIVE_NAME -C sutazai_extracted"
    
    # Log the archive creation
    mkdir -p "$PROJECT_ROOT/logs"
    echo "[$(date)] - Created archive $ARCHIVE_NAME (size: $ARCHIVE_SIZE)" >> "$PROJECT_ROOT/logs/backup.log"
else
    echo -e "${RED}Failed to create archive. Please check the error messages above.${NC}"
    exit 1
fi

exit 0
