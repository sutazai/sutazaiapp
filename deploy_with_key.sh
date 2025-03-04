#!/bin/bash

# Script to deploy code to the deployment server using SSH key authentication
# No passwords needed - more secure!

echo "Starting deployment using SSH key authentication..."

# Directory to deploy
SOURCE_DIR="/opt/sutazaiapp"
TARGET="root@192.168.100.100:/opt/sutazaiapp"
SSH_KEY="/root/.ssh/sutazaiapp_sync_key"

# Sync the code (excluding unnecessary directories)
rsync -av --exclude=venv --exclude=__pycache__ --exclude=.git --exclude=.pytest_cache -e "ssh -i $SSH_KEY" $SOURCE_DIR/core_system $SOURCE_DIR/ai_agents $SOURCE_DIR/scripts $SOURCE_DIR/backend $SOURCE_DIR/tests $TARGET/

echo "Deployment complete!"
