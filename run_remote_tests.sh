#!/bin/bash

# Script to run tests on the deployment server using SSH key authentication
# No passwords needed - more secure!

echo "Running tests on deployment server using SSH key authentication..."

# SSH key to use
SSH_KEY="/root/.ssh/sutazaiapp_sync_key"
REMOTE_SERVER="root@192.168.100.100"

# Run the tests on the remote server
ssh -i $SSH_KEY $REMOTE_SERVER "cd /opt/sutazaiapp && source venv/bin/activate && ./scripts/run_tests.sh"

echo "Remote tests complete!" 