#!/bin/bash

# Navigate to application directory
cd /opt/sutazaiapp

# Pull the latest code (assuming a git repository is used)
git pull origin main

# Activate the virtual environment and run any database migrations if needed
source /opt/sutazaiapp/venv-sutazaiapp/bin/activate
flask db upgrade

echo "Deployment completed."