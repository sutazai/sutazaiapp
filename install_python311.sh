#!/bin/bash
# Install Python 3.11 on the deployment server

# Update package lists
apt-get update

# Install Python 3.11
apt-get install -y python3.11 python3.11-venv python3.11-dev

# Verify installation
python3.11 --version

# Create symbolic link
ln -sf /usr/bin/python3.11 /usr/bin/python3 