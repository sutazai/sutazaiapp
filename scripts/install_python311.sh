#!/bin/bash
# Script to remove Python 3.12 and install Python 3.11 on Ubuntu
# Created as part of the SutazAI codebase optimization

set -e  # Exit on error
echo "Starting Python 3.11 installation script..."

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root"
  exit 1
fi

# Add deadsnakes PPA for Python 3.11
echo "Adding deadsnakes PPA for Python 3.11..."
apt-get update
apt-get install -y software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update

# Install Python 3.11 and related packages
echo "Installing Python 3.11 and related packages..."
apt-get install -y python3.11 python3.11-venv python3.11-dev python3.11-distutils

# Create symbolic links to make Python 3.11 the default if requested
echo "Do you want to make Python 3.11 the default python3? (y/n)"
read -p "> " make_default

if [[ "$make_default" == "y" || "$make_default" == "Y" ]]; then
  # Check if python3 is configured with alternatives
  if update-alternatives --list python3 > /dev/null 2>&1; then
    # Remove existing alternatives
    echo "Removing existing Python alternatives..."
    update-alternatives --remove-all python3 || true
  fi
  
  # Add Python 3.11 as an alternative
  echo "Setting up Python 3.11 as default using update-alternatives..."
  update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
  update-alternatives --set python3 /usr/bin/python3.11
  
  # Install pip for Python 3.11
  echo "Installing pip for Python 3.11..."
  curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
  
  # Create symbolic link for pip3
  if [ -f /usr/bin/pip3 ]; then
    mv /usr/bin/pip3 /usr/bin/pip3.bak
  fi
  ln -sf /usr/local/bin/pip3.11 /usr/bin/pip3
fi

# Verify installation
echo "Verifying Python 3.11 installation..."
python3 --version
pip3 --version

echo "Python 3.11 installation completed successfully!"
echo "To activate Python 3.11 in your current shell, run: exec $SHELL"

# Setup virtual environment for the SutazAI application
echo "Do you want to create a Python 3.11 virtual environment for SutazAI? (y/n)"
read -p "> " create_venv

if [[ "$create_venv" == "y" || "$create_venv" == "Y" ]]; then
  echo "Creating virtual environment..."
  cd /opt/sutazaiapp
  python3.11 -m venv venv
  echo "Virtual environment created. Activate it with: source /opt/sutazaiapp/venv/bin/activate"
  echo "Then install requirements with: pip install -r /opt/sutazaiapp/requirements.txt"
fi

echo "Installation and setup complete!" 