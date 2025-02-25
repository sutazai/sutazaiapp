#!/bin/bash
# Python Security Environment Configuration

# Ensure randomized hash seed for security
export PYTHONHASHSEED=$(od -An -N4 -i /dev/urandom)

# Enable safe path for imports
export PYTHONSAFEPATH=1

# Disable bytecode generation for security
export PYTHONDONTWRITEBYTECODE=1

# Prevent loading user-specific site-packages
export PYTHONNOUSERSITE=1

# Add the script to system-wide environment
sudo tee /etc/profile.d/python_security.sh << EOF
export PYTHONHASHSEED=$PYTHONHASHSEED
export PYTHONSAFEPATH=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
EOF

# Make the script executable
sudo chmod +x /etc/profile.d/python_security.sh

echo "Python security environment variables configured." 