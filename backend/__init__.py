"""
SutazAI Backend Package
This package contains all the backend components including API, models, and services.
"""

import sys
import os
import logging

# Add project root to path to allow module discovery
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Import essential modules to make them available at the package level
# Removed unused imports: database, config, security

# Backend module initialization file
# This is the main backend package for SutazAI

# You can add initialization code here if needed
# For example, setting up logging, configurations, etc.


# Example: Configure logging for the backend module
logging.getLogger(__name__).addHandler(logging.NullHandler())
