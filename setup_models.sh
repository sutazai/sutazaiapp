#!/bin/bash
set -e

echo "Setting up model directories..."

# Create necessary directories
mkdir -p /opt/sutazaiapp/data/models

# Note for manual download
echo "Note: You need to manually download the models to /opt/sutazaiapp/data/models"
echo "      or modify deploy script to point to existing model location."

echo "Model directories created. Please copy models manually." 