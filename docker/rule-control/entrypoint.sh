#!/bin/bash
set -e

echo "🧠 Starting Rule Control API..."

# Ensure config directory exists
mkdir -p /app/config

# Start the application
echo "🎯 Starting rule control server..."
exec python app.py --port 8100