#!/bin/bash

# Fix Ollama health check by using a Python-based check instead of curl
# The Ollama container doesn't have curl but has Python

echo "Fixing Ollama health check..."

# Create a Python health check script
cat > /tmp/ollama_health.py << 'EOF'
#!/usr/bin/env python3
import sys
import urllib.request
import json

try:
    with urllib.request.urlopen('http://localhost:11434/api/tags', timeout=5) as response:
        data = json.loads(response.read())
        if 'models' in data:
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Failure
except Exception as e:
    sys.exit(1)  # Failure
EOF

# Copy the script into the container
docker cp /tmp/ollama_health.py sutazai-ollama:/health_check.py
docker exec sutazai-ollama chmod +x /health_check.py

# Test the health check
if docker exec sutazai-ollama python3 /health_check.py; then
    echo "✓ Ollama health check script works"
else
    echo "✗ Ollama health check script failed"
fi

# Update resource limits for Ollama (reduce from 23GB to 4GB)
echo "Optimizing Ollama memory allocation..."
docker update --memory="4g" --memory-swap="4g" sutazai-ollama 2>/dev/null || echo "Could not update memory limits"

echo "Ollama fix complete"