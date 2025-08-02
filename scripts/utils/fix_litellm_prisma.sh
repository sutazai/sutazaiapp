#!/bin/bash

# Script to fix LiteLLM Prisma environment variable issue

echo "Fixing LiteLLM Prisma environment variable issue..."

# Find the prisma_client.py file
PRISMA_CLIENT_FILE="/usr/lib/python3.13/site-packages/litellm/proxy/db/prisma_client.py"

if [ -f "$PRISMA_CLIENT_FILE" ]; then
    echo "Found prisma_client.py at: $PRISMA_CLIENT_FILE"
    
    # Create a backup
    cp "$PRISMA_CLIENT_FILE" "${PRISMA_CLIENT_FILE}.backup"
    
    # Apply the fix using sed
    # Add env=os.environ.copy() to the subprocess.run call
    sed -i '/subprocess\.run(/,/check=True,/ {
        /check=True,/a\                        env=os.environ.copy(),
    }' "$PRISMA_CLIENT_FILE"
    
    echo "Fix applied successfully!"
    echo "Restarting LiteLLM container..."
    
else
    echo "Error: prisma_client.py not found at expected location"
    exit 1
fi