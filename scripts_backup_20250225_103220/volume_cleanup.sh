#!/bin/bash
# Clean up unused Docker volumes
docker volume prune -f
echo "Unused Docker volumes cleaned up successfully!" 