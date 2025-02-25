#!/bin/bash
# Clean up unused Docker networks
docker network prune -f
echo "Unused Docker networks cleaned up successfully!" 