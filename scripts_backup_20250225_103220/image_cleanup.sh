#!/bin/bash
# Clean up unused Docker images
docker image prune -a -f
echo "Unused Docker images cleaned up successfully!" 