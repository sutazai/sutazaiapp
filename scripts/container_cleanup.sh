#!/bin/bash
# Clean up stopped containers
docker container prune -f
echo "Stopped containers cleaned up successfully!" 