#!/bin/bash
set -e

echo "Cleaning up deployment..."
docker-compose -f ai-stack.yml down -v
sudo rm -rf /var/lib/postgresql/data /var/lib/redis/data
echo "Cleanup complete!" 