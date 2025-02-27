#!/bin/bash
# Comprehensive update system

# Create update script
cat > /usr/local/bin/update_sutazai <<EOF
#!/bin/bash
set -e

# Pull latest images
docker-compose -f docker-compose.yml pull
docker-compose -f docker-compose-super.yml pull
docker-compose -f docker-compose-ai.yml pull

# Restart services
docker-compose -f docker-compose.yml up -d
docker-compose -f docker-compose-super.yml up -d
docker-compose -f docker-compose-ai.yml up -d

# Run health checks
./health_check.sh

echo "System updated successfully!"
EOF

# Make executable
chmod +x /usr/local/bin/update_sutazai

# Schedule weekly updates
(crontab -l 2>/dev/null; echo "0 3 * * 0 /usr/local/bin/update_sutazai") | crontab -

echo "Update system configured successfully!" 