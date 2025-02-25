#!/bin/bash

# Ensure script is run with root privileges
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root" 
   exit 1
fi

# Services to enable
SERVICES=(
    "sutazai-file-structure-manager"
    "sutazai-system-integration"
    "sutazai-system-health-monitor"
    "sutazai-project-optimizer"
    "sutazai-auto-remediation"
    "sutazai-master"
)

# Enable and start services
for service in "${SERVICES[@]}"; do
    systemctl enable "${service}.service"
    systemctl start "${service}.service"
    echo "Enabled and started ${service} service"
done

# Reload systemd to recognize new services
systemctl daemon-reload

echo "SutazAI services have been set up and started successfully!"