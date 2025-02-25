#!/bin/bash
echo "Generating credentials in /root/sutazai/v1..."

# Ensure security directory exists
mkdir -p /root/sutazai/v1/security/

# Generate credentials
head -c 256 /dev/urandom > /root/sutazai/v1/security/docker_creds.gpg

# Generate checksum
cd /root/sutazai/v1
sha256sum security/docker_creds.gpg > security/docker_creds.sha256

echo "🔑 Generated placeholder credentials for development"
exit 0
