#!/bin/bash
# SutazAi Security Hardening Script

# File permissions
find /root/sutazai/v1 -type d -exec chmod 755 {} \;
find /root/sutazai/v1 -type f -exec chmod 644 {} \;
chmod 600 /root/sutazai/v1/security/*

# SELinux policies
semanage fcontext -a -t etc_t "/root/sutazai/v1/config(/.*)?"
restorecon -Rv /root/sutazai/v1

# Docker security
docker exec sutazai-core chmod 600 /security/jwt_secret.key 