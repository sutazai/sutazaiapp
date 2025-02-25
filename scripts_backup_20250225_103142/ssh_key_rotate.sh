#!/bin/bash
# Rotate SSH keys
ssh-keygen -t rsa -b 4096 -f /root/.ssh/id_rsa -N ""
echo "SSH keys rotated successfully!" 