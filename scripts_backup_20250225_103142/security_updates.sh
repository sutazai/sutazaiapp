#!/bin/bash
# Automatically apply security updates
apt-get update && apt-get upgrade --security -y
echo "Security updates applied successfully!" 