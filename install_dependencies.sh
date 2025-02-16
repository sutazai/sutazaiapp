#!/bin/bash
# Automatically install missing dependencies
DEPS=("docker" "docker-compose" "curl" "git")
for dep in "${DEPS[@]}"; do
    if ! command -v $dep &>/dev/null; then
        apt-get install -y $dep
        echo "Installed $dep"
    fi
done 