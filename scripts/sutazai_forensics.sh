#!/bin/bash

# Find textual remnants
grep -rniI --color=always \
    --exclude-dir=.git \
    --exclude-dir=.venv \
    --exclude-dir=node_modules \
    -e '[Qq]\(uantum\|uantum\|UANTUM\)' .

# Find filename remnants
find . -iname "*sutazai*" -print

# Verify environment variables
env | grep -i sutazai

# Check running processes
ps aux | grep -i sutazai

# Verify complete eradication
if [ $? -eq 1 ]; then
    echo "System clean of sutazai contamination"
else
    echo "SUTAZAI DETECTED IN ACTIVE MEMORY!" >&2
    exit 1
fi 