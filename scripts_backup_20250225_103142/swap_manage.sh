#!/bin/bash
# Manage swap space
SWAP_USAGE=$(free | awk '/Swap/{print $3/$2 * 100.0}')
if (( $(echo "$SWAP_USAGE > 80" | bc -l) )); then
    swapon --all
    echo "Swap space activated!"
fi 