#!/bin/bash

# Kill any existing instances of this script
pkill -f "kill_bash_processes.sh"

echo "Starting bash process monitoring..."

while true; do
    # Find all bash processes using more than 20% CPU
    for pid in $(ps aux | grep bash | grep -v grep | grep -v "kill_bash_processes" | awk '{if($3 > 20.0) print $2}'); do
        # Skip Cursor terminals
        if ! ps -p $pid -o cmd= | grep -q "shellIntegration"; then
            echo "Killing high-CPU bash process: $pid"
            kill -9 $pid 2>/dev/null
        fi
    done
    
    # Sleep briefly
    sleep 2
done 