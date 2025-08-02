#!/bin/bash

# Kill any existing cpulimit processes
pkill -f cpulimit

# Limit main Cursor server process
cpulimit -p 1397 -l 25 &

# Limit extension host process
cpulimit -p 1429 -l 25 &

# Limit Pylance server process
cpulimit -p 1752 -l 25 &

echo "CPU usage has been limited to 25% for Cursor processes" 