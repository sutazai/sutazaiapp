#!/bin/bash

# Source the script to get the functions
source /opt/sutazaiapp/scripts/monitoring/live_logs.sh

# Override show_menu to prevent it from running
show_menu() {
    echo "Menu would be shown here"
}

# Set up colors
NC='\033[0m'
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'

# Call the function directly
show_unified_live_logs