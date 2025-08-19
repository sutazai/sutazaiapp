#!/bin/bash

# This script properly runs option 10 of the live logs menu

exec 3<&0  # Save stdin
exec < /dev/tty  # Redirect stdin from terminal

# Run the monitoring script with option 10
/opt/sutazaiapp/scripts/monitoring/live_logs.sh << EOF
10
EOF

exec 0<&3  # Restore stdin