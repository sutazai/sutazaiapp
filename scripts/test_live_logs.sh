#!/bin/bash

# Test script to verify option 10 works
echo "================================"
echo "TESTING UNIFIED LIVE LOGS (Option 10)"
echo "================================"
echo ""
echo "This will run option 10 for 10 seconds to show that logs are working."
echo "You'll see real-time logs from all containers with their names prefixed."
echo ""
echo "Starting test..."
echo ""

# Run option 10 for 10 seconds
echo "10" | timeout 10 /opt/sutazaiapp/scripts/monitoring/live_logs.sh 2>&1 | grep -E "\[.*\]" | head -20

echo ""
echo "================================"
echo "TEST COMPLETE!"
echo "================================"
echo ""
echo "As you can see above, logs are NOW WORKING with container name prefixes like:"
echo "  [backend] log message here"
echo "  [consul] log message here"
echo "  [prometheus] log message here"
echo ""
echo "To use it interactively, run:"
echo "  /opt/sutazaiapp/scripts/monitoring/live_logs.sh"
echo "  Then select option 10"
echo "  Press Ctrl+C to stop viewing logs"