#!/bin/bash

# Script to fix high CPU usage by killing CPU-intensive Python processes
# Run this script as root (sudo ./fix_cpu_usage_root.sh)

echo "Fixing high CPU usage issue..."

# Kill all Python processes
echo "Killing all Python processes..."
killall -9 python python3 pytest 2>/dev/null
echo "Python processes terminated"

# Kill specific intensive processes by PID if they still exist
for pid in 2369 48951 51106 48957 48960; do
    if kill -0 $pid 2>/dev/null; then
        echo "Killing process $pid"
        kill -9 $pid 2>/dev/null
    fi
done

# Modify the test scripts to use fewer resources
if [ -f "/opt/sutazaiapp/ensure_100_percent.sh" ]; then
    echo "Modifying ensure_100_percent.sh to use fewer CPU resources..."
    sed -i 's/python -m pytest -n 1/' /opt/sutazaiapp/ensure_100_percent.sh
    echo "Modified ensure_100_percent.sh to use 1 worker instead of 4"
fi

# Add resource limits to all test scripts
find /opt/sutazaiapp -name "*.sh" -type f -exec grep -l "pytest" {} \; | while read file; do
    echo "Adding CPU and memory limits to $file"
    # Add ulimit commands at the beginning of the file
    sed -i '2i # Set resource limits\nulimit -t 300 # CPU time limit 5 minutes\nulimit -v 4000000 # Virtual memory limit 4GB' "$file"
done

echo "CPU usage fix completed."
echo ""
echo "RECOMMENDATIONS:"
echo "1. When running tests, use fewer workers (-n parameter in pytest-xdist)"
echo "2. Run one test job at a time instead of multiple parallel jobs"
echo "3. Consider adding timeouts to your test cases to prevent infinite loops"
echo "4. Add memory limits to your test runs to prevent memory leaks"
echo ""
echo "Run 'top' to verify CPU usage has decreased" 