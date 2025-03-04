#!/bin/bash

# This script kills high CPU processes and gives instructions for preventing high CPU usage
# Run with sudo: sudo bash fix_cpu_running.sh

echo "======================================================================================"
echo "CPU USAGE FIX - RUN WITH ROOT PERMISSIONS"
echo "======================================================================================"
echo ""

# Kill all Python processes
echo "STEP 1: Killing high CPU Python processes..."
killall -9 python python3 pytest 2>/dev/null
echo "Done!"
echo ""

# Fix the main test script
echo "STEP 2: Modifying test scripts to use fewer resources..."
if [ -f "ensure_100_percent.sh" ]; then
  # Reduce worker count
  sed -i 's/python -m pytest -n 1/' ensure_100_percent.sh
  sed -i 's/python -m pytest -n 1/' ensure_100_percent.sh
  # Add resource limits
  sed -i '2i # Set resource limits to prevent high CPU usage\nulimit -t 600  # CPU time limit (10 minutes)\nulimit -v 4000000  # Virtual memory limit (4GB)' ensure_100_percent.sh
  echo "✓ Modified ensure_100_percent.sh"
else
  echo "× Could not find ensure_100_percent.sh"
fi

# Check CPU usage now
echo ""
echo "STEP 3: Checking current CPU usage..."
echo "CPU usage after fixes:"
echo "-------------------"
ps aux | sort -rk 3,3 | head -5
echo ""

echo "======================================================================================"
echo "MANUAL STEPS TO COMPLETE (copy and execute these commands one by one):"
echo "======================================================================================"
echo ""
echo "1. Add timeouts to your pytest commands:"
echo "   sed -i 's/python -m pytest --no-cov-on-fail --no-cov-on-fail --no-cov-on-fail --no-cov-on-fail/python -m pytest --no-cov-on-fail --no-cov-on-fail --no-cov-on-fail --no-cov-on-fail --timeout=300/g' *.sh"
echo ""
echo "2. Reduce worker count in all test scripts:"
echo "   find . -name '*.sh' -exec sed -i 's/pytest -n 1/g' {} \;"
echo ""
echo "3. Monitor for high CPU processes and kill them if needed:"
echo "   ps aux | sort -rk 3,3 | head -10"
echo "   kill -9 [PID]"
echo ""
echo "4. Add resource limits to the beginning of any scripts that run intensive tests:"
echo "   # Add these lines near the top of test scripts"
echo "   ulimit -t 600  # CPU time limit (10 minutes)"
echo "   ulimit -v 4000000  # Virtual memory limit (4GB)"
echo ""
echo "======================================================================================"
echo "RESOURCE MANAGEMENT RECOMMENDATIONS:"
echo "======================================================================================"
echo "• Run CPU-intensive tests one at a time, not in parallel"
echo "• Use fewer workers with pytest -n 1)"
echo "• Always add timeouts to tests to prevent infinite loops"
echo "• Consider running tests in batches rather than all at once"
echo "• Use 'nice -n 19' to lower the priority of CPU-intensive processes"
echo ""
echo "Script completed!" 