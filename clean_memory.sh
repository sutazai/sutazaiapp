#!/bin/bash
# Quick memory cleanup script

echo "Memory cleanup starting..."

# Kill memory-intensive Python processes
echo "Killing high memory Python processes..."
for pid in $(ps -eo pid,rss,comm | grep python | sort -nr -k2 | head -5 | awk '{print $1}'); do
  echo "Killing PID $pid (high memory usage)"
  kill -9 $pid 2>/dev/null
done

# Clear disk caches if running as root
if [ "$EUID" -eq 0 ]; then
  echo "Clearing disk caches..."
  sync
  echo 3 > /proc/sys/vm/drop_caches
  echo "Caches cleared"
fi

# Clear pytest cache
echo "Clearing pytest cache..."
rm -rf .pytest_cache
rm -rf .coverage .coverage.*
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete
echo "Python caches cleared"

# Clear swap if possible
if [ "$EUID" -eq 0 ]; then
  echo "Clearing swap..."
  swapoff -a && swapon -a
  echo "Swap cleared"
fi

echo "Memory cleanup complete!"
echo "Current memory status:"
free -h 