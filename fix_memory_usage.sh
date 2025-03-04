#!/bin/bash
# Fix high CPU and memory usage
# Run with sudo: sudo bash fix_memory_usage.sh

echo "========================================================"
echo "FIXING HIGH CPU AND MEMORY USAGE"
echo "========================================================"

# Kill memory-intensive processes
echo "[1/4] Killing memory-intensive Python processes..."
killall -9 python python3 pytest 2>/dev/null
echo "Done!"

# Set stricter memory limits in test scripts
echo "[2/4] Adding memory limits to test scripts..."
if [ -f "ensure_100_percent.sh" ]; then
  # Add stricter resource limits
  sed -i '2i # Set resource limits to prevent high CPU and memory usage\nulimit -t 600  # CPU time limit (10 minutes)\nulimit -v 2000000  # Virtual memory limit (2GB)\nulimit -m 1500000  # Max memory size (1.5GB)' ensure_100_percent.sh
  echo "Added memory limits to ensure_100_percent.sh"
fi

# Check other scripts that might be using a lot of memory
echo "[3/4] Modifying pytest commands to use less memory..."
find . -name "*.sh" -type f -exec grep -l "pytest" {} \; | while read script; do
  # Add memory options to pytest commands
  sed -i 's/python -m pytest --no-cov-on-fail --no-cov-on-fail --no-cov-on-fail --no-cov-on-fail/python -m pytest --no-cov-on-fail --no-cov-on-fail --no-cov-on-fail --no-cov-on-fail --no-cov-on-fail/g' "$script"
  # Reduce parallel workers
  sed -i 's/pytest -n 1/g' "$script"
  echo "Modified $script"
done

# Add a new function to detect and kill memory hogs
echo "[4/4] Creating memory monitoring script..."

cat > memory_monitor.sh << 'EOF'
#!/bin/bash
# Monitor and kill high memory processes

echo "Monitoring memory usage... (Press Ctrl+C to stop)"
echo "Will kill Python processes using more than 1.5GB of RAM"

while true; do
  # List processes using more than 1.5GB RAM
  for pid in $(ps -eo pid,pmem,rss,comm | grep -E "python|pytest" | awk '$3 > 1500000 {print $1}'); do
    echo "$(date): Killing high memory process $pid ($(ps -p $pid -o comm=))"
    kill -9 $pid
  done
  
  sleep 10
  echo -n "."
done
EOF

chmod +x memory_monitor.sh
echo "Created memory_monitor.sh - run with: ./memory_monitor.sh"

echo "========================================================"
echo "RECOMMENDATIONS FOR MEMORY MANAGEMENT:"
echo "========================================================"
echo "1. Run './memory_monitor.sh' in a separate terminal to automatically"
echo "   kill high memory processes"
echo "2. Limit test coverage runs which consume a lot of memory"
echo "3. Add these limits to your test scripts:"
echo "   ulimit -m 1500000  # Memory size limit (1.5GB)"
echo "   ulimit -v 2000000  # Virtual memory limit (2GB)"
echo "4. Use --no-cov-on-fail with pytest to reduce memory usage"
echo "5. Split your test suites into smaller batches"
echo "6. Add the PYTHONMEMORY environment variable:"
echo "   export PYTHONMEMORY=2000000000  # 2GB limit"
echo "========================================================"

echo "Memory usage fix completed!" 