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
