#!/bin/bash
# Optimize memory usage
sync; echo 3 > /proc/sys/vm/drop_caches
echo "Memory optimized successfully!" 