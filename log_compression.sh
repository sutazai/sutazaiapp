#!/bin/bash
# Compress old logs
find /var/log/sutazai -type f -mtime +7 -exec gzip {} \;
echo "Old logs compressed successfully!" 