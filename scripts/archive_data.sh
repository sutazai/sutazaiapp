#!/bin/bash
# Archive old data
find /var/data -type f -mtime +90 -exec tar -czf /var/archives/archive_$(date +%Y%m%d).tar.gz {} +
echo "Data archived successfully!" 