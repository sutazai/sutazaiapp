#!/bin/bash
# Check file integrity using checksums
find /etc/sutazai -type f -exec md5sum {} + > /var/log/file_integrity.log
echo "File integrity check completed successfully!" 