#!/bin/bash

# Automated log integrity checking
AUTO_INTEGRITY() {
    echo "Starting automated log integrity check..."
    
    # Create checksums
    create_checksums() {
        for logfile in $(find /var/log -type f -name "*.log"); do
            sha256sum $logfile > $logfile.sha256
        done
    }
    
    # Verify checksums
    verify_checksums() {
        for checksum in $(find /var/log -type f -name "*.sha256"); do
            sha256sum -c $checksum
            if [ $? -ne 0 ]; then
                echo "Log integrity check failed for ${checksum%.sha256}" | mail -s "Log Integrity Alert" admin@example.com
            fi
        done
    }
    
    create_checksums
    verify_checksums
    echo "Log integrity check completed at $(date)" >> /var/log/log_integrity.log
}

echo "=== Log Integrity Check ==="
echo "Last 5 System Errors:"
journalctl -p 3 -xb -n 5