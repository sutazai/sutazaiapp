#\!/bin/bash
cd /opt/sutazaiapp
if [ -f system_monitor.log ] && [  -gt 1048576 ]; then
    mv system_monitor.log system_monitor.log.20250718
    touch system_monitor.log
    find . -name "system_monitor.log.*" -mtime +7 -delete
fi
