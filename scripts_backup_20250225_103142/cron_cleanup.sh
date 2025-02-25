#!/bin/bash
# Clean up old cron jobs
crontab -l | grep -v "old_job" | crontab -
echo "Old cron jobs cleaned up successfully!" 