#!/bin/bash
# Monitor active user sessions
who > /var/log/user_sessions.log
echo "User sessions monitored successfully!" 