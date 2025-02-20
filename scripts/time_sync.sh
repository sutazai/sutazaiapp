#!/bin/bash
# Synchronize system time with NTP
timedatectl set-ntp true
systemctl restart systemd-timesyncd
echo "Time synchronization configured successfully!" 