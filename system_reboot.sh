#!/bin/bash
# Reboot the system if uptime exceeds 30 days
UPTIME=$(awk '{print int($1/86400)}' /proc/uptime)
if (( $UPTIME > 30 )); then
    reboot
fi 