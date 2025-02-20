#!/bin/bash
# Generate system inventory
echo "CPU: $(lscpu | grep "Model name")" > /var/log/system_inventory.log
echo "Memory: $(free -h | grep Mem)" >> /var/log/system_inventory.log
echo "Disk: $(lsblk)" >> /var/log/system_inventory.log 