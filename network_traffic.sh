#!/bin/bash
# Analyze network traffic
ifstat -i eth0 1 5 > /var/log/network_traffic.log
echo "Network traffic analysis completed!" 