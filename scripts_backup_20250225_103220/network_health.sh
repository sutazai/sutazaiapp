#!/bin/bash
# Check network connectivity and latency
ping -c 4 google.com > /dev/null && echo "Network is healthy" || echo "Network issues detected" 