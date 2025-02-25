#!/bin/bash

# System Performance Benchmarking Script

log_message "=== Starting System Performance Benchmark ==="

# CPU benchmark
log_message "Running CPU benchmark..."
cpu_score=$(sysbench cpu --cpu-max-prime=20000 run | grep "events per second" | awk '{print $4}')
log_message "CPU Score: $cpu_score events/sec"

# Memory benchmark
log_message "Running Memory benchmark..."
mem_score=$(sysbench memory run | grep "Operations performed" | awk '{print $4}')
log_message "Memory Score: $mem_score ops/sec"

# Disk I/O benchmark
log_message "Running Disk I/O benchmark..."
io_score=$(sysbench fileio --file-total-size=1G --file-test-mode=rndrw prepare run cleanup | grep "Requests/sec" | awk '{print $1}')
log_message "Disk I/O Score: $io_score req/sec"

log_message "=== System Performance Benchmark Completed ===" 