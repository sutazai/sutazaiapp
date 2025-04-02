#!/bin/bash

# Keep track of processes we've already handled
declare -A handled_pids

# Function to limit high-CPU processes
limit_high_cpu() {
    # Kill any existing cpulimit processes
    pkill -f cpulimit
    
    # Get all processes using more than 50% CPU
    high_cpu_pids=$(ps aux | grep -v grep | awk '{if($3 > 50) print $2}')
    
    for pid in $high_cpu_pids; do
        # Skip system processes and monitoring scripts
        if [[ $pid -gt 100 ]] && ! ps -p $pid | grep -q "monitor\|cpulimit"; then
            # Check if process is already being limited
            if ! ps aux | grep -v grep | grep -q "cpulimit.*$pid"; then
                echo "Limiting PID $pid to 50% CPU"
                cpulimit -p $pid -l 50 &
                # Lower process priority
                renice +19 $pid 2>/dev/null
            fi
        fi
    done
}

# Function to kill unnecessary high-CPU processes
kill_unnecessary() {
    # Kill all bash processes using high CPU that aren't Cursor-related
    high_cpu_bash=$(ps aux | grep bash | grep -v grep | grep -v "cursor-server" | grep -v "shellIntegration" | awk '{if($3 > 50) print $2}')
    
    for pid in $high_cpu_bash; do
        if ps -p $pid > /dev/null; then
            echo "Killing unnecessary high-CPU bash process: $pid"
            kill -9 $pid
            handled_pids[$pid]=1
        fi
    done
    
    # Kill any process using more than 80% CPU that isn't essential
    very_high_cpu=$(ps aux | grep -v grep | grep -v "cursor-server" | grep -v "shellIntegration" | awk '{if($3 > 80) print $2}')
    
    for pid in $very_high_cpu; do
        if ps -p $pid > /dev/null; then
            echo "Killing very high-CPU process: $pid"
            kill -9 $pid
            handled_pids[$pid]=1
        fi
    done
}

# Function to check and kill zombie processes
kill_zombies() {
    zombie_pids=$(ps aux | grep -v grep | awk '{if($8=="Z") print $2}')
    for pid in $zombie_pids; do
        echo "Killing zombie process: $pid"
        kill -9 $pid
        handled_pids[$pid]=1
    done
}

# Function to kill new high-CPU processes
kill_new_processes() {
    current_pids=$(ps aux | grep -v grep | awk '{print $2}')
    for pid in $current_pids; do
        if [[ ! ${handled_pids[$pid]} ]] && [[ $pid -gt 100 ]]; then
            cpu_usage=$(ps -p $pid -o %cpu | tail -n 1 | awk '{print $1}')
            if (( $(echo "$cpu_usage > 50" | bc -l) )); then
                echo "Killing new high-CPU process: $pid"
                kill -9 $pid
                handled_pids[$pid]=1
            fi
        fi
    done
}

# Function to lower priority of all non-essential processes
lower_priorities() {
    for pid in $(ps aux | grep -v grep | awk '{print $2}'); do
        if [[ $pid -gt 100 ]] && ! ps -p $pid | grep -q "monitor\|cpulimit\|systemd"; then
            renice +19 $pid 2>/dev/null
        fi
    done
}

# Kill any existing monitoring processes
pkill -f "monitor_cpu.sh"
pkill -f cpulimit
pkill -f bash

# Set system-wide CPU limit
echo "Setting system-wide CPU limit..."
echo 50 > /proc/sys/kernel/sched_rt_runtime_us

echo "Starting CPU monitoring..."

# Main monitoring loop
while true; do
    # Kill zombies first
    kill_zombies
    
    # Then kill unnecessary processes
    kill_unnecessary
    
    # Kill any new high-CPU processes
    kill_new_processes
    
    # Lower priorities of all non-essential processes
    lower_priorities
    
    # Then limit remaining high-CPU processes
    limit_high_cpu
    
    # Log current high-CPU processes
    echo "Current high-CPU processes:"
    ps aux | grep -v grep | awk '{if($3 > 50) print $2, $3, $11, $12, $13}'
    
    # If CPU usage is still too high, kill more processes
    if [ $(top -b -n 1 | grep "Cpu(s)" | awk '{print $2}' | cut -d. -f1) -gt 80 ]; then
        echo "CPU still too high, killing more processes..."
        kill_unnecessary
        kill_new_processes
        lower_priorities
    fi
    
    sleep 1  # Check every second
done 