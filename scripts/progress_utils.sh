#!/bin/bash

# Advanced progress bar system with real-time monitoring and adaptive coloring

# Function to get system metrics
get_system_metrics() {
    local metrics=()
    
    # CPU usage
    metrics+=($(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}'))
    
    # Memory usage
    metrics+=($(free -m | awk 'NR==2{printf "%.1f", $3*100/$2 }'))
    
    # Disk I/O
    metrics+=($(iostat -d | awk 'NR==4{print $2}'))
    
    # Network usage
    metrics+=($(ifstat 1 1 | awk 'NR==3{print $1}'))
    
    echo "${metrics[@]}"
}

# Function to calculate adaptive colors
calculate_adaptive_colors() {
    local metrics=($(get_system_metrics))
    local cpu=${metrics[0]}
    local mem=${metrics[1]}
    local disk=${metrics[2]}
    local net=${metrics[3]}
    
    # Calculate colors based on system load
    local color1=$((21 + (cpu * 200 / 100)))
    local color2=$((21 + (mem * 200 / 100)))
    local color3=$((21 + (disk * 200 / 100)))
    local color4=$((21 + (net * 200 / 100)))
    
    echo "$color1 $color2 $color3 $color4"
}

# Function to display system stats
display_system_stats() {
    local colors=($(calculate_adaptive_colors))
    
    printf "\n\e[38;5;%dmCPU: %3d%%\e[0m | " ${colors[0]} $(get_system_metrics | awk '{print $1}')
    printf "\e[38;5;%dmMEM: %3d%%\e[0m | " ${colors[1]} $(get_system_metrics | awk '{print $2}')
    printf "\e[38;5;%dmDISK: %3dMB/s\e[0m | " ${colors[2]} $(get_system_metrics | awk '{print $3}')
    printf "\e[38;5;%dmNET: %3dKB/s\e[0m\n" ${colors[3]} $(get_system_metrics | awk '{print $4}')
}

# Advanced progress bar with multiple layers
advanced_progress_bar() {
    local label1=$1
    local label2=$2
    local label3=$3
    local total=100
    local current=0
    
    # Create multiple progress layers
    while IFS= read -r line; do
        current=$((current + 1))
        percent=$((current * 100 / total))
        
        # Get adaptive colors
        local colors=($(calculate_adaptive_colors))
        
        # Main progress bar
        printf "\r\e[38;5;%dm%s: \e[38;5;%dm[%-50s]\e[38;5;%dm %d%%\e[0m\n" \
            ${colors[0]} "$label1" ${colors[1]} \
            "$(printf '█%.0s' $(seq 1 $((percent/2))))$(printf ' %.0s' $(seq 1 $((50 - percent/2))))" \
            ${colors[2]} "$percent"
            
        # Secondary progress bar
        printf "\r\e[38;5;%dm%s: \e[38;5;%dm[%-50s]\e[38;5;%dm %d%%\e[0m\n" \
            ${colors[1]} "$label2" ${colors[2]} \
            "$(printf '▄%.0s' $(seq 1 $((percent/2))))$(printf ' %.0s' $(seq 1 $((50 - percent/2))))" \
            ${colors[3]} "$percent"
            
        # Tertiary progress bar
        printf "\r\e[38;5;%dm%s: \e[38;5;%dm[%-50s]\e[38;5;%dm %d%%\e[0m\n" \
            ${colors[2]} "$label3" ${colors[3]} \
            "$(printf '▀%.0s' $(seq 1 $((percent/2))))$(printf ' %.0s' $(seq 1 $((50 - percent/2))))" \
            ${colors[0]} "$percent"
        
        # Display system stats
        display_system_stats
        
        # Move cursor up 4 lines for the next update
        printf "\033[4A"
        
        # Add smooth animation delay
        sleep 0.05
    done
    
    # Move cursor down 4 lines after completion
    printf "\033[4B\n"
    
    # Add completion animation
    for i in {1..3}; do
        printf "\r\e[38;5;46m✓\e[0m "
        sleep 0.1
    done
    echo
}

# Function to create a spinning animation
spinner() {
    local pid=$1
    local delay=0.1
    local spinstr='|/-\'
    
    while [ "$(ps a | awk '{print $1}' | grep $pid)" ]; do
        local temp=${spinstr#?}
        printf "\r\e[38;5;51m[%c]\e[0m " "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
    done
    printf "\r   \r"
}

# Function to display a loading screen
loading_screen() {
    clear
    display_logo
    echo -e "\n\e[1;38;5;226mInitializing deployment...\e[0m"
    
    # Create a background process for the spinner
    sleep 10 &
    spinner $!
}

# Function to display a completion animation
completion_animation() {
    local frames=("⠋" "⠙" "⠹" "⠸" "⠼" "⠴" "⠦" "⠧" "⠇" "⠏")
    local colors=(196 202 208 214 220 226 190 154 118 82)
    
    for i in {1..10}; do
        for frame in "${frames[@]}"; do
            printf "\r\e[38;5;${colors[$i]}m%s\e[0m Completing deployment..." "$frame"
            sleep 0.1
        done
    done
    echo
} 