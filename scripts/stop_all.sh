#!/bin/bash
# title        :stop_all.sh
# description  :This script stops all SutazAI components (Backend, SuperAGI, Web UI, Vector Store, Monitoring)
# author       :SutazAI Team
# version      :2.0
# usage        :sudo bash scripts/stop_all.sh
# notes        :Requires bash 4.0+ and standard Linux utilities

# Navigate to the project root directory
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# Save original terminal settings
ORIGINAL_STTY=$(stty -g 2>/dev/null || echo "")

# Set strict terminal width to avoid text wrapping
export TERM=linux
stty cols 80 2>/dev/null || true

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
BOLD='\033[1m'
DIM='\033[2m'
RESET='\033[0m'
NC='\033[0m' # No Color

# Cursor control
CURSOR_UP='\033[1A'
CURSOR_DOWN='\033[1B'
CURSOR_HIDE='\033[?25l'
CURSOR_SHOW='\033[?25h'
CLEAR_LINE='\033[2K'
CLEAR_SCREEN='\033[2J\033[H'

# Box drawing characters
BOX_TL="┌"
BOX_TR="┐"
BOX_BL="└"
BOX_BR="┘"
BOX_V="│"
BOX_H="─"
BOX_VR="├"
BOX_VL="┤"
BOX_HU="┬"
BOX_HD="┴"
BOX_VH="┼"

# Set to 1 for verbose output
VERBOSE=0

# Fixed widths for formatting
MAIN_WIDTH=77
INNER_WIDTH=$((MAIN_WIDTH - 4))
LEFT_COLUMN=25
STATUS_COLUMN=20

# Control output flow
USE_SPINNER=1
FULL_BOX=1
COMPACT_MODE=0
SPINNER_CHARS='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
SPINNER_LENGTH=${#SPINNER_CHARS}

# Timestamps
START_TIME=$(date +%s)

# Parse command-line arguments
FORCE=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --force) FORCE=true; shift ;;
        *) echo "Unknown parameter: $1"; shift ;;
    esac
done

# Function to get elapsed time in a nice format
get_elapsed_time() {
    local end_time=$(date +%s)
    local elapsed=$((end_time - START_TIME))
    printf "%02d:%02d" $((elapsed / 60)) $((elapsed % 60))
}

# Redirect stderr for silent commands
silent_exec() {
    "$@" >/dev/null 2>&1
}

# Status tracking
declare -A PROCESS_STATUS
TOTAL_PROCESSES=0
SUCCESSFUL_PROCESSES=0

# Function for verbose logging
log_verbose() {
    if [ $VERBOSE -eq 1 ]; then
        echo -e "$1"
    fi
}

# Create a spinner animation
spin() {
    local pid=$1
    local message="$2"
    local index=0
    local spin_char=""

    if [ $USE_SPINNER -eq 1 ]; then
        while kill -0 $pid 2>/dev/null; do
            spin_char="${SPINNER_CHARS:index:1}"
            printf "${CLEAR_LINE}${CYAN}%s${NC} %s" "$spin_char" "$message"
            index=$(( (index + 1) % SPINNER_LENGTH ))
            sleep 0.1
            printf "\r"
        done
        printf "${CLEAR_LINE}"
    fi
}

# Draw a full box with a title
draw_box() {
    local title="$1"
    local width=$MAIN_WIDTH
    local title_len=${#title}
    local padding=$(( (width - title_len - 2) / 2 ))
    
    # Top border with title
    echo -ne "$BOX_TL"
    for ((i=0; i<padding; i++)); do echo -ne "$BOX_H"; done
    echo -ne " ${BOLD}${title}${NC} "
    for ((i=0; i<padding; i++)); do echo -ne "$BOX_H"; done
    # Add one extra dash if width is odd
    if [ $(( (width - title_len) % 2 )) -eq 1 ]; then
        echo -ne "$BOX_H"
    fi
    echo -e "$BOX_TR"
}

# Draw the bottom of a box
draw_box_bottom() {
    local width=$MAIN_WIDTH
    echo -ne "$BOX_BL"
    for ((i=0; i<width-2; i++)); do echo -ne "$BOX_H"; done
    echo -e "$BOX_BR"
}

# Draw a horizontal separator inside a box
draw_separator() {
    local width=$MAIN_WIDTH
    echo -ne "$BOX_VR"
    for ((i=0; i<width-2; i++)); do echo -ne "$BOX_H"; done
    echo -e "$BOX_VL"
}

# Print a line inside a box
print_box_line() {
    local text="$1"
    local right_text="$2"
    echo -ne "$BOX_V "
    
    if [ -n "$right_text" ]; then
        local text_len=${#text}
        local right_len=${#right_text}
        local space_len=$((MAIN_WIDTH - text_len - right_len - 4))
        echo -ne "$text"
        for ((i=0; i<space_len; i++)); do echo -ne " "; done
        echo -ne "$right_text"
    else
        local padded_text=$(printf "%-$((MAIN_WIDTH-4))s" "$text")
        echo -ne "$padded_text"
    fi
    
    echo -e " $BOX_V"
}

# Print a centered line inside a box
print_centered_line() {
    local text="$1"
    local width=$MAIN_WIDTH
    local text_len=${#text}
    local padding=$(( (width - text_len - 4) / 2 ))
    
    echo -ne "$BOX_V "
    for ((i=0; i<padding; i++)); do echo -ne " "; done
    echo -ne "$text"
    for ((i=0; i<padding; i++)); do echo -ne " "; done
    # Add one extra space if width is odd
    if [ $(( (width - text_len) % 2 )) -eq 1 ]; then
        echo -ne " "
    fi
    echo -e " $BOX_V"
}

# Print a header with fancy box
print_header() {
    local text="$1"
    
    if [ $FULL_BOX -eq 1 ]; then
        draw_box "$text"
    else
        echo -e "\n${BOLD}$text${NC}"
        echo -e "${BOLD}$(printf '%*s' ${#text} | tr ' ' '=')${NC}"
    fi
}

# Print a section header
print_section() {
    local title="$1"
    
    if [ $FULL_BOX -eq 1 ]; then
        print_box_line ""
        print_centered_line "${BOLD}${BLUE}$title${NC}"
    else
        echo -e "\n${BLUE}$title${NC}"
    fi
}

# Print a table header for service status
print_status_header() {
    if [ $FULL_BOX -eq 1 ]; then
        print_box_line ""
        print_box_line "${BOLD}SERVICE${NC}${DIM}${BOX_V}${NC} ${BOLD}PORT${NC}${DIM}${BOX_V}${NC} ${BOLD}STATUS${NC}"
        
        # Fancy separator line with table dividers
        echo -ne "$BOX_VR"
        for ((i=0; i<LEFT_COLUMN+1; i++)); do echo -ne "$BOX_H"; done
        echo -ne "$BOX_VH"
        for ((i=0; i<10; i++)); do echo -ne "$BOX_H"; done
        echo -ne "$BOX_VH"
        for ((i=0; i<MAIN_WIDTH-LEFT_COLUMN-15; i++)); do echo -ne "$BOX_H"; done
        echo -e "$BOX_VL"
    fi
}

# Print status message with improved categorization
print_status() {
    local symbol="$1"
    local color="$2"
    local message="$3"
    local is_informational="${4:-false}"
    
    if [ $FULL_BOX -eq 1 ]; then
        print_box_line "${color}${symbol}${NC} ${message}"
    else
        echo -e "${color}${symbol}${NC} ${message}"
    fi
    
    # Update statistics
    TOTAL_PROCESSES=$((TOTAL_PROCESSES+1))
    if [ "$color" = "$GREEN" ]; then
        SUCCESSFUL_PROCESSES=$((SUCCESSFUL_PROCESSES+1))
    elif [ "$is_informational" = "true" ]; then
        # Don't count informational messages as failures
        SUCCESSFUL_PROCESSES=$((SUCCESSFUL_PROCESSES+1))
    fi
}

# Print success message
print_success() {
    print_status "✓" "${GREEN}" "$1"
    PROCESS_STATUS["$1"]="success"
}

# Print error message
print_error() {
    print_status "✗" "${RED}" "$1"
    PROCESS_STATUS["$1"]="error"
}

# Print warning message
print_warning() {
    print_status "!" "${YELLOW}" "$1"
    PROCESS_STATUS["$1"]="warning"
}

# Print info message (now with indicator of informativeness)
print_info() {
    local message="$1"
    local is_informational="${2:-false}"
    
    print_status "i" "${BLUE}" "$message" "$is_informational"
    PROCESS_STATUS["$message"]="info"
}

# Update how we call print_info for notifications that shouldn't count against success
# Use this function throughout the script for informational messages that aren't failures
print_notification() {
    print_info "$1" "true"
}

# Print service status in table format
print_service_status() {
    local name="$1"
    local port="$2"
    local status="$3"
    
    if [ $FULL_BOX -eq 1 ]; then
        local name_col=$(printf "%-${LEFT_COLUMN}s" "$name")
        local port_col=$(printf "%-8s" "$port")
        print_box_line "${name_col}${DIM}${BOX_V}${NC} ${port_col}${DIM}${BOX_V}${NC} ${status}"
    else
        echo -e "${BLUE}▸${NC} $name (Port $port): $status"
    fi
}

# Print a summary of operations
print_summary() {
    if [ $FULL_BOX -eq 1 ]; then
        draw_separator
        print_box_line ""
        print_centered_line "${BOLD}OPERATION SUMMARY${NC}"
        print_box_line ""
        
        local success_rate=$(( (SUCCESSFUL_PROCESSES * 100) / TOTAL_PROCESSES ))
        
        print_box_line "Total Operations:" "$TOTAL_PROCESSES"
        print_box_line "Successful:" "${GREEN}$SUCCESSFUL_PROCESSES${NC}"
        print_box_line "Failed/Warnings:" "${YELLOW}$((TOTAL_PROCESSES - SUCCESSFUL_PROCESSES))${NC}"
        print_box_line "Success Rate:" "${BOLD}${success_rate}%${NC}"
        print_box_line "Elapsed Time:" "$(get_elapsed_time)"
        print_box_line ""
    fi
}

# Function for animated output capture
exec_with_spinner() {
    local message="$1"
    shift
    
    if [ $FULL_BOX -eq 1 ]; then
        print_box_line "${DIM}Executing:${NC} $message..."
    else
        echo "Executing: $message..."
    fi
    
    # Create a unique temporary file for this command
    local temp_output_file=$(mktemp /tmp/sutazai_exec_XXXXXX)
    
    # Execute the command in background and capture its output
    # Using grouped command to preserve exit status
    {
        set -o pipefail
        "$@" > "$temp_output_file" 2>&1
        echo $? > "$temp_output_file.exit"
    } &
    local pid=$!
    
    # Show spinner while the command is running
    if [ $USE_SPINNER -eq 1 ]; then
        local index=0
        local spin_char=""
        
        while kill -0 $pid 2>/dev/null; do
            spin_char="${SPINNER_CHARS:index:1}"
            if [ $FULL_BOX -eq 1 ]; then
                # Draw the spinner directly in the TUI
                echo -ne "\r${BOX_V} ${CYAN}$spin_char${NC} $message"
                printf "%$((MAIN_WIDTH - ${#message} - 5))s${BOX_V}" " "
            else
                printf "\r${CYAN}%s${NC} %s" "$spin_char" "$message"
            fi
            index=$(( (index + 1) % SPINNER_LENGTH ))
            sleep 0.1
        done
        
        # Clear the spinner line
        if [ $FULL_BOX -eq 1 ]; then
            echo -ne "\r${BOX_V}$(printf "%$((MAIN_WIDTH-2))s")${BOX_V}\r"
        else
            printf "\r%-$((${#message} + 10))s\r" " "
        fi
    else
        # Wait for process to complete if not using spinner
        wait $pid
    fi
    
    # Retrieve exit code
    local exit_code=0
    if [ -f "$temp_output_file.exit" ]; then
        exit_code=$(cat "$temp_output_file.exit")
        rm -f "$temp_output_file.exit"
    fi
    
    # Process the output
    if [ -f "$temp_output_file" ]; then
        # Only show output on error or if verbose is enabled
        if [ $exit_code -ne 0 ] || [ $VERBOSE -eq 1 ]; then
            if [ $FULL_BOX -eq 1 ]; then
                print_box_line "${DIM}Output:${NC}"
                # Read output line by line to ensure proper formatting
                while IFS= read -r line || [ -n "$line" ]; do
                    # Truncate long lines to fit in the box
                    if [ ${#line} -gt $((MAIN_WIDTH-6)) ]; then
                        line="${line:0:$((MAIN_WIDTH-10))}..."
                    fi
                    print_box_line "  $line"
                done < "$temp_output_file"
            else
                echo "Output:"
                while IFS= read -r line || [ -n "$line" ]; do
                    echo "  $line"
                done < "$temp_output_file"
            fi
        fi
        
        # Clean up
        rm -f "$temp_output_file"
    fi
    
    return $exit_code
}

# Function to handle terminal control and cleanup
setup_terminal() {
    # Save original terminal settings
    ORIGINAL_STTY=$(stty -g 2>/dev/null || echo "")
    
    # Set strict terminal width to avoid text wrapping
    export TERM=linux
    stty cols 80 2>/dev/null || true
    
    # Hide cursor for cleaner display
    echo -ne "$CURSOR_HIDE"
    
    # Clear screen to start fresh
    echo -ne "$CLEAR_SCREEN"
}

# Handle cleanup on exit
cleanup() {
    # Show cursor again
    echo -ne "$CURSOR_SHOW"
    
    # Restore original terminal settings
    if [ -n "$ORIGINAL_STTY" ]; then
        stty "$ORIGINAL_STTY" 2>/dev/null || true
    fi
    
    # Ensure full cleanup of any temporary files
    rm -f /tmp/sutazai_exec_*
    
    # Ensure we exit cleanly
    echo -e "\n"
}

# Add proper signals and cleanup behaviors
trap cleanup EXIT INT TERM

# Only at the start of the script:
setup_terminal

# Draw main box
print_header "SutazAI System Shutdown"

# Check if we're running in the context of a systemd service
IS_SYSTEMD=false
if [ -n "$INVOCATION_ID" ] || [ -n "$JOURNAL_STREAM" ]; then
    IS_SYSTEMD=true
    print_status "ℹ️" "$BLUE" "Running as a systemd service" "true"
fi

# Call stop_sutazai.sh to ensure a unified shutdown process
print_status "🛑" "$BLUE" "Stopping SutazAI components using stop_sutazai.sh" "true"

# Execute the stop_sutazai.sh script with appropriate flags
STOP_ARGS=""
if [ "$FORCE" = true ]; then
    STOP_ARGS="--force"
    print_status "⚠️" "$YELLOW" "Using force option to stop services" "true"
fi

# Execute the stop_sutazai.sh script and capture its output
if [ $VERBOSE -eq 1 ]; then
    # In verbose mode, show all output from stop_sutazai.sh
    "$PROJECT_ROOT/scripts/stop_sutazai.sh" $STOP_ARGS
    STOP_RESULT=$?
else
    # Otherwise, capture and format the output
    STOP_OUTPUT=$("$PROJECT_ROOT/scripts/stop_sutazai.sh" $STOP_ARGS 2>&1)
    STOP_RESULT=$?
    
    # Display key information from output
    echo "$STOP_OUTPUT" | grep -E "Stopping service:|ERROR:|WARNING:|stopped successfully" | while read -r line; do
        if echo "$line" | grep -q "ERROR:"; then
            print_status "❌" "$RED" "$(echo "$line" | sed 's/\[.*\]//')"
        elif echo "$line" | grep -q "WARNING:"; then
            print_status "⚠️" "$YELLOW" "$(echo "$line" | sed 's/\[.*\]//')"
        elif echo "$line" | grep -q "stopped successfully"; then
            print_status "✅" "$GREEN" "$(echo "$line" | sed 's/\[.*\]//')"
        else
            print_status "🔄" "$BLUE" "$(echo "$line" | sed 's/\[.*\]//')" "true"
        fi
            done
        fi
        
# Check the result
if [ $STOP_RESULT -eq 0 ]; then
    print_status "✅" "$GREEN" "SutazAI stopped successfully"
else
    print_status "⚠️" "$YELLOW" "SutazAI shutdown completed with warnings"
fi

# Cleanup any remaining resources
print_status "🧹" "$BLUE" "Cleaning up remaining resources" "true"

# Kill any leftover processes if force flag is set
if [ "$FORCE" = true ]; then
    print_status "🔄" "$BLUE" "Force stopping any remaining Python processes" "true"
    pkill -f "python.*backend.main" || true
    pkill -f "python.*streamlit" || true
    pkill -f "python.*vector_db" || true
    pkill -f "python.*system_monitor" || true
    pkill -f "python.*cpu_monitor" || true
fi

# Clean up temporary files
print_status "🔄" "$BLUE" "Removing temporary files" "true"
rm -f /tmp/sutazai-startup.lock
rm -f "$PROJECT_ROOT/pids/"*.pid 2>/dev/null || true

# Print final status summary
ELAPSED_TIME=$(get_elapsed_time)
SUCCESS_RATE=$(( (SUCCESSFUL_PROCESSES * 100) / TOTAL_PROCESSES ))

print_section "Shutdown Summary"
print_box_line "Total processes: ${BOLD}$TOTAL_PROCESSES${NC}"
print_box_line "Successful: ${BOLD}$SUCCESSFUL_PROCESSES${NC}"
print_box_line "Success rate: ${BOLD}${SUCCESS_RATE}%${NC}"
print_box_line "Elapsed time: ${BOLD}${ELAPSED_TIME}${NC}"

if [ "$IS_SYSTEMD" = "true" ]; then
    print_box_line "${YELLOW}Service was running under systemd${NC}"
else
    print_box_line "${BLUE}Use 'scripts/start_all.sh' to restart all services${NC}"
fi

# Restore terminal settings
if [ -n "$ORIGINAL_STTY" ]; then
    stty "$ORIGINAL_STTY" 2>/dev/null || true
fi

# Reset cursor
echo -ne "$CURSOR_SHOW"

exit $STOP_RESULT 