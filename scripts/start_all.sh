#!/bin/bash
# title        :start_all.sh
# description  :This script starts all SutazAI components (Backend, Web UI, Vector Store, Monitoring)
# author       :SutazAI Team
# version      :2.0
# usage        :sudo bash scripts/start_all.sh
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

# Parse command-line arguments
DEBUG=false
GPU_MODE=false
SKIP_MODEL_CHECK=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --debug) DEBUG=true; shift ;;
        --gpu) GPU_MODE=true; shift ;;
        --skip-model-check) SKIP_MODEL_CHECK=true; shift ;;
        *) echo "Unknown parameter: $1"; shift ;;
    esac
done

# Pass arguments to start_sutazai.sh
SUTAZAI_ARGS=""
if [ "$DEBUG" = true ]; then
    SUTAZAI_ARGS="$SUTAZAI_ARGS --debug"
fi
if [ "$GPU_MODE" = true ]; then
    SUTAZAI_ARGS="$SUTAZAI_ARGS --gpu"
fi
if [ "$SKIP_MODEL_CHECK" = true ]; then
    SUTAZAI_ARGS="$SUTAZAI_ARGS --skip-model-check"
fi

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

# Timestamps
START_TIME=$(date +%s)

# Print header
print_header "SutazAI System Startup"

# Check if we're running in the context of a systemd service
IS_SYSTEMD=false
if [ -n "$INVOCATION_ID" ] || [ -n "$JOURNAL_STREAM" ]; then
    IS_SYSTEMD=true
    print_status "ℹ️" "$BLUE" "Running as a systemd service" "true"
fi

# Call start_sutazai.sh to ensure a unified startup process
print_status "🚀" "$BLUE" "Starting SutazAI components using start_sutazai.sh" "true"

# Execute the start_sutazai.sh script and capture its output
if [ $VERBOSE -eq 1 ]; then
    # In verbose mode, show all output from start_sutazai.sh
    "$PROJECT_ROOT/scripts/start_sutazai.sh" $SUTAZAI_ARGS
    START_RESULT=$?
else
    # Otherwise, capture and format the output
    START_OUTPUT=$("$PROJECT_ROOT/scripts/start_sutazai.sh" $SUTAZAI_ARGS 2>&1)
    START_RESULT=$?
    
    # Display key information from output
    echo "$START_OUTPUT" | grep -E "Starting service:|ERROR:|WARNING:|ready on port" | while read -r line; do
        if echo "$line" | grep -q "ERROR:"; then
            print_status "❌" "$RED" "$(echo "$line" | sed 's/\[.*\]//')"
        elif echo "$line" | grep -q "WARNING:"; then
            print_status "⚠️" "$YELLOW" "$(echo "$line" | sed 's/\[.*\]//')"
        elif echo "$line" | grep -q "ready on port"; then
            print_status "✅" "$GREEN" "$(echo "$line" | sed 's/\[.*\]//')"
        else
            print_status "🔄" "$BLUE" "$(echo "$line" | sed 's/\[.*\]//')" "true"
        fi
    done
fi

# Check the result
if [ $START_RESULT -eq 0 ]; then
    print_status "✅" "$GREEN" "SutazAI started successfully"
else
    print_status "❌" "$RED" "Failed to start SutazAI system. Check logs for details."
fi

# Print final status summary
ELAPSED_TIME=$(get_elapsed_time)
SUCCESS_RATE=$(( (SUCCESSFUL_PROCESSES * 100) / TOTAL_PROCESSES ))

print_section "Startup Summary"
print_box_line "Total processes: ${BOLD}$TOTAL_PROCESSES${NC}"
print_box_line "Successful: ${BOLD}$SUCCESSFUL_PROCESSES${NC}"
print_box_line "Success rate: ${BOLD}${SUCCESS_RATE}%${NC}"
print_box_line "Elapsed time: ${BOLD}${ELAPSED_TIME}${NC}"

if [ "$IS_SYSTEMD" = "true" ]; then
    print_box_line "${YELLOW}Running as systemd service - use 'systemctl' commands to manage${NC}"
else
    print_box_line "${BLUE}Use 'scripts/stop_all.sh' to stop all services${NC}"
fi

# Restore terminal settings
    if [ -n "$ORIGINAL_STTY" ]; then
        stty "$ORIGINAL_STTY" 2>/dev/null || true
    fi
    
# Reset cursor
echo -ne "$CURSOR_SHOW"

exit $START_RESULT