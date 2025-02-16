#!/bin/bash
set -e
set -euo pipefail  # Fail on error
trap "echo 'Deployment failed'; exit 1" ERR

# Source deployment utilities
source "$(dirname "${BASH_SOURCE[0]}")/deploy_utils.sh"

# Terminal colors using tput
RED=$(tput setaf 1)
GREEN=$(tput setaf 2)
CYAN=$(tput setaf 6)
BLUE=$(tput setaf 4)
RESET=$(tput sgr0)
BOLD=$(tput bold)

# Large ASCII Art Logo
show_logo() {
  clear
  echo "${RED}"
  cat << "EOF"
   _____       _          _   _  ___ 
  / ____|     | |        | | | |/ _ \
 | (___   ___ | |_ _   _ | | | | (_) |
  \___ \ / _ \| __| | | || | | |\__, |
  ____) | (_) | |_| |_| || |_| |  / / 
 |_____/ \___/ \__|\__,_| \___/  /_/  
EOF
  echo "${RESET}"
}

# Static logo positioning
freeze_logo() {
  tput sc
  tput cup 0 0
  show_logo
  tput rc
}

# Three-line status display
status_panel() {
  local line1=$1
  local line2=$2
  local line3=$3
  
  tput sc
  tput cup 8 0
  printf "\033[K\n\033[K\n\033[K" # Clear existing lines
  echo -e "${CYAN}${line1}${RESET}"
  echo -e "${BLUE}${line2}${RESET}"
  echo -e "${GREEN}${line3}${RESET}"
  tput rc
}

# Modern progress bar with frozen logo
modern_progress_bar() {
  local total_steps=$1
  local current_step=$2
  local label1=$3
  local label2=$4
  local label3=$5
  
  freeze_logo

  # Calculate progress
  local percentage=$((current_step * 100 / total_steps))
  local bar_length=50
  local filled=$((bar_length * current_step / total_steps))
  local empty=$((bar_length - filled))

  # Build progress bar
  local bar="["
  bar+="${GREEN}"
  bar+=$(printf "%${filled}s" | tr ' ' '█')
  bar+="${RESET}"
  bar+=$(printf "%${empty}s" | tr ' ' ' ')
  bar+="]"

  # Display three-line status
  status_panel \
    "STEP ${current_step}/${total_steps}: ${label1}" \
    "${label2}" \
    "${label3:-}"

  printf "\r%s %3d%% - %s" "$bar" "$percentage" "$label1"
}

# Main deployment process
deploy() {
  init_logging
  show_logo
  freeze_logo
  
  echo -e "\n${BLUE}${BOLD}Deploying your infinite possibilities SutazAi!${RESET}"
  echo -e "${CYAN}Sit back, relax, and enjoy the show...${RESET}\n"

  # Add version check
  if ! check_version; then
    echo "Version mismatch detected"
    exit 1
  fi

  # Deployment steps
  modern_progress_bar 5 1 "Validating System Requirements"
  validate_system_requirements
  
  modern_progress_bar 5 2 "Creating Virtual Environment"
  create_virtualenv
  
  modern_progress_bar 5 3 "Installing Dependencies"
  install_dependencies "$(dirname "${BASH_SOURCE[0]}")/requirements.txt"
  
  modern_progress_bar 5 4 "Configuring AI Models"
  python3 "$(dirname "${BASH_SOURCE[0]}")/ai/setup.py" --config "$(dirname "${BASH_SOURCE[0]}")/config/ai_config.yaml"
  
  modern_progress_bar 5 5 "Starting Services"
  systemctl start ai-core
  systemctl start ai-worker
  
  echo -e "\n${GREEN}${BOLD}✓ Deployment Completed!${RESET}"
  echo -e "${CYAN}Welcome to SutazAi!${RESET}"
  echo -e "\n${BOLD}Access your instance at:${RESET}"
  echo -e "Web Interface: ${BLUE}https://sutaz.ai/dashboard${RESET}"
  echo -e "API Endpoint:  ${BLUE}https://api.sutaz.ai/v1${RESET}"
  echo -e "\n${GREEN}System ready for AI-powered innovation!${RESET}"
}

# Run deployment with logging
deploy 2>&1 | tee /var/log/sutazai_deploy.log

# Deploy the UI components
echo "Deploying UI components..."
deploy_ui_components() {
    echo "Deploying UI components..."
    # Add actual deployment logic here
}

# Add verification step
echo "Verifying UI deployment..."
verify_ui_deployment() {
    echo "Verifying UI deployment..."
    # Add actual verification logic here
}

# Deploy the UI components
echo "Deploying UI components..."
if ! deploy_ui_components; then
    echo "${RED}UI deployment failed!${RESET}"
    exit 1
fi

# Add verification step
echo "Verifying UI deployment..."
if ! verify_ui_deployment; then
    echo "${RED}UI verification failed!${RESET}"
    exit 1
fi

echo "${GREEN}UI deployment successful!${RESET}"

DEPLOY_DIR=${DEPLOY_DIR:-"/var/www/sutazai"}
LOG_DIR=${LOG_DIR:-"/var/log/sutazai"}

# Example fix: Adding error handling for dependency installation
if ! npm install; then
    echo "Failed to install dependencies"
    exit 1
fi