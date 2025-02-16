#!/bin/bash
set -eo pipefail

verify_deployment() {
  declare -A CHECKS=(
    ["CPU Utilization"]="top -bn1 | grep 'Cpu(s)'"
    ["Memory Usage"]="free -m | awk '/Mem:/ {print \$3\"MB\"}'"
    ["Service Status"]="systemctl status sutazai-core"
    ["Log Health"]="grep -i 'error\|fail' /var/log/sutazai/*.log"
  )

  for check in "${!CHECKS[@]}"; do
    if eval "${CHECKS[$check]}" >/dev/null 2>&1; then
      echo -e "${GREEN}✓ $check: PASS${RESET}"
    else
      echo -e "${RED}✗ $check: FAIL${RESET}"
    fi
  done
}

generate_report() {
  echo -e "\n${CYAN}=== Deployment Health Report ===${RESET}"
  echo -e "Deployment Mode: ${BOLD}$DEPLOY_MODE${RESET}"
  echo -e "System Load: $(uptime)"
  echo -e "Storage Free: $(df -h / | awk 'NR==2 {print $4}')"
  verify_deployment
} 