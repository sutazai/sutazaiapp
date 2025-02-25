#!/bin/bash
set -e

# Load SutazAi UI Framework
source "$(dirname "$0")/sutazai_framework.sh"

# Initialize AI Core
ai_animation &
ANIM_PID=$!
trap "kill $ANIM_PID" EXIT

echo -e "\n${UI_PALETTE[ai]}${BOLD}🤖 SutazAI Neural Integrity Scan${RESET}"
sutazai_progress 0.02 &
PROGRESS_PID=$!

# System Integrity Verification Script
# Usage: ./system_integrity.sh [generate|verify]

CONFIG_DIR="/etc"
LOG_DIR="/var/log/system_check"
MANIFEST="$LOG_DIR/system_manifest.sha256"
TARGET_BACKUP_DIR="/backup"

# Create log directory if missing
mkdir -p "$LOG_DIR" || {
  echo -e "${UI_PALETTE[danger]}⛔ Failed to create neural log matrix!${RESET}"
  systemctl start sutazai-audit.service
  exit 1
}

# Dependency check
command -v sha256sum >/dev/null || { 
  echo -e "${UI_PALETTE[danger]}⛔ AI Checksum Module Missing!${RESET}"; exit 1; 
}

command -v rsync >/dev/null || { 
  echo -e "${UI_PALETTE[danger]}⛔ Neural Synchronization Engine Offline!${RESET}"; exit 1; 
}

case "$1" in
  generate)
    echo -e "\n${CYAN}Generating system manifest...${RESET}"
    find "$CONFIG_DIR" -type f -exec sha256sum {} \; > "$MANIFEST"
    log_system_event "Manifest Created" "$MANIFEST"
    echo -e "\n${UI_PALETTE[success]}✅ AI-Optimized Manifest Generated:${RESET} $MANIFEST"
    ;;
  verify)
    echo -e "\n${CYAN}Verifying system integrity...${RESET}"
    if [ ! -f "$MANIFEST" ]; then
      echo -e "${RED}No manifest found. Generate first with: $0 generate${RESET}"
      exit 1
    fi
    
    # Verify checksums and find changes
    sha256sum -c "$MANIFEST" > "$LOG_DIR/last_verify.log" 2>&1
    if grep -q "FAILED" "$LOG_DIR/last_verify.log"; then
      echo -e "${UI_PALETTE[danger]}⛔ Neural Anomalies Detected!${RESET}"
      systemctl start sutazai-audit.service
    else
      echo -e "${UI_PALETTE[success]}✅ Neural Continuity Verified${RESET}"
    fi
    
    # Generate change report
    grep "FAILED" "$LOG_DIR/last_verify.log" | awk -F: '{print $1}' > "$LOG_DIR/changed_files.list"
    ;;
  sync)
    echo -e "\n${CYAN}Synchronizing system files...${RESET}"
    sutazai_progress 0.02 &
    PID=$!
    rsync -avh --checksum --delete --log-file="$LOG_DIR/sync_$(date +%F).log" \
      --exclude='*.swp' --exclude='*.bak' \
      "$CONFIG_DIR/" "$TARGET_BACKUP_DIR"
    kill $PID
    wait $PID 2>/dev/null
    echo -e "\n${GREEN}✓ Synchronization complete!${RESET}"
    ;;
  *)
    echo -e "${RED}Usage: $0 {generate|verify|sync}${RESET}"
    exit 1
esac 